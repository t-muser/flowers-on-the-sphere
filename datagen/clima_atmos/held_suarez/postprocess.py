"""Post-process a ClimaAtmos Held-Suarez run directory into our HS-3D Zarr.

What ClimaAtmos writes (one NetCDF per variable, under
``<out_dir>/clima/output_NNNN/``):

  - ``ua_6h_inst_pressure.nc``, ``va_...``, ``ta_...``
      Dims: ``(pressure_level, lat, lon, time)``. Pressure axis carries
      37 default levels in Pa (e.g. 100, 200, 300, 500, 700, 1000,
      2000, 3000, **5000, 10000, 25000, 50000, 70000, 85000, 92500,
      100000**, ...). The bolded subset is exactly the 8 ERA5 levels
      our consumer expects.

  - ``pfull_6h_inst.nc``
      Dims: ``(z, lat, lon, time)`` on the native model height axis
      (~100 m bottom level to z_max top). Used to derive surface
      pressure as ``pfull[z=lowest, ...]``.

What we emit (matches ``fots.data.held_suarez.HeldSuarezDataModule``):

  - ``run.zarr`` with ``u, v, T (time, level, lat, lon)`` on the 8 ERA5
    pressure levels + ``ps (time, lat, lon)``. Levels in hPa, lat/lon
    in degrees, time in seconds since simulation start.

This module does **not** do any interpolation in pressure — the 8 ERA5
levels are already a subset of ClimaAtmos's 37-level default. If a
future ClimaAtmos release drops one of those exact levels, this will
error loudly and a cubic-spline path can be re-added.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from datagen.resample import write_latlon_zarr_3d

log = logging.getLogger(__name__)


# Locked to ZarrDataModule consumer ordering (stats.json key order).
ERA5_LEVELS_HPA: tuple[int, ...] = (50, 100, 250, 500, 700, 850, 925, 1000)
ERA5_LEVELS_PA = tuple(L * 100 for L in ERA5_LEVELS_HPA)

_VAR_RENAME: dict[str, str] = {"ua": "u", "va": "v", "ta": "T"}


def _find_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
    lat = next((c for c in ("lat", "latitude") if c in ds.coords), None)
    lon = next((c for c in ("lon", "longitude") if c in ds.coords), None)
    if lat is None or lon is None:
        raise ValueError(
            f"Could not locate lat/lon coords; have {list(ds.coords)}"
        )
    return lat, lon


def _resolve_pressure_indices(level_axis_pa: np.ndarray) -> list[int]:
    """For each ERA5 level, find the index of the matching pressure
    value in ``level_axis_pa`` (tolerance 0.1 Pa). Fail loud if any
    level is missing — that's a regression we want to catch immediately.
    """
    idx = []
    for target_pa in ERA5_LEVELS_PA:
        hits = np.argwhere(np.abs(level_axis_pa - target_pa) < 0.1).ravel()
        if hits.size == 0:
            raise ValueError(
                f"ERA5 level {target_pa} Pa not present in ClimaAtmos "
                f"output's pressure axis {list(level_axis_pa)}"
            )
        idx.append(int(hits[0]))
    return idx


def _read_3d_pressure(nc_path: Path, var: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(arr_on_era5, era5_pa)``.

    ``arr_on_era5`` has shape ``(time, 8, lat, lon)`` — picked directly
    from the 37-level pressure axis with no interpolation.
    """
    ds = xr.open_dataset(str(nc_path), decode_times=False)
    if var not in ds.data_vars:
        raise KeyError(f"{nc_path} does not contain variable {var!r}")
    if "pressure_level" not in ds.coords:
        raise KeyError(
            f"{nc_path} has no pressure_level coord — expected pressure-coord output"
        )
    lat, lon = _find_lat_lon(ds)
    # Transpose to (time, pressure_level, lat, lon) regardless of input order.
    da = ds[var].transpose("time", "pressure_level", lat, lon)
    level_pa = ds["pressure_level"].values.astype(np.float64)
    idx = _resolve_pressure_indices(level_pa)
    arr = da.isel(pressure_level=idx).values.astype(np.float32)
    return arr, np.asarray(ERA5_LEVELS_PA, dtype=np.float64)


def _read_ps_from_pfull(nc_path: Path) -> np.ndarray:
    """Return ``ps`` with shape ``(time, lat, lon)`` — taken as the
    lowest-z slab of pfull. ClimaAtmos's z axis is ascending (bottom-up
    when the values are heights in metres), so index 0 is the bottom.
    """
    ds = xr.open_dataset(str(nc_path), decode_times=False)
    if "pfull" not in ds.data_vars:
        raise KeyError(f"{nc_path} does not contain pfull")
    if "z" not in ds.coords:
        raise KeyError(
            f"{nc_path} has no z coord — expected non-pressure-coord pfull"
        )
    lat, lon = _find_lat_lon(ds)
    da = ds["pfull"].transpose("time", "z", lat, lon)
    z = ds["z"].values
    # z is ascending in metres; bottom-most level = surface
    bottom_idx = int(np.argmin(z))
    ps = da.isel(z=bottom_idx).values.astype(np.float32)
    return ps


def _open_run(nc_paths: list[Path]) -> xr.Dataset:
    """Backwards-compat: open one or many NetCDFs and merge along time.

    Currently unused on the ClimaAtmos production path (we open files
    by variable in ``postprocess``); kept for the synthetic-fixture
    test that still uses a single multi-var NetCDF.
    """
    if not nc_paths:
        raise FileNotFoundError("No NetCDF inputs found")
    if len(nc_paths) == 1:
        return xr.open_dataset(str(nc_paths[0]), decode_times=False)
    return xr.open_mfdataset(
        [str(p) for p in nc_paths],
        combine="by_coords",
        decode_times=False,
    )


def _find_clima_files(clima_dir: Path) -> dict[str, Path]:
    """Locate the per-variable NetCDFs ClimaAtmos wrote under
    ``clima_dir`` (which contains one or more ``output_NNNN/``
    directories — we pick the latest by sort order).
    """
    if (clima_dir / "output_active").is_dir():
        out_dir = clima_dir / "output_active"
    else:
        candidates = sorted(clima_dir.glob("output_*"))
        if not candidates:
            raise FileNotFoundError(f"No output_NNNN under {clima_dir}")
        out_dir = candidates[-1]
    files: dict[str, Path] = {}
    for clima_var in ("ua", "va", "ta"):
        ncs = list(out_dir.glob(f"{clima_var}_*_pressure.nc"))
        if not ncs:
            raise FileNotFoundError(
                f"missing {clima_var} pressure-coord NetCDF under {out_dir}"
            )
        files[clima_var] = ncs[0]
    # pfull (non-pressure-coord) for ps derivation. Prefer 6h_inst over
    # 1d_average to keep cadence aligned with the 3D fields.
    pfull_ncs = sorted(out_dir.glob("pfull_*.nc"))
    pfull_z = [p for p in pfull_ncs if "pressure" not in p.name]
    if not pfull_z:
        raise FileNotFoundError(
            f"missing non-pressure pfull NetCDF under {out_dir} — add "
            "a `pfull` diagnostic without `pressure_coordinates: true` "
            "alongside ua/va/ta"
        )
    # Prefer the highest-cadence one (filenames sort lexically: 1d_ < 6h_).
    files["pfull"] = sorted(pfull_z, key=lambda p: ("inst" not in p.name, p.name))[0]
    return files


def postprocess_clima_dir(
    clima_dir: Path,
    out_path: Path,
    *,
    run_id: int | None,
    params: dict | None,
    description: str | None = None,
) -> None:
    """Read a ClimaAtmos run output directory and write our HS-3D Zarr."""
    files = _find_clima_files(clima_dir)
    log.info("Resolved files: %s", {k: str(v) for k, v in files.items()})

    fields_3d: dict[str, np.ndarray] = {}
    level_arr: np.ndarray | None = None
    for clima_var, hs_var in _VAR_RENAME.items():
        arr, level_arr_v = _read_3d_pressure(files[clima_var], clima_var)
        fields_3d[hs_var] = arr
        level_arr = level_arr_v

    ps = _read_ps_from_pfull(files["pfull"])
    fields_2d: dict[str, np.ndarray] = {"ps": ps}

    # If we are snapshotting an in-flight run, Julia may have written one
    # extra timestep to some files but not others between our cp calls.
    # Truncate every field to the shortest common time axis so xr.Dataset
    # can assemble them.
    n_time = min(
        *(arr.shape[0] for arr in fields_3d.values()),
        *(arr.shape[0] for arr in fields_2d.values()),
    )
    for k in fields_3d:
        if fields_3d[k].shape[0] > n_time:
            log.info("Truncating %s from %d to %d timesteps", k, fields_3d[k].shape[0], n_time)
            fields_3d[k] = fields_3d[k][:n_time]
    for k in fields_2d:
        if fields_2d[k].shape[0] > n_time:
            log.info("Truncating %s from %d to %d timesteps", k, fields_2d[k].shape[0], n_time)
            fields_2d[k] = fields_2d[k][:n_time]

    # Pull lat/lon/time from one of the 3D files (they all share these
    # coords in ClimaAtmos's diagnostic output). Time also gets trimmed
    # to n_time for the same reason.
    ds_ref = xr.open_dataset(str(files["ua"]), decode_times=False)
    lat_name, lon_name = _find_lat_lon(ds_ref)
    lat_deg = ds_ref[lat_name].values.astype(np.float64)
    lon_deg = ds_ref[lon_name].values.astype(np.float64)
    time_arr = ds_ref["time"].values.astype(np.float64)[:n_time]
    lat_target = np.deg2rad(lat_deg)
    lon_target = np.deg2rad(lon_deg)

    # Lat in degrees ascending, lon centered around 0 by default in
    # ClimaAtmos. The downstream HS DataModule does not require [0,360);
    # it only checks dims and attrs.
    level_hpa = np.asarray(ERA5_LEVELS_HPA, dtype=np.float64)

    if description is None:
        description = (
            "ClimaAtmos.jl spectral-element Held-Suarez run, "
            "remapped to regular (lat, lon) on ClimaAtmos's pressure "
            "axis (8 ERA5 levels: a subset of the default 37-level grid)."
        )

    write_latlon_zarr_3d(
        out_path,
        time_arr=time_arr,
        fields_3d=fields_3d,
        fields_2d=fields_2d,
        lat_target=lat_target,
        lon_target=lon_target,
        level_hpa=level_hpa,
        description=description,
        run_id=run_id,
        params=params,
    )
    log.info(
        "Wrote Zarr %s  time=%d  level=%d  lat=%d  lon=%d",
        out_path, time_arr.size, level_hpa.size, lat_deg.size, lon_deg.size,
    )


# ─── Legacy synthetic-fixture path (kept for tests) ──────────────────────────

def postprocess(
    inputs: list[Path],
    out_path: Path,
    *,
    run_id: int | None,
    params: dict | None,
    target_levels_hpa: tuple[int, ...] = ERA5_LEVELS_HPA,
    description: str | None = None,
) -> None:
    """Synthetic-fixture path: open one merged NetCDF, vertically
    interpolate ua/va/ta from a 1-D pressure axis (``pfull`` in Pa or
    ``pressure_level`` in Pa) to the 8 ERA5 levels via cubic splines,
    extract surface pressure if a ``ps``/``psurf`` var is present.

    This path is what ``tests/test_postprocess.py`` exercises — its
    fixtures predate the per-variable ClimaAtmos layout. The real
    production path is :func:`postprocess_clima_dir`.
    """
    from scipy.interpolate import CubicSpline

    ds = _open_run(inputs)

    lat_name, lon_name = _find_lat_lon(ds)
    p_name = next(
        (c for c in ("pressure_level", "pfull", "plev", "pressure", "lev")
         if c in ds.coords or c in ds.dims),
        None,
    )
    if p_name is None:
        raise ValueError(
            f"Could not locate pressure axis on dataset; coords={list(ds.coords)}"
        )

    p_native_pa = ds[p_name].values.astype(np.float64)
    if p_native_pa.max() < 2000.0:
        log.warning(
            "Pressure axis max=%.1f looks like hPa, not Pa; multiplying by 100",
            p_native_pa.max(),
        )
        p_native_pa = p_native_pa * 100.0

    target_pa = np.asarray(list(target_levels_hpa), dtype=np.float64) * 100.0
    log_p_native = np.log(p_native_pa)
    log_p_target = np.log(target_pa)
    order = np.argsort(log_p_native)
    log_p_sorted = log_p_native[order]

    fields_3d: dict[str, np.ndarray] = {}
    for clima_name, hs_name in _VAR_RENAME.items():
        if clima_name not in ds.data_vars:
            raise KeyError(
                f"missing {clima_name!r}; have {list(ds.data_vars)}"
            )
        da = ds[clima_name].transpose("time", p_name, lat_name, lon_name)
        arr = da.values
        arr_sorted = np.take(arr, order, axis=1)
        spline = CubicSpline(log_p_sorted, arr_sorted, axis=1, extrapolate=False)
        out = spline(log_p_target).astype(np.float32)
        if not np.all(np.isfinite(out)):
            raise ValueError(f"NaN in {clima_name} after pressure interpolation")
        fields_3d[hs_name] = out

    ps_name = next(
        (c for c in ("ps", "psurf", "p_surface", "sfcps") if c in ds.data_vars),
        None,
    )
    if ps_name is not None:
        ps = ds[ps_name].transpose("time", lat_name, lon_name).values.astype(np.float32)
    elif "pfull" in ds.data_vars:
        log.info("No surface-pressure diagnostic; deriving ps from pfull[lowest level]")
        pfull = ds["pfull"].transpose("time", p_name, lat_name, lon_name).values
        bottom_idx = int(np.argmax(p_native_pa))
        ps = pfull[:, bottom_idx, :, :].astype(np.float32)
    else:
        raise KeyError(
            f"No ps and no pfull in {list(ds.data_vars)}"
        )

    lat_deg = ds[lat_name].values.astype(np.float64)
    lon_deg = ds[lon_name].values.astype(np.float64)
    time_arr = ds["time"].values.astype(np.float64)

    write_latlon_zarr_3d(
        out_path,
        time_arr=time_arr,
        fields_3d=fields_3d,
        fields_2d={"ps": ps},
        lat_target=np.deg2rad(lat_deg),
        lon_target=np.deg2rad(lon_deg),
        level_hpa=np.asarray(target_levels_hpa, dtype=np.float64),
        description=description or "synthetic ClimaAtmos-shaped HS-3D Zarr",
        run_id=run_id,
        params=params,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--clima-dir", type=Path,
        help="ClimaAtmos run directory (containing output_NNNN/ or output_active/).",
    )
    src.add_argument(
        "--input", nargs="+", type=Path,
        help="Legacy: one or more pre-merged NetCDFs (synthetic-fixture path).",
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="Output Zarr store path (e.g. <out_dir>/run.zarr).",
    )
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--params-json", type=Path, default=None)
    ap.add_argument("--description", default=None)
    args = ap.parse_args()
    _setup_logging()

    params: dict | None = None
    if args.params_json is not None:
        with open(args.params_json, encoding="utf-8") as f:
            entry = json.load(f)
        params = entry.get("params")

    if args.clima_dir is not None:
        postprocess_clima_dir(
            clima_dir=args.clima_dir,
            out_path=args.output,
            run_id=args.run_id,
            params=params,
            description=args.description,
        )
    else:
        postprocess(
            inputs=args.input,
            out_path=args.output,
            run_id=args.run_id,
            params=params,
            description=args.description,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
