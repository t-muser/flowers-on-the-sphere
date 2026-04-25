"""Resample Dedalus native-grid snapshots onto a regular ``(lat, lon)`` grid.

The native Dedalus ``SphereBasis`` grid is equispaced in longitude ``phi`` and
at Gauss-Legendre nodes in colatitude ``theta``. The output benchmark grid is
equispaced in both latitude and longitude.

When the native ``Nphi`` matches the output ``Nlon`` and the native ``Ntheta``
matches or oversamples the output ``Nlat``, the zonal axis is copied
element-wise and only the meridional axis needs interpolation. We use a
per-longitude cubic spline in colatitude, which is effectively exact for
bandlimited fields sampled at Gauss-Legendre nodes below the Nyquist cutoff.

The set of output fields is configurable via the ``field_specs`` argument.
Each entry maps an output name to a ``(task_name, component)`` pair, where
``component`` is ``None`` for scalar tasks and an integer index for
vector-valued tasks. See ``GALEWSKY_FIELD_SPECS`` and ``MICKELIN_FIELD_SPECS``
for the two current datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline

from datagen.galewsky._units import METER, SECOND


FieldSpec = Tuple[str, Optional[int]]


GALEWSKY_FIELD_SPECS: Dict[str, FieldSpec] = {
    "u_phi": ("u", 0),
    "u_theta": ("u", 1),
    "h": ("h", None),
    "vorticity": ("vorticity", None),
}


MICKELIN_FIELD_SPECS: Dict[str, FieldSpec] = {
    "vorticity": ("vorticity", None),
}


# Multiplier to convert sim-unit field values back to physical SI units.
# Velocities:  u_sim = u_phys · METER/SECOND   ⇒  u_phys = u_sim · SECOND/METER
# Heights:     h_sim = h_phys · METER          ⇒  h_phys = h_sim / METER
# Vorticity:   ω_sim = ω_phys / SECOND         ⇒  ω_phys = ω_sim · SECOND
GALEWSKY_FIELD_TO_PHYS: Dict[str, float] = {
    "u_phi":     SECOND / METER,
    "u_theta":   SECOND / METER,
    "h":         1.0 / METER,
    "vorticity": SECOND,
}


def regular_lat_grid(Nlat: int) -> np.ndarray:
    """Equispaced latitudes in radians, excluding the poles."""
    return -np.pi / 2.0 + (np.arange(Nlat) + 0.5) * np.pi / Nlat


def regular_lon_grid(Nlon: int) -> np.ndarray:
    """Equispaced longitudes in radians on ``[0, 2π)``."""
    return np.arange(Nlon) * 2.0 * np.pi / Nlon


def _interp_theta(
    data: np.ndarray,
    theta_native: np.ndarray,
    theta_target: np.ndarray,
    theta_axis: int,
) -> np.ndarray:
    """Cubic-spline interpolation along ``theta_axis`` from native to target nodes."""
    order = np.argsort(theta_native)
    theta_sorted = theta_native[order]
    data_sorted = np.take(data, order, axis=theta_axis)
    spline = CubicSpline(theta_sorted, data_sorted, axis=theta_axis)
    return spline(theta_target)


def _read_task(
    tasks_group: h5py.Group, task_name: str, component: Optional[int]
) -> np.ndarray:
    """Read an HDF5 Dedalus task as a ``(Nt, Nphi, Ntheta)`` array."""
    arr = tasks_group[task_name][:]
    if component is None:
        return arr
    return arr[:, component]


def _resolve_phys_scales(
    field_specs: Mapping[str, FieldSpec],
    field_phys_scales: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """Fill in sim→phys conversion factors, defaulting to Galewsky scales
    for known field names and 1.0 otherwise."""
    if field_phys_scales is not None:
        return {name: float(field_phys_scales.get(name, 1.0)) for name in field_specs}
    return {name: GALEWSKY_FIELD_TO_PHYS.get(name, 1.0) for name in field_specs}


def resample_run(
    raw_dir: Path,
    out_path: Path,
    Nlat: int = 256,
    Nlon: int = 512,
    time_offset_s: float = 0.0,
    time_offset_tol: float = 1.0,
    params: dict | None = None,
    run_id: int | None = None,
    field_specs: Mapping[str, FieldSpec] | None = None,
    field_phys_scales: Mapping[str, float] | None = None,
    description: str | None = None,
) -> None:
    """Read all Dedalus HDF5 snapshots in ``raw_dir`` and write a Zarr store.

    Fields are resampled from the native Gauss-Legendre colatitude grid to a
    regular ``(lat, lon)`` grid via per-longitude cubic-spline interpolation.
    The output Zarr has dims ``(time, field, lat, lon)`` with ``field``
    corresponding to the keys of ``field_specs`` (default: Galewsky layout
    ``{u_phi, u_theta, h, vorticity}``). Values are stored as ``float32``.

    The Dedalus solver runs in sim units (see ``datagen.galewsky._units``); this
    function converts ``sim_time`` and every field back to physical SI
    units before writing. Use ``field_phys_scales`` to override the
    default sim→phys multipliers (``GALEWSKY_FIELD_TO_PHYS``); unknown
    fields fall back to 1.0 (no conversion).

    ``time_offset_s`` drops all snapshots with sim-time below the given
    threshold (in physical seconds) and rebases the remaining ``time``
    coordinate to start at 0 (useful for discarding the linear spinup
    phase of a run).
    """
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)

    if field_specs is None:
        field_specs = GALEWSKY_FIELD_SPECS
    field_names = list(field_specs.keys())
    phys_scales = _resolve_phys_scales(field_specs, field_phys_scales)

    h5_files = sorted(raw_dir.glob("*_s*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No Dedalus snapshot files under {raw_dir}")

    lat_target = regular_lat_grid(Nlat)
    lon_target = regular_lon_grid(Nlon)
    theta_target = np.pi / 2.0 - lat_target

    times_all: list[np.ndarray] = []
    data_all: dict[str, list[np.ndarray]] = {name: [] for name in field_names}

    theta_native: np.ndarray | None = None
    phi_native: np.ndarray | None = None

    for fpath in h5_files:
        with h5py.File(fpath, mode="r") as f:
            tasks = f["tasks"]
            sim_time = f["scales/sim_time"][:]
            if theta_native is None:
                # Use the first listed task to discover the native grid.
                # Dim ordering is ``(time, [component,] phi, theta)`` so
                # the last two axes are always (phi, theta) regardless of
                # tensor rank. h5py rejects negative dim indices, hence
                # the explicit positive computation.
                first_task = tasks[list(field_specs.values())[0][0]]
                ndim = first_task.ndim
                theta_native = first_task.dims[ndim - 1][0][:]
                phi_native = first_task.dims[ndim - 2][0][:]
            raw_fields = {
                name: _read_task(tasks, task_name, component)
                for name, (task_name, component) in field_specs.items()
            }

        if phi_native.size != Nlon:
            raise ValueError(
                f"Native Nphi={phi_native.size} does not match output Nlon={Nlon}; "
                "longitudinal resampling is not implemented."
            )

        def _remap(field: np.ndarray) -> np.ndarray:
            # field shape: (Nt, Nphi, Ntheta)
            out = _interp_theta(field, theta_native, theta_target, theta_axis=-1)
            return np.ascontiguousarray(np.moveaxis(out, 1, 2))

        for name, arr in raw_fields.items():
            data_all[name].append(_remap(arr))
        times_all.append(sim_time)

    # Dedalus writes ``scales/sim_time`` and field values in the solver's
    # sim units (1 time unit = 1 hour, R_earth = 1). Convert everything back
    # to physical SI here so downstream consumers and the ``time_offset_s``
    # filter see plain seconds / m / m·s⁻¹ / s⁻¹.
    time_arr = np.concatenate(times_all) / SECOND
    arrays = {
        name: (np.concatenate(data_all[name], axis=0)
               * phys_scales[name]).astype(np.float32)
        for name in field_names
    }

    if time_offset_s > 0.0:
        # Keep snapshots at or after the cutoff (with a small tolerance for
        # CFL-induced timestep jitter near the boundary) and rebase to t=0.
        mask = time_arr >= (time_offset_s - time_offset_tol)
        if not mask.any():
            raise ValueError(
                f"time_offset_s={time_offset_s} discards every snapshot "
                f"(max time={time_arr.max()})."
            )
        time_arr = time_arr[mask] - time_offset_s
        arrays = {name: arr[mask] for name, arr in arrays.items()}

    if description is None:
        description = "Dedalus sphere snapshot, resampled to regular (lat, lon)."

    _write_latlon_zarr(
        out_path,
        time_arr=time_arr,
        field_arrays=[arrays[name] for name in field_names],
        field_names=field_names,
        lat_target=lat_target,
        lon_target=lon_target,
        description=description,
        run_id=run_id,
        params=params,
    )


def _write_latlon_zarr(
    out_path: Path,
    time_arr: np.ndarray,
    field_arrays: Sequence[np.ndarray],
    field_names: Sequence[str],
    lat_target: np.ndarray,
    lon_target: np.ndarray,
    *,
    description: str,
    run_id: int | None,
    params: dict | None,
    time_units: str = "seconds since simulation start",
) -> None:
    """Write a stack of ``(Nt, Nlat, Nlon)`` field arrays to a Zarr store.

    ``field_arrays`` and ``field_names`` are expected to be aligned:
    ``field_arrays[i]`` corresponds to ``field_names[i]``. ``lat_target``
    and ``lon_target`` are in radians and are converted to degrees here
    for the on-disk coordinates.
    """
    out_path = Path(out_path)
    fields = np.stack(list(field_arrays), axis=1)
    Nlat = lat_target.size
    Nlon = lon_target.size

    ds = xr.Dataset(
        data_vars={
            "fields": (("time", "field", "lat", "lon"), fields),
        },
        coords={
            "time": ("time", time_arr.astype(np.float64)),
            "field": ("field", list(field_names)),
            "lat": ("lat", np.rad2deg(lat_target).astype(np.float64)),
            "lon": ("lon", np.rad2deg(lon_target).astype(np.float64)),
        },
        attrs={
            "description": description,
            "time_units": time_units,
            "lat_units": "degrees_north",
            "lon_units": "degrees_east",
            "run_id": run_id if run_id is not None else -1,
            **({f"param_{k}": v for k, v in params.items()} if params else {}),
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"fields": {"chunks": (1, len(field_names), Nlat, Nlon)}}
    ds.to_zarr(str(out_path), mode="w", consolidated=True, encoding=encoding)
