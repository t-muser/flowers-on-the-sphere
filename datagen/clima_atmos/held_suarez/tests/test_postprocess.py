"""Schema test for the ClimaAtmos → HS-3D Zarr post-processor.

Builds a tiny synthetic ClimaAtmos-shaped NetCDF (ua/va/ta on a ``pfull``
pressure axis, plus a surface ``ps``), runs ``postprocess``, then checks:

  - the Zarr has the exact dims/vars/coords the consumer expects;
  - lat/lon are stored in degrees;
  - time is in seconds since simulation start;
  - vertical interpolation hits the 8 ERA5 levels;
  - ``HeldSuarezDataModule._stack_fields`` consumes the result cleanly.

No Julia / ClimaAtmos needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datagen.clima_atmos.held_suarez.postprocess import (
    ERA5_LEVELS_HPA,
    ERA5_LEVELS_PA,
    postprocess,
    postprocess_clima_dir,
)


def _make_synthetic_clima_nc(
    nc_path: Path,
    *,
    Nt: int = 3,
    Nlev: int = 12,
    Nlat: int = 16,
    Nlon: int = 32,
) -> None:
    """Write a minimal ClimaAtmos-shaped NetCDF.

    Pressure axis is in pascals, spanning ~25 hPa at top to ~1010 hPa at
    bottom so the 8 ERA5 target levels all sit inside the native range
    and the cubic-spline interpolation succeeds.
    """
    rng = np.random.default_rng(0)
    pfull_pa = np.linspace(2500.0, 101000.0, Nlev)
    lat = np.linspace(-87.5, 87.5, Nlat)
    lon = np.linspace(0.0, 360.0, Nlon, endpoint=False)
    time = np.linspace(0.0, 86400.0 * 2, Nt)  # seconds since start

    def _field(scale: float, bias: float) -> np.ndarray:
        # Smooth in pressure so cubic interpolation is well-conditioned.
        p_norm = (pfull_pa[None, :, None, None] - pfull_pa.min()) / (
            pfull_pa.max() - pfull_pa.min()
        )
        base = bias + scale * (1.0 - p_norm)
        base = base + 0.01 * scale * rng.standard_normal((Nt, Nlev, Nlat, Nlon))
        return base.astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "ua": (("time", "pfull", "lat", "lon"), _field(30.0, 0.0)),
            "va": (("time", "pfull", "lat", "lon"), _field(5.0, 0.0)),
            "ta": (("time", "pfull", "lat", "lon"), _field(40.0, 240.0)),
            "ps": (("time", "lat", "lon"),
                   (101000.0 + 100.0 * rng.standard_normal((Nt, Nlat, Nlon)))
                   .astype(np.float32)),
        },
        coords={
            "time":  ("time",  time),
            "pfull": ("pfull", pfull_pa),
            "lat":   ("lat",   lat),
            "lon":   ("lon",   lon),
        },
        attrs={"source": "synthetic ClimaAtmos-shaped fixture"},
    )
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(nc_path))


# ─── postprocess output schema ───────────────────────────────────────────────

class TestPostprocessSchema:
    def _run(self, tmp_path: Path):
        nc_path = tmp_path / "in.nc"
        out_path = tmp_path / "run.zarr"
        _make_synthetic_clima_nc(nc_path)
        postprocess(
            inputs=[nc_path],
            out_path=out_path,
            run_id=42,
            params={
                "tau_drag_days": 1.0,
                "delta_T_y":     60.0,
                "delta_theta_z": 10.0,
                "tau_surf_days": 4.0,
                "tau_atm_days":  40.0,
                "seed":          0,
            },
        )
        return out_path

    def test_writes_zarr_store(self, tmp_path: Path):
        out_path = self._run(tmp_path)
        assert out_path.is_dir()
        ds = xr.open_zarr(str(out_path), consolidated=True)
        assert set(ds.data_vars) >= {"u", "v", "T", "ps"}

    def test_dims_match_consumer(self, tmp_path: Path):
        out_path = self._run(tmp_path)
        ds = xr.open_zarr(str(out_path), consolidated=True)
        # Per fots.data.held_suarez: u/v/T are (time, level, lat, lon),
        # ps is (time, lat, lon).
        for v in ("u", "v", "T"):
            assert ds[v].dims == ("time", "level", "lat", "lon"), (
                f"{v} dims = {ds[v].dims}"
            )
        assert ds["ps"].dims == ("time", "lat", "lon")

    def test_pressure_levels_locked(self, tmp_path: Path):
        out_path = self._run(tmp_path)
        ds = xr.open_zarr(str(out_path), consolidated=True)
        assert tuple(int(round(v)) for v in ds["level"].values) == ERA5_LEVELS_HPA

    def test_lat_lon_in_degrees(self, tmp_path: Path):
        out_path = self._run(tmp_path)
        ds = xr.open_zarr(str(out_path), consolidated=True)
        assert ds.attrs["lat_units"] == "degrees_north"
        assert ds.attrs["lon_units"] == "degrees_east"
        assert ds["lat"].values.min() >= -90.0
        assert ds["lat"].values.max() <= 90.0
        assert ds["lon"].values.min() >= 0.0
        assert ds["lon"].values.max() < 360.0 + 1e-9

    def test_time_units_attr(self, tmp_path: Path):
        out_path = self._run(tmp_path)
        ds = xr.open_zarr(str(out_path), consolidated=True)
        assert "second" in ds.attrs["time_units"].lower()

    def test_run_id_and_params_attrs(self, tmp_path: Path):
        out_path = self._run(tmp_path)
        ds = xr.open_zarr(str(out_path), consolidated=True)
        assert int(ds.attrs["run_id"]) == 42
        assert float(ds.attrs["param_delta_T_y"]) == 60.0
        assert float(ds.attrs["param_tau_drag_days"]) == 1.0

    def test_finite_values(self, tmp_path: Path):
        out_path = self._run(tmp_path)
        ds = xr.open_zarr(str(out_path), consolidated=True)
        for v in ("u", "v", "T", "ps"):
            assert np.all(np.isfinite(ds[v].values)), f"NaN/Inf in {v}"


# ─── DataModule consumes our output ──────────────────────────────────────────

class TestDataModuleCompat:
    """Build a 1-run train split and run HeldSuarezDataModule against it.

    This is the real contract test: if the consumer can ``setup("fit")``
    without changes and ``_stack_fields`` produces (T, 25, H, W) tensors,
    we're done. Importing the consumer requires the ``fots`` package, so
    the test is skipped when that env is not on the Python path.
    """

    def _build_split(self, tmp_path: Path) -> Path:
        root = tmp_path / "hs3d_clima"
        train_dir = root / "train"
        val_dir = root / "val"
        test_dir = root / "test"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        # Two short runs in train, one each in val/test so the loader has
        # at least one window in every split (need T_in + T_out ≤ Nt).
        for sd, rid in [(train_dir, 0), (train_dir, 1), (val_dir, 2), (test_dir, 3)]:
            nc = tmp_path / f"raw_{rid}.nc"
            _make_synthetic_clima_nc(nc, Nt=5)
            postprocess(
                inputs=[nc],
                out_path=sd / f"run_{rid:04d}.zarr",
                run_id=rid,
                params={"tau_drag_days": 1.0, "delta_T_y": 60.0,
                        "delta_theta_z": 10.0, "tau_surf_days": 4.0,
                        "tau_atm_days": 40.0, "seed": rid},
            )

        # Minimal splits.json so the loader doesn't try to glob.
        splits_payload = {
            "n_runs": 4,
            "train": [0, 1], "val": [2], "test": [3],
        }
        (root / "splits.json").write_text(json.dumps(splits_payload) + "\n")
        return root

    def test_consumer_loads(self, tmp_path: Path):
        torch = pytest.importorskip("torch")
        try:
            from fots.data.held_suarez import HeldSuarezDataModule
        except ImportError:
            pytest.skip("fots package not importable in this env")

        root = self._build_split(tmp_path)

        # Channel order locked to consumer: u_50..u_1000, v_50..v_1000,
        # T_50..T_1000, ps  ⇒  3 × 8 + 1 = 25.
        field_names = tuple(
            [f"{v}_{lvl}hpa" for v in ("u", "v", "T") for lvl in ERA5_LEVELS_HPA]
            + ["ps"]
        )

        dm = HeldSuarezDataModule(
            root=str(root),
            time_steps_per_run=5,
            dim_in=25, dim_out=25,
            spatial_resolution=(16, 32),
            field_names=field_names,
            batch_size=1,
            n_steps_input=2,
            n_steps_output=1,
            num_workers=0,
            dataset_name="hs3d_clima_test",
        )

        sample = dm.train_dataset[0]
        assert "input_fields" in sample and "output_fields" in sample
        assert sample["input_fields"].shape  == (2, 25, 16, 32)
        assert sample["output_fields"].shape == (1, 25, 16, 32)
        assert torch.all(torch.isfinite(sample["input_fields"]))
        assert torch.all(torch.isfinite(sample["output_fields"]))


# ─── postprocess_clima_dir: production-path schema test ──────────────────────

def _make_clima_dir(tmp_path: Path, *, Nt: int = 4, Nlat: int = 16, Nlon: int = 32) -> Path:
    """Build a synthetic ClimaAtmos run directory matching v0.39's layout:

      clima/
        output_active/
          ua_6h_inst_pressure.nc     (pressure_level, lat, lon, time)
          va_6h_inst_pressure.nc
          ta_6h_inst_pressure.nc
          pfull_6h_inst.nc           (z, lat, lon, time)

    Pressure axis carries the same 37 default levels ClimaAtmos uses
    (so all 8 ERA5 levels are picked by exact match, no interpolation).
    """
    rng = np.random.default_rng(7)
    out_dir = tmp_path / "clima" / "output_active"
    out_dir.mkdir(parents=True)

    # ClimaAtmos's default 37 pressure levels in Pa (must contain all
    # ERA5 levels — listing them explicitly keeps this contract pinned).
    pressure_pa = np.array([
        100, 200, 300, 500, 700, 1000, 2000, 3000,
        5000, 7000, 10000, 12500, 15000, 17500, 20000, 22500,
        25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000,
        65000, 70000, 75000, 77500, 80000, 82500, 85000, 87500,
        90000, 92500, 95000, 97500, 100000,
    ], dtype=np.float64)
    z_m = np.linspace(100.0, 55000.0, 31)
    lat = np.linspace(-87.5, 87.5, Nlat)
    lon = np.linspace(0.0, 360.0, Nlon, endpoint=False)
    time = np.arange(Nt) * 21600.0  # seconds, 6h cadence

    def _3d_pressure(scale: float, bias: float) -> np.ndarray:
        # Smooth in pressure-log so ranges look reasonable.
        lp = np.log(pressure_pa)
        lp_norm = (lp - lp.min()) / (lp.max() - lp.min())
        base = bias + scale * lp_norm[None, :, None, None]
        noise = 0.05 * scale * rng.standard_normal((Nt, pressure_pa.size, Nlat, Nlon))
        return (base + noise).astype(np.float32).transpose(1, 2, 3, 0)  # → (P, lat, lon, time)

    for name, scale, bias in [("ua", 30.0, 0.0), ("va", 5.0, 0.0), ("ta", 60.0, 235.0)]:
        ds = xr.Dataset(
            data_vars={name: (("pressure_level", "lat", "lon", "time"), _3d_pressure(scale, bias))},
            coords={
                "pressure_level": ("pressure_level", pressure_pa,
                                   {"units": "Pa", "axis": "Z"}),
                "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"}),
                "time": ("time", time, {"units": "s", "axis": "T"}),
            },
        )
        ds.to_netcdf(str(out_dir / f"{name}_6h_inst_pressure.nc"))

    # pfull on z axis (real heights): values decay from ~1013 hPa
    # near surface to ~50 Pa at z_max=55 km. Add small spatial
    # variation so ps = pfull[z=0] is non-degenerate.
    g, Rd, T0 = 9.81, 287.0, 280.0
    H = Rd * T0 / g  # ~8.2 km scale height
    p_z = 101325.0 * np.exp(-z_m / H)
    pfull = np.empty((31, Nlat, Nlon, Nt), dtype=np.float32)
    for ti in range(Nt):
        spatial = 1.0 + 0.001 * np.cos(np.deg2rad(lat))[:, None] * np.sin(np.deg2rad(lon))[None, :]
        pfull[:, :, :, ti] = (p_z[:, None, None] * spatial[None, :, :]).astype(np.float32)
    ds_pf = xr.Dataset(
        data_vars={"pfull": (("z", "lat", "lon", "time"), pfull)},
        coords={
            "z": ("z", z_m, {"units": "m", "axis": "Z"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
            "time": ("time", time, {"units": "s", "axis": "T"}),
        },
    )
    ds_pf.to_netcdf(str(out_dir / "pfull_6h_inst.nc"))

    return tmp_path / "clima"


class TestPostprocessClimaDir:
    """Exercise the production path: per-variable NetCDFs in a ClimaAtmos
    run directory → run.zarr matching the HS-3D consumer schema."""

    def test_picks_era5_levels_exactly(self, tmp_path: Path):
        clima_dir = _make_clima_dir(tmp_path)
        out_path = tmp_path / "run.zarr"
        postprocess_clima_dir(
            clima_dir=clima_dir, out_path=out_path,
            run_id=11, params={"delta_T_y": 60.0},
        )
        ds = xr.open_zarr(str(out_path), consolidated=True)
        assert tuple(int(round(v)) for v in ds["level"].values) == ERA5_LEVELS_HPA
        for v in ("u", "v", "T"):
            assert ds[v].dims == ("time", "level", "lat", "lon")
        assert ds["ps"].dims == ("time", "lat", "lon")

    def test_ps_derived_from_pfull(self, tmp_path: Path):
        clima_dir = _make_clima_dir(tmp_path)
        out_path = tmp_path / "run.zarr"
        postprocess_clima_dir(
            clima_dir=clima_dir, out_path=out_path,
            run_id=0, params=None,
        )
        ds = xr.open_zarr(str(out_path), consolidated=True)
        ps = ds["ps"].values
        # Synthetic pfull[z=0] ≈ 100100 (scale-height-decayed MSLP at z=100m)
        # ± a small spatial modulation; bounds are loose.
        assert ps.min() > 99_500
        assert ps.max() < 101_500

    def test_run_attrs_and_finiteness(self, tmp_path: Path):
        clima_dir = _make_clima_dir(tmp_path)
        out_path = tmp_path / "run.zarr"
        postprocess_clima_dir(
            clima_dir=clima_dir, out_path=out_path,
            run_id=42, params={"tau_drag_days": 2.0},
        )
        ds = xr.open_zarr(str(out_path), consolidated=True)
        assert int(ds.attrs["run_id"]) == 42
        assert float(ds.attrs["param_tau_drag_days"]) == 2.0
        for v in ("u", "v", "T", "ps"):
            assert np.all(np.isfinite(ds[v].values))

    def test_missing_pressure_level_errors(self, tmp_path: Path):
        """Drop one of the ERA5 levels from the synthetic NetCDFs and
        confirm postprocess refuses to silently fall back."""
        clima_dir = _make_clima_dir(tmp_path)
        active = clima_dir / "output_active"
        # Rewrite ua without the 850 hPa level so the level lookup fails.
        ua = xr.open_dataset(str(active / "ua_6h_inst_pressure.nc"), decode_times=False).load()
        keep = ua["pressure_level"].values != 85000.0
        ua_trimmed = ua.isel(pressure_level=keep)
        (active / "ua_6h_inst_pressure.nc").unlink()
        ua_trimmed.to_netcdf(str(active / "ua_6h_inst_pressure.nc"))

        out_path = tmp_path / "run.zarr"
        with pytest.raises(ValueError, match=r"85000(\.0)? Pa not present"):
            postprocess_clima_dir(
                clima_dir=clima_dir, out_path=out_path,
                run_id=0, params=None,
            )
