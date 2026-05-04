"""Tests for ``datagen/cpl_aim_ocn/regrid.py``.

The regrid utility is optional (used offline by downstream consumers),
so these tests exercise its correctness via synthetic grids — no real
MITgcm output required.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_regrid.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datagen.cpl_aim_ocn import regrid as regrid_mod
from datagen.cpl_aim_ocn.regrid import (
    DEFAULT_NLAT,
    DEFAULT_NLON,
    RegridWeights,
    _build_target_grid,
    _lonlat_to_xyz,
    apply_weights,
    build_weights,
    regrid_to_latlon,
)


# ─── Geometry helpers ────────────────────────────────────────────────────────

class TestLonLatToXyz:
    def test_origin_is_unit_x(self):
        np.testing.assert_allclose(
            _lonlat_to_xyz(0.0, 0.0), np.array([1.0, 0.0, 0.0]),
            atol=1e-12,
        )

    def test_north_pole_is_unit_z(self):
        np.testing.assert_allclose(
            _lonlat_to_xyz(123.0, 90.0), np.array([0.0, 0.0, 1.0]),
            atol=1e-12,
        )

    def test_south_pole_is_minus_z(self):
        np.testing.assert_allclose(
            _lonlat_to_xyz(-45.0, -90.0), np.array([0.0, 0.0, -1.0]),
            atol=1e-12,
        )

    def test_outputs_are_unit_norm(self):
        rng = np.random.default_rng(0)
        lon = rng.uniform(-180, 180, 50)
        lat = rng.uniform(-90, 90, 50)
        xyz = _lonlat_to_xyz(lon, lat)
        norms = np.linalg.norm(xyz, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


class TestBuildTargetGrid:
    def test_shape(self):
        lat, lon = _build_target_grid(64, 128)
        assert lat.shape == (64,)
        assert lon.shape == (128,)

    def test_lat_strictly_inside_pole(self):
        lat, _ = _build_target_grid(64, 128)
        assert -90.0 < lat.min() < lat.max() < 90.0

    def test_lon_covers_full_circle(self):
        _, lon = _build_target_grid(64, 128)
        assert lon[0] > -180.0 and lon[-1] < 180.0
        # Spacing roughly 360 / nlon.
        np.testing.assert_allclose(np.diff(lon), 360.0 / 128, atol=1e-12)


# ─── Synthetic cs32 grid: fake but plausible ─────────────────────────────────

def _fake_cs32_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return XC, YC arrays of shape (6, 32, 32) covering the sphere
    well enough for the regrid tests.

    We do *not* need a faithful cs32 — we just need 6144 points
    distributed over the sphere so that for any target lat/lon there
    is a reasonably-close source. Use a Fibonacci sphere wrapped into
    six 32×32 panels (purely for shape compatibility with the writer).
    """
    n = 6 * 32 * 32
    i = np.arange(n)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    lat_rad = np.arcsin(1.0 - 2.0 * (i + 0.5) / n)
    lon_rad = (i / phi) % 1.0 * 2.0 * np.pi - np.pi

    XC = np.rad2deg(lon_rad).reshape(6, 32, 32).astype(np.float64)
    YC = np.rad2deg(lat_rad).reshape(6, 32, 32).astype(np.float64)
    return XC, YC


# ─── build_weights ──────────────────────────────────────────────────────────

class TestBuildWeights:
    def test_nearest_returns_k1_weights(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=8, nlon=16, method="nearest")
        assert w.k == 1
        assert w.indices.shape == (8, 16, 1)
        assert w.weights.shape == (8, 16, 1)
        np.testing.assert_array_equal(w.weights, 1.0)

    def test_idw_default_k_is_4(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=8, nlon=16, method="idw")
        assert w.k == 4
        assert w.weights.shape == (8, 16, 4)

    def test_idw_weights_sum_to_one_per_target(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=8, nlon=16, method="idw")
        sums = w.weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-12)

    def test_idw_weights_non_negative(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=8, nlon=16, method="idw")
        assert np.all(w.weights >= 0.0)

    def test_unknown_method_raises(self):
        XC, YC = _fake_cs32_grid()
        with pytest.raises(ValueError, match="Unknown regrid method"):
            build_weights(XC, YC, method="snake")

    def test_idw_k_must_be_at_least_2(self):
        XC, YC = _fake_cs32_grid()
        with pytest.raises(ValueError, match="k"):
            build_weights(XC, YC, method="idw", k=1)


# ─── apply_weights ───────────────────────────────────────────────────────────

class TestApplyWeights:
    def test_constant_field_round_trips_through_nearest(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=16, nlon=32, method="nearest")
        src = np.full((6, 32, 32), 7.5)
        out = apply_weights(src, w)
        np.testing.assert_allclose(out, 7.5)

    def test_constant_field_round_trips_through_idw(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=16, nlon=32, method="idw")
        src = np.full((6, 32, 32), -3.14)
        out = apply_weights(src, w)
        np.testing.assert_allclose(out, -3.14, atol=1e-9)

    def test_latitude_banded_field_remains_banded(self):
        # Source f(lat) → output should be ≈ f(target_lat) up to
        # near-neighbour discretisation. We use a smooth zonal pattern
        # (cosine of latitude) so the staircase from nearest-neighbour
        # is not more than one source-cell width.
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=32, nlon=64, method="idw")
        src = np.cos(np.deg2rad(YC))  # purely a function of latitude
        out = apply_weights(src, w)
        # For each target latitude row, the output should be roughly
        # a constant equal to cos(target_lat).
        target_lat, _ = _build_target_grid(32, 64)
        for j, tl in enumerate(target_lat):
            row = out[j]
            assert np.std(row) < 0.05   # near-constant zonally
            assert abs(row.mean() - np.cos(np.deg2rad(tl))) < 0.05

    def test_supports_leading_time_dim(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=16, nlon=32, method="nearest")
        src = np.full((3, 6, 32, 32), 2.5)   # (time, face, j, i)
        out = apply_weights(src, w)
        assert out.shape == (3, 16, 32)
        np.testing.assert_allclose(out, 2.5)

    def test_supports_leading_time_and_vertical(self):
        XC, YC = _fake_cs32_grid()
        w = build_weights(XC, YC, nlat=16, nlon=32, method="nearest")
        src = np.full((3, 5, 6, 32, 32), 4.0)  # (time, sigma, face, j, i)
        out = apply_weights(src, w)
        assert out.shape == (3, 5, 16, 32)


# ─── regrid_to_latlon end-to-end ─────────────────────────────────────────────

def _fake_cs32_dataset(nt: int = 2) -> xr.Dataset:
    XC, YC = _fake_cs32_grid()
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            "atm_T2m": (("time", "face", "j", "i"),
                        rng.normal(290, 5, (nt, 6, 32, 32))),
            "atm_UVEL": (("time", "Zsigma", "face", "j", "i"),
                         rng.normal(0, 5, (nt, 5, 6, 32, 32))),
        },
        coords={
            "time": np.arange(nt) * 86400.0,
            "Zsigma": np.linspace(50, 1000, 5) * 100.0,
            "XC": (("face", "j", "i"), XC),
            "YC": (("face", "j", "i"), YC),
        },
    )


class TestRegridToLatLon:
    def test_default_target_grid_dims(self):
        ds = _fake_cs32_dataset()
        out = regrid_to_latlon(ds)
        assert out.sizes["lat"] == DEFAULT_NLAT
        assert out.sizes["lon"] == DEFAULT_NLON

    def test_2d_field_keeps_time(self):
        ds = _fake_cs32_dataset(nt=3)
        out = regrid_to_latlon(ds, nlat=16, nlon=32)
        assert out["atm_T2m"].dims == ("time", "lat", "lon")
        assert out["atm_T2m"].sizes == {"time": 3, "lat": 16, "lon": 32}

    def test_3d_field_keeps_vertical(self):
        ds = _fake_cs32_dataset()
        out = regrid_to_latlon(ds, nlat=16, nlon=32)
        assert "Zsigma" in out["atm_UVEL"].dims
        assert out["atm_UVEL"].sizes == {
            "time": 2, "Zsigma": 5, "lat": 16, "lon": 32,
        }

    def test_subset_variables(self):
        ds = _fake_cs32_dataset()
        out = regrid_to_latlon(ds, nlat=16, nlon=32,
                               variables=["atm_T2m"])
        assert "atm_T2m" in out.data_vars
        assert "atm_UVEL" not in out.data_vars

    def test_attrs_carried_over(self):
        ds = _fake_cs32_dataset()
        ds.attrs["run_id"] = 42
        out = regrid_to_latlon(ds, method="idw", k=4, nlat=16, nlon=32)
        assert out.attrs.get("run_id") == 42
        assert out.attrs.get("regrid_method") == "idw"
        assert out.attrs.get("regrid_k") == 4

    def test_writes_to_zarr(self, tmp_path: Path):
        ds = _fake_cs32_dataset()
        out_path = tmp_path / "regridded.zarr"
        regrid_to_latlon(ds, nlat=16, nlon=32, out_path=out_path)
        assert out_path.is_dir()
        # Round-trip: reading back gives the same shape.
        rt = xr.open_zarr(out_path)
        assert rt.sizes["lat"] == 16
        assert rt.sizes["lon"] == 32

    def test_no_face_dim_is_clear_error(self):
        ds = xr.Dataset(
            data_vars={"x": (("a", "b"), np.zeros((4, 4)))},
        )
        with pytest.raises(ValueError, match="face"):
            regrid_to_latlon(ds, nlat=16, nlon=32)

    def test_no_j_or_i_dim_is_clear_error(self):
        ds = xr.Dataset(
            data_vars={"x": (("face",), np.zeros(6))},
            coords={"face": np.arange(6)},
        )
        with pytest.raises(ValueError, match="j"):
            regrid_to_latlon(ds, nlat=16, nlon=32)


# ─── End-to-end consistency: source mean ≈ target mean (constant field) ─────

class TestRegridConservationSpotChecks:
    def test_idw_preserves_constant_field_globally(self):
        ds = _fake_cs32_dataset()
        ds = ds.assign(atm_T2m=ds["atm_T2m"] * 0 + 290.0)
        out = regrid_to_latlon(ds, method="idw", nlat=32, nlon=64)
        np.testing.assert_allclose(
            out["atm_T2m"].values, 290.0, atol=1e-9
        )
