"""Unit tests for the 3-D pressure-level extraction path in solver.py.

Covers:
  - ``pressure_levels=None``: legacy single-level extraction is unchanged.
  - ``pressure_levels=(500.0,)``: 1-level 3-D output matches the
    single-level path numerically.
  - ``pressure_levels=(50, 100, 250, 500, 700, 850, 925, 1000)``: the
    8 ERA5-standard levels resolve to nearest model-k slabs and the
    arrays have the expected ``(Nt, 8, Nlat, Nlon)`` shape.

Run::

    uv run --project datagen pytest datagen/mitgcm/held_suarez/tests/test_extract_3d.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from datagen.mitgcm.held_suarez._constants import P0
from datagen.mitgcm.held_suarez.ic import pressure_thicknesses
from datagen.mitgcm.held_suarez.solver import (
    RunConfig,
    _extract_fields,
    _extract_fields_3d,
)


def _model_pressure_centers(Nr: int) -> np.ndarray:
    """Replicate the pressure-centre formula used by the solver/IC code."""
    del_r = pressure_thicknesses(Nr)
    upper_edges = P0 - np.concatenate([[0.0], np.cumsum(del_r[:-1])])
    return (upper_edges - 0.5 * del_r).astype(np.float32)


def _make_synthetic_data(Nt: int = 3, Nr: int = 20, Nlat: int = 64, Nlon: int = 128):
    """Build a minimal MDS-style dict of the form ``_read_mitgcm_output`` returns."""
    rng = np.random.default_rng(42)
    return {
        "UVEL":  rng.standard_normal((Nt, Nr, Nlat, Nlon)).astype(np.float32),
        "VVEL":  rng.standard_normal((Nt, Nr, Nlat, Nlon)).astype(np.float32),
        "THETA": (250.0 + rng.standard_normal((Nt, Nr, Nlat, Nlon))).astype(np.float32),
        "ETAN":  rng.standard_normal((Nt, Nlat, Nlon)).astype(np.float32) * 100.0,
        "pressure": _model_pressure_centers(Nr),
        "time": np.arange(Nt, dtype=np.float64) * 86400.0,
    }


# ─── single-level path: regression guard ────────────────────────────────────

def test_extract_fields_single_level_default():
    cfg = RunConfig()  # pressure_levels=None, pressure_hpa=500.0
    data = _make_synthetic_data()

    field_arrays, field_names, time_arr = _extract_fields(data, cfg)

    assert field_names == ["u_500hpa", "v_500hpa", "T_500hpa", "ps"]
    assert len(field_arrays) == 4
    for arr in field_arrays:
        assert arr.dtype == np.float32
        assert arr.shape == (3, 64, 128)
    # Time should be rebased to start at 0.
    assert time_arr[0] == 0.0


def test_extract_fields_single_level_picks_nearest_k():
    """A 500 hPa request should pick the model-k whose centre is nearest 500 hPa."""
    cfg = RunConfig(pressure_hpa=500.0)
    data = _make_synthetic_data()

    p_centers_hpa = data["pressure"] / 100.0
    expected_k = int(np.argmin(np.abs(p_centers_hpa - 500.0)))

    field_arrays, _, _ = _extract_fields(data, cfg)
    u_expected = data["UVEL"][:, expected_k, :, :].astype(np.float32)
    np.testing.assert_array_equal(field_arrays[0], u_expected)


# ─── 3-D path ────────────────────────────────────────────────────────────────

def test_extract_fields_3d_single_level_matches_legacy():
    """``pressure_levels=(500,)`` must produce the same numerical content
    as the single-level path (up to the new level axis)."""
    data = _make_synthetic_data()
    cfg_legacy = RunConfig(pressure_hpa=500.0)
    cfg_3d = RunConfig(pressure_levels=(500.0,))

    legacy_arrays, _, legacy_time = _extract_fields(data, cfg_legacy)
    fields_3d, fields_2d, level_hpa, p_actual, time_3d = _extract_fields_3d(data, cfg_3d)

    assert level_hpa.tolist() == [500.0]
    assert fields_3d["u"].shape == (3, 1, 64, 128)
    np.testing.assert_array_equal(fields_3d["u"][:, 0, :, :], legacy_arrays[0])
    np.testing.assert_array_equal(fields_3d["v"][:, 0, :, :], legacy_arrays[1])
    np.testing.assert_array_equal(fields_3d["T"][:, 0, :, :], legacy_arrays[2])
    np.testing.assert_array_equal(fields_2d["ps"], legacy_arrays[3])
    np.testing.assert_array_equal(time_3d, legacy_time)
    # The actual matched pressure should be within one cell-thickness of 500 hPa.
    assert abs(p_actual[0] - 500.0) < 50.0


def test_extract_fields_3d_eight_era5_levels():
    """The 8 ERA5-standard levels should resolve to 8 distinct slabs in
    a (Nt, 8, Nlat, Nlon) tensor for u/v/T."""
    requested = (50.0, 100.0, 250.0, 500.0, 700.0, 850.0, 925.0, 1000.0)
    cfg = RunConfig(pressure_levels=requested)
    data = _make_synthetic_data()

    fields_3d, fields_2d, level_hpa, p_actual, _ = _extract_fields_3d(data, cfg)

    # Shape check.
    assert fields_3d["u"].shape == (3, 8, 64, 128)
    assert fields_3d["v"].shape == (3, 8, 64, 128)
    assert fields_3d["T"].shape == (3, 8, 64, 128)
    assert fields_2d["ps"].shape == (3, 64, 128)

    # Coords.
    assert level_hpa.tolist() == list(requested)
    assert p_actual.shape == (8,)

    # Each requested level should be matched to a model-k centre within the
    # local cell thickness (~45 hPa for the bulk of the column, 150 hPa for
    # the top layer).
    for req, act in zip(requested, p_actual):
        if req <= 100.0:  # near or above the fat top layer
            assert abs(act - req) <= 150.0
        else:
            assert abs(act - req) <= 50.0

    # Sanity: each slab in `u` matches the corresponding raw model-k slab.
    p_centers_hpa = data["pressure"] / 100.0
    for i, req in enumerate(requested):
        expected_k = int(np.argmin(np.abs(p_centers_hpa - req)))
        np.testing.assert_array_equal(
            fields_3d["u"][:, i, :, :],
            data["UVEL"][:, expected_k, :, :].astype(np.float32),
        )


def test_extract_fields_3d_unique_levels_after_matching():
    """Two close-together requests collapse to the same model-k; that's
    expected behaviour, but we surface it via the actual-pressure coord."""
    cfg = RunConfig(pressure_levels=(498.0, 502.0))
    data = _make_synthetic_data()

    fields_3d, _, level_hpa, p_actual, _ = _extract_fields_3d(data, cfg)

    assert level_hpa.tolist() == [498.0, 502.0]
    # Both should resolve to the same model-k → same actual pressure.
    assert p_actual[0] == p_actual[1]
    np.testing.assert_array_equal(fields_3d["u"][:, 0, :, :], fields_3d["u"][:, 1, :, :])
