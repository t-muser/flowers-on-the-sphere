"""Unit tests for the pure-numpy reductions in ``diagnostics``.

Synthetic analytic fields with known answers — no Julia, no ClimaAtmos, no
on-disk Zarr (matching the fixture-only style of ``test_postprocess.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

from datagen.clima_atmos.held_suarez.diagnostics import (
    area_weights,
    eddy_heat_flux,
    eddy_mom_flux,
    eke,
    hemispheric_asymmetry,
    jet_metrics,
    ke_spectrum,
    mke_from_zonal,
    weighted_lat_mean,
    zonal_mean,
)


@pytest.fixture
def grid():
    lat = np.linspace(-90.0, 90.0, 73)        # symmetric about equator
    lon = np.linspace(0.0, 360.0, 144, endpoint=False)
    return lat, lon


# ─── area weighting ─────────────────────────────────────────────────────────


def test_area_weights_poles_downweighted(grid):
    lat, _ = grid
    w = area_weights(lat)
    assert w[0] == pytest.approx(0.0, abs=1e-6)      # south pole
    assert w[-1] == pytest.approx(0.0, abs=1e-6)     # north pole
    assert w[lat.size // 2] == pytest.approx(1.0)    # equator


def test_weighted_lat_mean_equals_cos_average(grid):
    lat, _ = grid
    # mean of a constant field is the constant, regardless of weighting
    field = np.full(lat.size, 3.0)
    assert weighted_lat_mean(field, lat) == pytest.approx(3.0)


# ─── zonal mean ─────────────────────────────────────────────────────────────


def test_zonal_mean_of_lon_constant(grid):
    lat, lon = grid
    field = np.tile(np.cos(np.deg2rad(lat))[:, None], (1, lon.size))
    zm = zonal_mean(field)
    np.testing.assert_allclose(zm, np.cos(np.deg2rad(lat)), atol=1e-12)


def test_zonal_mean_of_pure_wave_is_zero(grid):
    lat, lon = grid
    field = np.cos(np.deg2rad(lon))[None, :] * np.ones((lat.size, 1))
    np.testing.assert_allclose(zonal_mean(field), 0.0, atol=1e-12)


# ─── EKE / MKE ──────────────────────────────────────────────────────────────


def test_eke_zero_for_zonally_symmetric_flow(grid):
    lat, lon = grid
    u = np.tile(20.0 * np.cos(np.deg2rad(lat))[:, None], (1, lon.size))
    v = np.zeros_like(u)
    np.testing.assert_allclose(eke(u, v), 0.0, atol=1e-12)


def test_eke_matches_wave_amplitude(grid):
    lat, lon = grid
    amp = 4.0
    # u' = amp*cos(lon): zonal variance = amp^2/2, so EKE = 0.5*amp^2/2.
    u = amp * np.cos(np.deg2rad(2 * lon))[None, :] * np.ones((lat.size, 1))
    v = np.zeros_like(u)
    expected = 0.5 * (amp ** 2 / 2.0)
    np.testing.assert_allclose(eke(u, v), expected, rtol=1e-6)


def test_mke_from_zonal(grid):
    lat, _ = grid
    u_zm = 10.0 * np.ones(lat.size)
    v_zm = np.zeros(lat.size)
    np.testing.assert_allclose(mke_from_zonal(u_zm, v_zm), 50.0)


# ─── eddy fluxes ────────────────────────────────────────────────────────────


def test_eddy_heat_flux_sign(grid):
    lat, lon = grid
    # positively correlated v' and T' → positive [v'T']
    wave = np.cos(np.deg2rad(lon))[None, :] * np.ones((lat.size, 1))
    v = 2.0 * wave
    T = 3.0 * wave
    flux = eddy_heat_flux(v, T)
    assert np.all(flux > 0)
    # anti-correlated → negative
    assert np.all(eddy_mom_flux(2.0 * wave, -wave) < 0)


# ─── KE spectrum ────────────────────────────────────────────────────────────


def test_ke_spectrum_peaks_at_input_wavenumber(grid):
    lat, lon = grid
    k0 = 6
    u = np.cos(np.deg2rad(k0 * lon))[None, :] * np.ones((lat.size, 1))
    v = np.zeros_like(u)
    k, power = ke_spectrum(u, v, lat)
    assert int(k[np.argmax(power)]) == k0


# ─── jet metrics ────────────────────────────────────────────────────────────


def test_jet_metrics_double_jet(grid):
    lat, _ = grid
    # two westerly jets at +/-45 deg, peak 30 m/s, plus tropical easterlies
    ubar = 30.0 * np.exp(-((np.abs(lat) - 45.0) ** 2) / (2 * 10.0 ** 2))
    ubar -= 8.0 * np.exp(-(lat ** 2) / (2 * 15.0 ** 2))
    m = jet_metrics(ubar, lat)
    assert m["jet_max_nh"] == pytest.approx(30.0, abs=1.0)
    assert m["jet_lat_nh"] == pytest.approx(45.0, abs=3.0)
    assert m["jet_lat_sh"] == pytest.approx(-45.0, abs=3.0)
    assert m["n_jets"] == 2
    assert m["jet_fwhm_nh"] > 0


# ─── hemispheric symmetry ───────────────────────────────────────────────────


def test_hemispheric_asymmetry(grid):
    lat, _ = grid
    sym = np.cos(np.deg2rad(lat))[None, :] * np.ones((4, 1))      # (level, lat)
    assert hemispheric_asymmetry(sym, lat) == pytest.approx(0.0, abs=1e-12)
    tilted = (lat / 90.0)[None, :] * np.ones((4, 1))             # odd → asymmetric
    assert hemispheric_asymmetry(tilted, lat) > 0.5
