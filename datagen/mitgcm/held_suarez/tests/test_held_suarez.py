"""Unit tests for the pure-Python Held-Suarez physics implementation.

All tests verify the mathematical properties stated in Held & Suarez (1994),
Eqs. (1)–(4). No MITgcm binary is required.

Run::

    uv run --project datagen pytest datagen/mitgcm/held_suarez/tests/test_held_suarez.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from datagen.mitgcm.held_suarez._constants import (
    KAPPA,
    P0,
    HS_KA,
    HS_KF,
    HS_KS,
    HS_DELTA_T_Y,
    HS_DELTA_T_Z,
    HS_SIGMAB,
    HS_T0,
    HS_T_MIN,
)
from datagen.mitgcm.held_suarez._physics import (
    equilibrium_temperature,
    equilibrium_temperature_K,
    newtonian_cooling_rate,
    rayleigh_friction_rate,
    reference_temperature_profile,
)


# ─── equilibrium_temperature ────────────────────────────────────────────────

class TestEquilibriumTemperature:
    def test_equator_surface_matches_T0(self):
        """At the equator (φ=0) and surface (p=p0), θ_eq = T0 exactly.

        At φ=0: sin²φ=0, cos²φ=1, ln(p/p0)=0.  The formula reduces to T0.
        The T_min branch is T_min*(p0/p0)^κ = T_min < T0, so T0 wins.
        """
        lat = np.array([0.0])
        p   = np.array([P0])
        theta_eq = equilibrium_temperature(lat, p)
        assert theta_eq[0] == pytest.approx(HS_T0, rel=1e-12)

    def test_pole_surface_cooler_than_equator(self):
        """At the surface (p=p0), the pole (φ=π/2) is colder by ΔTy."""
        lat_eq   = np.array([0.0])
        lat_pole = np.array([math.pi / 2.0])
        p = np.array([P0])
        theta_eq_eq   = equilibrium_temperature(lat_eq,   p)[0]
        theta_eq_pole = equilibrium_temperature(lat_pole, p)[0]
        # At surface: ln(p/p0)=0, so θ_eq = T0 − ΔTy·sin²φ − 0
        assert theta_eq_eq - theta_eq_pole == pytest.approx(HS_DELTA_T_Y, rel=1e-10)

    def test_bounded_below_by_T_min_times_poisson(self):
        """θ_eq ≥ T_min·(p0/p)^κ everywhere (the stratospheric temperature floor)."""
        rng = np.random.default_rng(0)
        lat = rng.uniform(-math.pi / 2, math.pi / 2, 500)
        p   = rng.uniform(0.01 * P0, P0, 500)
        theta_eq  = equilibrium_temperature(lat, p)
        theta_min = HS_T_MIN * (P0 / p) ** KAPPA
        assert np.all(theta_eq >= theta_min - 1e-10)

    def test_temperature_K_converts_correctly(self):
        """T_eq = θ_eq · (p/p0)^κ ≥ T_min everywhere."""
        rng = np.random.default_rng(1)
        lat = rng.uniform(-math.pi / 2, math.pi / 2, 200)
        p   = rng.uniform(0.01 * P0, P0, 200)
        T_eq = equilibrium_temperature_K(lat, p)
        assert np.all(T_eq >= HS_T_MIN - 1e-10)

    def test_equator_surface_temperature_K_equals_T0(self):
        """At equatorial surface, T_eq = θ_eq (since (p/p0)^κ = 1 at p=p0)."""
        lat = np.array([0.0])
        p   = np.array([P0])
        T_eq = equilibrium_temperature_K(lat, p)
        assert T_eq[0] == pytest.approx(HS_T0, rel=1e-12)

    @pytest.mark.parametrize("delta_T_y", [40.0, 60.0, 80.0])
    def test_equator_pole_gradient_linear_in_delta_T_y(self, delta_T_y):
        """θ_eq(pole) − θ_eq(equator) at the surface equals −ΔTy."""
        lat_eq   = np.array([0.0])
        lat_pole = np.array([math.pi / 2.0])
        p = np.array([P0])
        diff = (
            equilibrium_temperature(lat_eq,   p, delta_T_y=delta_T_y)[0]
            - equilibrium_temperature(lat_pole, p, delta_T_y=delta_T_y)[0]
        )
        assert diff == pytest.approx(delta_T_y, rel=1e-10)

    def test_aloft_temperature_limited_by_T_min(self):
        """At the very top of the atmosphere θ_eq is dominated by the T_min floor."""
        lat = np.array([0.0])
        p_top = np.array([100.0])   # 1 hPa — very high
        theta_eq = equilibrium_temperature(lat, p_top)
        theta_min = HS_T_MIN * (P0 / p_top[0]) ** KAPPA
        # The T_min branch is much larger than T0 at 1 hPa; θ_eq should equal it.
        assert theta_eq[0] == pytest.approx(theta_min, rel=1e-10)

    def test_broadcasting_lat_p(self):
        """``lat`` and ``p`` broadcast correctly to a 2-D output."""
        lat = np.linspace(-math.pi / 2, math.pi / 2, 32)[:, None]
        p   = np.linspace(0.1 * P0, P0, 20)[None, :]
        theta_eq = equilibrium_temperature(lat, p)
        assert theta_eq.shape == (32, 20)
        assert np.all(np.isfinite(theta_eq))


# ─── rayleigh_friction_rate ─────────────────────────────────────────────────

class TestRayleighFriction:
    def test_zero_above_sigmab(self):
        """k_v = 0 for σ ≤ σ_b (no friction above the boundary layer)."""
        sigma_free = np.linspace(0.0, HS_SIGMAB, 100)
        kv = rayleigh_friction_rate(sigma_free)
        np.testing.assert_array_equal(kv, 0.0)

    def test_max_at_surface(self):
        """k_v(σ=1) = k_f (maximum friction at the surface)."""
        kv_surface = rayleigh_friction_rate(np.array([1.0]))
        assert kv_surface[0] == pytest.approx(HS_KF, rel=1e-12)

    def test_monotone_in_sigma(self):
        """k_v must increase monotonically from σ_b to 1."""
        sigma = np.linspace(HS_SIGMAB, 1.0, 100)
        kv = rayleigh_friction_rate(sigma)
        assert np.all(np.diff(kv) >= 0.0)

    def test_linear_profile_in_boundary_layer(self):
        """k_v is linear in (σ − σ_b) / (1 − σ_b) by construction."""
        sigma = np.array([HS_SIGMAB + 0.1, HS_SIGMAB + 0.2])
        kv = rayleigh_friction_rate(sigma)
        expected = HS_KF * (sigma - HS_SIGMAB) / (1.0 - HS_SIGMAB)
        np.testing.assert_allclose(kv, expected, rtol=1e-12)

    @pytest.mark.parametrize("kf", [1.0 / 43200.0, 1.0 / 86400.0, 1.0 / 172800.0])
    def test_surface_rate_equals_kf_param(self, kf):
        """k_v(σ=1) = kf for any kf value."""
        kv = rayleigh_friction_rate(np.array([1.0]), kf=kf)
        assert kv[0] == pytest.approx(kf, rel=1e-12)


# ─── newtonian_cooling_rate ─────────────────────────────────────────────────

class TestNewtonianCooling:
    def test_free_atmosphere_equals_ka(self):
        """Above the boundary layer (σ ≤ σ_b), k_T = k_a everywhere."""
        sigma_free = np.linspace(0.0, HS_SIGMAB, 50)
        lat = np.linspace(-math.pi / 2, math.pi / 2, 50)
        kT = newtonian_cooling_rate(lat, sigma_free)
        np.testing.assert_allclose(kT, HS_KA, rtol=1e-12)

    def test_pole_surface_approaches_ka(self):
        """At the pole (cos φ = 0), the cos⁴φ factor vanishes → k_T = k_a."""
        lat_pole = np.array([math.pi / 2.0])
        sigma_bl = np.array([1.0])
        kT = newtonian_cooling_rate(lat_pole, sigma_bl)
        assert kT[0] == pytest.approx(HS_KA, rel=1e-10)

    def test_equator_surface_equals_ks(self):
        """At the equatorial surface (σ=1, φ=0), k_T = k_s."""
        lat_eq = np.array([0.0])
        sigma_sfc = np.array([1.0])
        kT = newtonian_cooling_rate(lat_eq, sigma_sfc)
        assert kT[0] == pytest.approx(HS_KS, rel=1e-12)

    def test_bounded_between_ka_and_ks(self):
        """k_a ≤ k_T ≤ k_s everywhere (since k_s > k_a)."""
        rng = np.random.default_rng(7)
        lat   = rng.uniform(-math.pi / 2, math.pi / 2, 500)
        sigma = rng.uniform(0.0, 1.0, 500)
        kT = newtonian_cooling_rate(lat, sigma)
        assert np.all(kT >= HS_KA - 1e-14)
        assert np.all(kT <= HS_KS + 1e-14)

    def test_symmetric_in_latitude(self):
        """k_T(φ, σ) = k_T(−φ, σ) by the cos⁴φ factor."""
        lat = np.linspace(0.0, math.pi / 2, 30)
        sigma = np.full_like(lat, 0.9)
        kT_north = newtonian_cooling_rate(lat, sigma)
        kT_south = newtonian_cooling_rate(-lat, sigma)
        np.testing.assert_allclose(kT_north, kT_south, rtol=1e-14)

    def test_monotone_decrease_with_latitude_in_BL(self):
        """Within the boundary layer, k_T decreases from equator toward pole."""
        lat = np.linspace(0.0, math.pi / 2, 50)
        sigma = np.full_like(lat, 0.9)   # well inside boundary layer
        kT = newtonian_cooling_rate(lat, sigma)
        assert np.all(np.diff(kT) <= 0.0)


# ─── reference_temperature_profile ─────────────────────────────────────────

class TestReferenceTemperatureProfile:
    def test_shape(self):
        """Output matches Nr input levels."""
        Nr = 20
        p_centers = (np.arange(Nr) + 0.5) * (P0 / Nr)
        T_ref = reference_temperature_profile(p_centers)
        assert T_ref.shape == (Nr,)

    def test_bounded_below_by_T_min(self):
        """T_ref ≥ T_min (the stratospheric floor applies at the equator too)."""
        Nr = 20
        p_centers = (np.arange(Nr) + 0.5) * (P0 / Nr)
        T_ref = reference_temperature_profile(p_centers)
        assert np.all(T_ref >= HS_T_MIN - 1e-10)

    def test_surface_level_close_to_T0(self):
        """Near-surface temperature should be close to T0 = 315 K."""
        Nr = 20
        p_centers = (np.arange(Nr) + 0.5) * (P0 / Nr)
        T_ref = reference_temperature_profile(p_centers)
        # Bottom level (k=Nr, highest pressure ≈ p0): T ≈ T0.
        assert T_ref[-1] == pytest.approx(HS_T0, rel=0.01)

    def test_increases_toward_surface(self):
        """Temperature increases downward (stable atmosphere, no inversion)."""
        Nr = 20
        p_centers = (np.arange(Nr) + 0.5) * (P0 / Nr)
        T_ref = reference_temperature_profile(p_centers)
        # T_ref ordered top-to-bottom (index 0 = top, index Nr-1 = bottom).
        # Temperature should not decrease toward the surface.
        assert np.all(np.diff(T_ref) >= 0.0)
