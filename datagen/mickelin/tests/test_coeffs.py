"""Spectral-growth-rate sanity checks for the Mickelin coefficient map.

Given ``(R, Λ, κ, τ)``, ``coefficients_from_RLkT`` must place the unstable
band so that

1. The ℓ that maximises ``σ(ℓ) = f(-ℓ(ℓ+1)/R² + 4/R²)·(-ℓ(ℓ+1)/R² + 2/R²)``
   sits near the Mickelin scaling ``ℓ_peak ≈ πR/Λ``.
2. The bandwidth (full support of ``σ > 0``) in ℓ-units is ``≈ κR``.
3. The peak growth rate equals ``1/τ`` by construction of the amplitude
   rescaling.

Run::

    uv run --project datagen pytest datagen/mickelin/tests/test_coeffs.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from datagen.mickelin.coeffs import (
    coefficients_from_RLkT,
    growth_rate_spectrum,
    unstable_band_ell_max,
)


@pytest.mark.parametrize(
    "r_over_lambda, kappa_lambda",
    [
        (4.0, 0.3),
        (4.0, 0.7),
        (6.0, 0.5),
        (8.0, 0.5),
        (10.0, 0.3),
        (10.0, 0.7),
    ],
)
def test_coefficient_signs(r_over_lambda, kappa_lambda):
    R, tau = 1.0, 1.0
    Lambda = R / r_over_lambda
    kappa = kappa_lambda / Lambda
    Gamma_0, Gamma_2, Gamma_4 = coefficients_from_RLkT(R, Lambda, kappa, tau)
    assert Gamma_0 > 0.0, "Γ₀ must be positive"
    assert Gamma_2 < 0.0, "Γ₂ must be negative"
    assert Gamma_4 > 0.0, "Γ₄ must be positive"


@pytest.mark.parametrize("r_over_lambda", [4.0, 6.0, 8.0, 10.0])
@pytest.mark.parametrize("kappa_lambda", [0.3, 0.5, 0.7])
def test_peak_ell_matches_scaling(r_over_lambda, kappa_lambda):
    """argmax σ(ℓ) should sit near ℓ ≈ πR/Λ = π · r_over_lambda.

    Two sources of offset from the continuum scaling ``ℓ ≈ πR/Λ``:

    1. ``σ(k²) = f(x)·(2K−k²)`` is the product of two factors. ``|f|`` is
       maximised at the band midpoint ``(k₋² + k₊²)/2 = (π/Λ)²(1 + (κΛ/2)²)``,
       while ``|2K − k²|`` grows with ``k²``, so the joint peak sits slightly
       above the band midpoint.
    2. ``ℓ(ℓ+1)/R²`` versus ``(ℓ/R)²`` differs by a correction of order ``1/ℓ``.

    Both effects scale with ``κΛ``; a tolerance of ``κR`` (the band
    half-width ``πκR/π ≈ κR``) comfortably covers them.
    """
    R, tau = 1.0, 1.0
    Lambda = R / r_over_lambda
    kappa = kappa_lambda / Lambda
    Gamma_0, Gamma_2, Gamma_4 = coefficients_from_RLkT(R, Lambda, kappa, tau)

    ell = np.arange(2, 400)
    sigma = growth_rate_spectrum(ell, R, Gamma_0, Gamma_2, Gamma_4)
    ell_peak = ell[int(np.argmax(sigma))]
    ell_peak_expected = math.pi * R / Lambda
    tolerance = max(kappa * R, 2.0)
    assert abs(ell_peak - ell_peak_expected) <= tolerance, (
        f"ℓ_peak={ell_peak}, expected≈{ell_peak_expected:.2f}, "
        f"tolerance={tolerance:.2f}"
    )


@pytest.mark.parametrize("r_over_lambda", [6.0, 8.0, 10.0])
@pytest.mark.parametrize("kappa_lambda", [0.3, 1.0, 1.5])
def test_bandwidth_matches_kappa_R(r_over_lambda, kappa_lambda):
    """Width of the unstable band in ℓ-units should be ≈ κR (SM Eq. 86).

    Band edges in k are π/Λ ± κ/2; converting via ℓ ≈ kR gives width κR.
    """
    R, tau = 1.0, 1.0
    Lambda = R / r_over_lambda
    kappa = kappa_lambda / Lambda
    Gamma_0, Gamma_2, Gamma_4 = coefficients_from_RLkT(R, Lambda, kappa, tau)

    ell = np.arange(2, 600)
    sigma = growth_rate_spectrum(ell, R, Gamma_0, Gamma_2, Gamma_4)
    positive = sigma > 0.0
    assert positive.any()
    idx = np.where(positive)[0]
    band_lo = ell[idx[0]]
    band_hi = ell[idx[-1]]
    observed_width = band_hi - band_lo
    expected_width = kappa * R
    assert abs(observed_width - expected_width) <= 2.0, (
        f"band [{band_lo}, {band_hi}] width={observed_width}, "
        f"expected≈{expected_width:.2f}"
    )


@pytest.mark.parametrize(
    "r_over_lambda, kappa_lambda, tau",
    [
        (4.0, 0.5, 1.0),
        (8.0, 0.5, 1.0),
        (10.0, 0.7, 1.0),
        (6.0, 0.3, 2.5),
    ],
)
def test_peak_rate_equals_inverse_tau(r_over_lambda, kappa_lambda, tau):
    """The coefficient map normalises ``max σ = 1/τ`` on a fine ℓ grid."""
    R = 1.0
    Lambda = R / r_over_lambda
    kappa = kappa_lambda / Lambda
    Gamma_0, Gamma_2, Gamma_4 = coefficients_from_RLkT(R, Lambda, kappa, tau)

    ell = np.arange(1, 1024)
    sigma = growth_rate_spectrum(ell, R, Gamma_0, Gamma_2, Gamma_4)
    assert np.max(sigma) == pytest.approx(1.0 / tau, rel=1.0e-10)


@pytest.mark.parametrize(
    "r_over_lambda, kappa_lambda",
    [(4.0, 0.3), (4.0, 1.5), (6.0, 1.0), (8.0, 1.0), (10.0, 1.5)],
)
def test_unstable_band_ell_max(r_over_lambda, kappa_lambda):
    R = 1.0
    Lambda = R / r_over_lambda
    kappa = kappa_lambda / Lambda
    safety = 4
    expected_k_plus = math.pi / Lambda + 0.5 * kappa
    expected = int(math.ceil(R * expected_k_plus)) + safety
    assert unstable_band_ell_max(R, Lambda, kappa, safety=safety) == expected


def test_unstable_band_ell_max_covers_growth_peak():
    """The derived ``ell_init`` must cover the linearly-fastest-growing
    mode, otherwise the seed misses the unstable band entirely.
    """
    R = 1.0
    Lambda = R / 8.0
    kappa = 0.5 / Lambda
    Gamma_0, Gamma_2, Gamma_4 = coefficients_from_RLkT(R, Lambda, kappa, 1.0)
    ell = np.arange(1, 200)
    sigma = growth_rate_spectrum(ell, R, Gamma_0, Gamma_2, Gamma_4)
    ell_peak = ell[int(np.argmax(sigma))]
    assert unstable_band_ell_max(R, Lambda, kappa) >= ell_peak
