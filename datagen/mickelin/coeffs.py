"""Mickelin generalized Navier-Stokes (GNS) on the sphere.

Two public entry points:

- ``coefficients_from_RLkT(R, Lambda, kappa, tau)`` — closed-form map from
  the Mickelin physical parameters ``(R, Λ, κ, τ)`` to the three coefficients
  ``(Γ₀, Γ₂, Γ₄)`` of the band-limited linear operator
  ``f(Δ+4K)(Δ+2K)`` with ``f(x) = Γ₀ − Γ₂·x + Γ₄·x²``. The roots of ``f``
  are placed so that the unstable band in wavenumber ``k² = ℓ(ℓ+1)/R²``
  is ``(k₋², k₊²) = ((π/Λ)²(1−κΛ/2)², (π/Λ)²(1+κΛ/2)²)``. The overall
  amplitude is normalised so the peak growth rate equals ``1/τ``.

- ``set_initial_conditions(omega, dist, basis, seed, ell_init, epsilon)`` —
  small-amplitude random vorticity seed built from an orthonormal basis
  of real spherical harmonics ``Y_{ℓm}`` with ``ℓ = 2 … ℓ_init`` and
  i.i.d. ``𝒩(0, 1)`` coefficients, scaled by ``epsilon``.

Reference: Mickelin, Słomka, Burns, Lecoanet, Le Bars, Fauve, Dunkel,
*Anomalous chained turbulence in actively driven flows on spheres*,
PRL 120, 164503 (2018); arXiv:1710.05525.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from datagen._ylm import real_ylm as _real_ylm


def coefficients_from_RLkT(
    R: float, Lambda: float, kappa: float, tau: float = 1.0
) -> Tuple[float, float, float]:
    """Closed-form map ``(R, Λ, κ, τ) → (Γ₀, Γ₂, Γ₄)``.

    The linear operator ``L = f(Δ+4K)(Δ+2K)`` with Gaussian curvature
    ``K = 1/R²`` acts on a spherical-harmonic mode ``Y_ℓ^m`` as

        σ(k²) = f(−k² + 4K) · (2K − k²),        k² ≡ ℓ(ℓ+1)/R²,

    where ``f(x) = Γ₀ − Γ₂·x + Γ₄·x²``. Since ``Γ₄ > 0``, ``f`` is an
    upward-opening parabola negative between its two roots ``X_lo < X_hi``.
    Mapping back: ``σ > 0`` between ``k² = 4K − X_hi`` and ``4K − X_lo``.

    We choose the two roots so this unstable band in ``k²`` matches the
    Mickelin SM-specified interval

        k₋² = (π/Λ)² · (1 − κΛ/2)²,   k₊² = (π/Λ)² · (1 + κΛ/2)²,

    i.e. ``X_lo = 4K − k₊²`` and ``X_hi = 4K − k₋²``. Peak growth rate is
    normalised to ``1/τ`` by an overall rescaling of ``(Γ₀, Γ₂, Γ₄)``.
    """
    if Lambda <= 0:
        raise ValueError(f"Lambda must be positive, got {Lambda}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if kappa * Lambda >= 2.0:
        raise ValueError(
            f"Require κΛ < 2 so that k₋² > 0, got κΛ={kappa * Lambda}"
        )
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    K = 1.0 / R ** 2
    kc = math.pi / Lambda
    k_minus_sq = kc ** 2 * (1.0 - kappa * Lambda / 2.0) ** 2
    k_plus_sq = kc ** 2 * (1.0 + kappa * Lambda / 2.0) ** 2

    X_lo = 4.0 * K - k_plus_sq
    X_hi = 4.0 * K - k_minus_sq

    # f(x) = Γ₄ (x − X_lo)(x − X_hi) = Γ₀ − Γ₂·x + Γ₄·x²
    Gamma_4 = 1.0
    Gamma_2 = Gamma_4 * (X_lo + X_hi)
    Gamma_0 = Gamma_4 * X_lo * X_hi

    # Find peak growth rate numerically on a fine ℓ grid and rescale so
    # that σ(ℓ_peak) = 1/τ. Because σ(k²) is linear in (Γ₀, Γ₂, Γ₄), a
    # uniform rescale of all three leaves root locations (and the peak ℓ)
    # unchanged.
    ell_max = max(int(4.0 * math.pi * R / Lambda), 64)
    ell = np.arange(1, ell_max + 1, dtype=np.float64)
    k_sq = ell * (ell + 1.0) / R ** 2
    x = -k_sq + 4.0 * K
    f_raw = Gamma_0 - Gamma_2 * x + Gamma_4 * x ** 2
    sigma_raw = f_raw * (2.0 * K - k_sq)
    sigma_peak = float(np.max(sigma_raw))
    if sigma_peak <= 0.0:
        raise ValueError(
            "No unstable band found for the given (R, Λ, κ, τ); check inputs."
        )

    scale = 1.0 / (tau * sigma_peak)
    return (Gamma_0 * scale, Gamma_2 * scale, Gamma_4 * scale)


def growth_rate_spectrum(
    ell: np.ndarray,
    R: float,
    Gamma_0: float,
    Gamma_2: float,
    Gamma_4: float,
) -> np.ndarray:
    """Linear growth rate ``σ(ℓ)`` for the GNS operator on the sphere.

    ``σ(ℓ) = f(−ℓ(ℓ+1)/R² + 4/R²) · (−ℓ(ℓ+1)/R² + 2/R²)`` with
    ``f(x) = Γ₀ − Γ₂·x + Γ₄·x²``. Useful for unit-testing the
    ``coefficients_from_RLkT`` map against the Mickelin band-limited design.
    """
    ell = np.asarray(ell, dtype=np.float64)
    K = 1.0 / R ** 2
    k_sq = ell * (ell + 1.0) / R ** 2
    x = -k_sq + 4.0 * K
    f = Gamma_0 - Gamma_2 * x + Gamma_4 * x ** 2
    return f * (2.0 * K - k_sq)


def unstable_band_ell_max(R: float, Lambda: float, kappa: float, safety: int = 4) -> int:
    """Maximum ``ℓ`` covered by the Mickelin unstable band, plus a safety margin.

    The unstable band in wavenumber is
    ``k₊ = (π/Λ)·(1 + κΛ/2)``; converting to spherical-harmonic degree via
    ``ℓ ≈ k·R`` and rounding up gives the smallest ``ℓ_init`` that already
    seeds the linearly-fastest-growing modes. Adding ``safety`` margin pads
    a few degrees above the band edge so the initial spectrum doesn't have
    to climb up to the peak from below.
    """
    if Lambda <= 0:
        raise ValueError(f"Lambda must be positive, got {Lambda}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    k_plus = (math.pi / Lambda) * (1.0 + kappa * Lambda / 2.0)
    return int(math.ceil(R * k_plus)) + int(safety)


def set_initial_conditions(
    omega,
    dist,
    basis,
    seed: int,
    ell_init: int = 6,
    epsilon: float = 1e-3,
) -> None:
    """Populate the Dedalus field ``omega`` with a low-amplitude random seed.

    ``ω₀(θ, φ) = ε · Σ_{ℓ=2..ℓ_init} Σ_{m=-ℓ..ℓ} a_{ℓm} · Y_{ℓm}(θ, φ)``
    with ``a_{ℓm}`` i.i.d. ``𝒩(0, 1)`` drawn from
    ``np.random.Generator(np.random.PCG64(seed))``. Using the real
    orthonormal ``Y_{ℓm}`` basis makes the resulting field real by
    construction; no reality constraint on complex coefficients is needed.

    Modes ``ℓ = 0`` (constant) and ``ℓ = 1`` (rigid rotation) are
    excluded so that the seed projects strictly onto the non-trivial part
    of the vorticity space.
    """
    phi, theta = dist.local_grids(basis)  # broadcast shapes (Nphi_local, 1), (1, Ntheta_local)
    omega_grid = np.zeros(np.broadcast_shapes(phi.shape, theta.shape), dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for ell in range(2, int(ell_init) + 1):
        for m in range(-ell, ell + 1):
            a = float(rng.standard_normal())
            omega_grid += a * _real_ylm(ell, m, theta, phi)

    omega["g"] = epsilon * omega_grid
