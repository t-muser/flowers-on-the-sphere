"""Reproducible Gaussian-noise initial condition for Cahn-Hilliard on the sphere.

ψ(θ,φ) = mean_init + sqrt(variance) · Σ_{2 ≤ ℓ ≤ ℓ_init} Σ_m a_{ℓm}·Y_ℓ^m

with ``a_{ℓm}`` i.i.d. ``𝒩(0, 1)`` from ``PCG64(seed)``. ℓ=0 (mean) is
excluded — the mean is set explicitly via ``mean_init`` and conserved by
the equation since the only non-conservative term enters via ``∇²``. ℓ=1
(dipole) is also excluded because a uniform CH initial condition has no
preferred axis; the rotational symmetry would just be broken by numerics.
"""

from __future__ import annotations

import math

import numpy as np

from datagen._ylm import real_ylm


def set_initial_conditions(
    psi,
    dist,
    basis,
    seed: int,
    mean_init: float,
    variance: float,
    ell_init: int,
) -> None:
    """Populate the Dedalus field ``psi`` with a low-amplitude bandlimited
    Gaussian noise centred at ``mean_init``.
    """
    phi, theta = dist.local_grids(basis)
    field = np.zeros(np.broadcast_shapes(phi.shape, theta.shape), dtype=np.float64)

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for ell in range(2, int(ell_init) + 1):
        for m in range(-ell, ell + 1):
            a = float(rng.standard_normal())
            field += a * real_ylm(ell, m, theta, phi)

    psi["g"] = float(mean_init) + math.sqrt(float(variance)) * field
