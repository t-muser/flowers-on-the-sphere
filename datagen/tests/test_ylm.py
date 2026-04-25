"""Sanity checks for the shared real-spherical-harmonic basis."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special

from datagen._ylm import real_ylm


def _scipy_real_ylm(ell: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Reference real-Y_ℓm via scipy's complex sph_harm_y(n, m, theta, phi).

    scipy's complex spherical harmonic includes the Condon-Shortley phase
    via its underlying ``lpmv`` call, and our ``real_ylm`` builds on the
    same ``lpmv``, so the standard real-form combination
    ``sqrt(2) Re[Y]`` (resp. ``sqrt(2) Im[Y]``) for ``m > 0`` (resp.
    ``m < 0``) matches without an extra ``(-1)^m`` factor.
    """
    if m == 0:
        return special.sph_harm_y(ell, 0, theta, phi).real
    if m > 0:
        return math.sqrt(2.0) * special.sph_harm_y(ell, m, theta, phi).real
    return math.sqrt(2.0) * special.sph_harm_y(ell, -m, theta, phi).imag


@pytest.mark.parametrize("ell,m", [(0, 0), (1, 0), (1, 1), (1, -1),
                                    (2, 0), (2, 1), (2, 2), (3, -2), (4, 3)])
def test_real_ylm_matches_scipy(ell, m):
    """``real_ylm`` should match the standard real-Y_ℓm convention to
    spectral precision. Random sample of ``(θ, φ)``.
    """
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.05, math.pi - 0.05, 64)
    phi = rng.uniform(0.0, 2.0 * math.pi, 64)
    got = real_ylm(ell, m, theta, phi)
    want = _scipy_real_ylm(ell, m, theta, phi)
    np.testing.assert_allclose(got, want, atol=1.0e-10)


def test_real_ylm_orthonormal():
    """⟨Y_ℓm, Y_ℓ'm'⟩ on a Gauss-Legendre grid should equal δ_{ℓℓ'} δ_{mm'}."""
    Ntheta = 64
    Nphi = 128
    cos_t, w_gl = np.polynomial.legendre.leggauss(Ntheta)
    theta = np.arccos(cos_t)
    phi = np.linspace(0.0, 2.0 * math.pi, Nphi, endpoint=False)
    dphi = 2.0 * math.pi / Nphi

    P, T = np.meshgrid(phi, theta, indexing="xy")  # broadcasted (Ntheta, Nphi)
    weights = w_gl[:, None] * dphi  # area element (sin θ dθ dφ but with leggauss in cos θ)

    modes = [(0, 0), (1, 0), (1, 1), (2, 1), (2, -2), (3, 0)]
    Ys = np.stack(
        [real_ylm(ell, m, T, P) for ell, m in modes],
        axis=0,
    )  # (n_modes, Ntheta, Nphi)
    gram = np.einsum("aij,bij,ij->ab", Ys, Ys, weights)
    np.testing.assert_allclose(gram, np.eye(len(modes)), atol=1.0e-10)
