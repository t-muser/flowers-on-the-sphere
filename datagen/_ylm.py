"""Real orthonormal spherical harmonics on ``S²``.

Used by both the Mickelin GNS initial conditions and the Cahn-Hilliard
Krekhov forcing module. Convention: ``θ`` is colatitude (``0`` at the north
pole), ``φ`` is longitude. For ``m = 0`` the basis function is the standard
Legendre polynomial; for ``m > 0`` the real-cosine combination; for
``m < 0`` the real-sine combination, all with unit ``L²(S²)`` norm.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import lpmv


def real_ylm(ell: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real orthonormal spherical harmonic ``Y_{ℓm}(θ, φ)``."""
    x = np.cos(theta)
    if m == 0:
        norm = math.sqrt((2 * ell + 1) / (4.0 * math.pi))
        return norm * lpmv(0, ell, x)
    m_abs = abs(m)
    log_num = math.lgamma(ell - m_abs + 1)
    log_den = math.lgamma(ell + m_abs + 1)
    norm = math.sqrt((2 * ell + 1) / (2.0 * math.pi)) * math.exp(0.5 * (log_num - log_den))
    legendre = lpmv(m_abs, ell, x)
    if m > 0:
        return norm * legendre * np.cos(m_abs * phi)
    return norm * legendre * np.sin(m_abs * phi)
