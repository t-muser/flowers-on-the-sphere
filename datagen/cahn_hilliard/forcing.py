"""Krekhov-style traveling spatially-periodic forcing for Cahn-Hilliard on S².

Following Weith, Krekhov, Zimmermann 2009 (arXiv 0809.0211), the forcing
is a single ``Y_ℓ^m`` real spherical harmonic whose pattern rotates rigidly
about a per-trajectory tilted axis ``ê``:

    F(x̂, t) = Re[ Y_ℓ^m( R_ê(ω·t) · x̂ ) ]

The rotation axis ``ê`` is drawn uniformly on ``S²`` from ``seed``, so the
forcing is not aligned with the simulation grid. The pattern enters the
Cahn-Hilliard equation through ``∇² F``; for a pure ``Y_ℓ^m`` the Laplacian
is exactly ``−ℓ(ℓ+1)/R² · F`` (Dedalus computes this from spectral coeffs).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from datagen._ylm import real_ylm


def axis_from_seed(seed: int) -> np.ndarray:
    """Per-trajectory rotation axis, uniform on ``S²`` and reproducible."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    z = 2.0 * float(rng.uniform()) - 1.0
    phi = 2.0 * math.pi * float(rng.uniform())
    s = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([s * math.cos(phi), s * math.sin(phi), z], dtype=np.float64)


def update_forcing_field(
    forcing,
    dist,
    basis,
    t: float,
    ell: int,
    m: int,
    axis: np.ndarray,
    omega: float,
) -> None:
    """In-place set ``forcing['g'] = Re[Y_ℓ^m(R_axis(ω·t) · x̂)]``.

    Steps:
      1. Read the local ``(phi, theta)`` grid points from Dedalus.
      2. Convert each to a 3-D unit vector ``x̂``.
      3. Apply ``R_axis(−ω·t)`` so that lab-frame coordinates are
         expressed in the rotating frame; in the rotating frame the
         forcing is simply ``Y_ℓ^m(θ', φ')``.
      4. Convert back to ``(θ', φ')`` and evaluate ``real_ylm``.

    The forcing field is set on the dealiased grid because Dedalus
    evaluates the explicit RHS (``a·∇²F``) at dealias scale; assigning
    to ``forcing['g']`` at scale 1 would broadcast-mismatch.
    """
    forcing.change_scales(basis.dealias)
    phi, theta = dist.local_grids(basis, scales=basis.dealias)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t * np.ones_like(x)

    # Stack the per-point unit vectors as (..., 3) so the matrix multiply
    # broadcasts naturally over the local grid.
    xyz = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    R_inv = Rotation.from_rotvec(-omega * t * np.asarray(axis)).as_matrix()
    xyz_rot = xyz @ R_inv.T

    xyz_rot[..., 2] = np.clip(xyz_rot[..., 2], -1.0, 1.0)
    theta_rot = np.arccos(xyz_rot[..., 2])
    phi_rot = np.mod(np.arctan2(xyz_rot[..., 1], xyz_rot[..., 0]), 2.0 * math.pi)

    forcing["g"] = real_ylm(int(ell), int(m), theta_rot, phi_rot)
