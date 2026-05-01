"""Per-trajectory SO(3) tilt applied at postprocess time.

The Dedalus shallow-water solver runs in the canonical frame (rotation axis
= polar axis), because the Coriolis term ``2·Ω·zcross(u) = 2·Ω·sin(lat)·k̂×u``
is only correct in that frame. To break grid alignment without breaking the
solver, we draw a per-trajectory ``(axis, angle)`` keyed on the ``run_id``
(so every run gets a unique rotation) and apply it to the saved ``(lat, lon)``
snapshots.

Mathematically, sampling the standard-frame field at the back-rotated grid
points is equivalent to running the simulation on a tilted-axis sphere.
Vector-valued fields (the velocity ``(u_phi, u_theta)``) get the local
Jacobian pushed forward, since the local east/south basis rotates with the
sphere.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_from_seed(seed: int) -> Tuple[np.ndarray, float]:
    """Draw ``(axis, angle)`` deterministically from ``seed``.

    Axis is uniform on ``S²`` and angle is uniform on ``[0, 2π)``. Returns
    ``(axis_xyz, angle_rad)``; ``axis_xyz`` has unit norm.
    """
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    z = 2.0 * float(rng.uniform()) - 1.0
    phi = 2.0 * math.pi * float(rng.uniform())
    s = math.sqrt(max(0.0, 1.0 - z * z))
    axis = np.array([s * math.cos(phi), s * math.sin(phi), z], dtype=np.float64)
    angle = 2.0 * math.pi * float(rng.uniform())
    return axis, angle


def _local_basis(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return per-point local frame ``(ê_phi, ê_theta, r̂)`` of shape
    ``(..., 3, 3)`` with the three columns being the three basis vectors.

    ``θ`` is colatitude, ``φ`` longitude. ``ê_phi`` is eastward (∂/∂φ
    normalised) and ``ê_theta`` is southward (∂/∂θ normalised).
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    e_phi = np.stack([-sin_p, cos_p, np.zeros_like(sin_p)], axis=-1)
    e_theta = np.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], axis=-1)
    r_hat = np.stack([sin_t * cos_p, sin_t * sin_p, cos_t], axis=-1)
    return np.stack([e_phi, e_theta, r_hat], axis=-1)


def back_rotated_thetaphi(
    lat_target: np.ndarray,
    lon_target: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each output ``(lat_out, lon_out)`` return the input-frame ``(θ, φ)``.

    The output grid is an outer product ``(Nlat, Nlon)``; we form the unit
    vector ``r̂_out`` at every gridpoint, back-rotate by ``R⁻¹`` and convert
    back to ``(theta, phi)``.
    """
    lat_g, lon_g = np.meshgrid(lat_target, lon_target, indexing="ij")
    cos_lat = np.cos(lat_g)
    r_out = np.stack(
        [cos_lat * np.cos(lon_g), cos_lat * np.sin(lon_g), np.sin(lat_g)],
        axis=-1,
    )  # (Nlat, Nlon, 3)
    R_inv = Rotation.from_rotvec(angle * np.asarray(axis)).inv().as_matrix()
    r_in = r_out @ R_inv.T  # equivalent to (R_inv · r_out) per point
    # Numerical clip to keep arccos in domain.
    r_in[..., 2] = np.clip(r_in[..., 2], -1.0, 1.0)
    theta_in = np.arccos(r_in[..., 2])
    phi_in = np.mod(np.arctan2(r_in[..., 1], r_in[..., 0]), 2.0 * math.pi)
    return theta_in, phi_in


def vector_jacobian(
    theta_out: np.ndarray,
    phi_out: np.ndarray,
    theta_in: np.ndarray,
    phi_in: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Per-point 2x2 mixing matrix that maps ``(u_phi, u_theta)`` from the
    input local frame to the output local frame.

    The full local frame ``(ê_phi, ê_theta, r̂)`` rotates by ``R``: the
    output basis at ``r̂_out`` equals ``R`` applied to the input basis at
    ``R⁻¹·r̂_out``. Projecting ``R · ê_in_α`` onto ``ê_out_β`` for
    ``α, β ∈ {phi, theta}`` gives the 2x2 mixing matrix.
    """
    R = Rotation.from_rotvec(angle * np.asarray(axis)).as_matrix()
    # Local frames as (..., 3, 3). Last index is the basis vector.
    F_in = _local_basis(theta_in, phi_in)        # (..., 3, 3)
    F_out = _local_basis(theta_out, phi_out)     # (..., 3, 3)
    # Push forward input basis vectors.
    R_F_in = np.einsum("ij,...jk->...ik", R, F_in)
    # Project onto output basis. We only want the 2D tangent block.
    M = np.einsum("...ji,...jk->...ik", F_out, R_F_in)
    return M[..., :2, :2]
