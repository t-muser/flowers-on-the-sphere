"""Sanity checks for the Krekhov forcing module."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from datagen._ylm import real_ylm
from datagen.cahn_hilliard.forcing import axis_from_seed


def test_axis_from_seed_unit_norm_and_deterministic():
    for seed in (0, 1, 42, 12345):
        a1 = axis_from_seed(seed)
        a2 = axis_from_seed(seed)
        assert a1.shape == (3,)
        assert np.linalg.norm(a1) == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_array_equal(a1, a2)


def test_axis_from_seed_distinct_seeds_distinct_axes():
    a0 = axis_from_seed(0)
    a1 = axis_from_seed(1)
    assert not np.allclose(a0, a1)


def test_rotation_round_trip_identity():
    """Rotating ``r̂`` by ``R(α, ê)`` and back must return ``r̂`` to 1e-12."""
    rng = np.random.default_rng(0)
    n = 256
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    axis = axis_from_seed(7)
    angle = 0.7
    R = Rotation.from_rotvec(angle * axis).as_matrix()
    R_inv = Rotation.from_rotvec(-angle * axis).as_matrix()
    pts_back = (pts @ R.T) @ R_inv.T
    np.testing.assert_allclose(pts_back, pts, atol=1e-12)


def test_rotated_ylm_eigenvalue():
    """For a pure Y_ℓ^m, the rotated pattern is also a (rotated) Y_ℓ^m and so
    its surface Laplacian eigenvalue is still ``-ℓ(ℓ+1)``.

    We don't run Dedalus here; instead we verify that the rotated pattern
    preserves the L²-norm (which is what the eigenvalue argument relies on:
    the rotation is an isometry on each ℓ-shell).
    """
    Ntheta, Nphi = 64, 128
    cos_t, w_gl = np.polynomial.legendre.leggauss(Ntheta)
    theta = np.arccos(cos_t)
    phi = np.linspace(0.0, 2.0 * math.pi, Nphi, endpoint=False)
    P, T = np.meshgrid(phi, theta, indexing="xy")
    weights = w_gl[:, None] * (2.0 * math.pi / Nphi)

    ell, m = 6, 6
    Y = real_ylm(ell, m, T, P)
    norm_unrot = np.einsum("ij,ij,ij->", Y, Y, weights)
    assert norm_unrot == pytest.approx(1.0, rel=1.0e-10)

    # Rotate the *evaluation* points by R⁻¹ so that the value at (θ, φ)
    # equals the original Y at the rotated point — same trick as
    # update_forcing_field at t > 0.
    axis = axis_from_seed(11)
    angle = 1.234
    R_inv = Rotation.from_rotvec(-angle * axis).as_matrix()
    sin_t = np.sin(T)
    cos_tt = np.cos(T)
    sin_p = np.sin(P)
    cos_p = np.cos(P)
    xyz = np.stack([sin_t * cos_p, sin_t * sin_p, cos_tt], axis=-1)
    xyz_rot = xyz @ R_inv.T
    xyz_rot[..., 2] = np.clip(xyz_rot[..., 2], -1.0, 1.0)
    theta_rot = np.arccos(xyz_rot[..., 2])
    phi_rot = np.mod(np.arctan2(xyz_rot[..., 1], xyz_rot[..., 0]), 2.0 * math.pi)

    Y_rot = real_ylm(ell, m, theta_rot, phi_rot)
    norm_rot = np.einsum("ij,ij,ij->", Y_rot, Y_rot, weights)
    # Rotation is an L² isometry; norm is unchanged to quadrature precision.
    assert norm_rot == pytest.approx(1.0, rel=1.0e-3)
