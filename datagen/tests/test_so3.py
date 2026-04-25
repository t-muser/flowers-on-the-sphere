"""Tests for the Galewsky postprocess SO(3) tilt utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from datagen.galewsky.so3 import (
    _local_basis,
    back_rotated_thetaphi,
    rotation_from_seed,
    vector_jacobian,
)


def test_rotation_from_seed_deterministic_and_unit_axis():
    for seed in (0, 1, 42):
        axis_a, angle_a = rotation_from_seed(seed)
        axis_b, angle_b = rotation_from_seed(seed)
        assert np.linalg.norm(axis_a) == pytest.approx(1.0, rel=1e-12)
        assert 0.0 <= angle_a < 2.0 * math.pi
        np.testing.assert_array_equal(axis_a, axis_b)
        assert angle_a == angle_b


def test_back_rotated_round_trip():
    """Forward + back rotation should recover the input ``(θ, φ)`` to 1e-12."""
    Nlat, Nlon = 33, 64
    lat = -np.pi / 2.0 + (np.arange(Nlat) + 0.5) * np.pi / Nlat
    lon = np.arange(Nlon) * 2.0 * np.pi / Nlon

    axis, angle = rotation_from_seed(3)
    theta_in, phi_in = back_rotated_thetaphi(lat, lon, axis, angle)
    # Form the input-frame unit vectors and forward-rotate.
    sin_t = np.sin(theta_in)
    xyz_in = np.stack(
        [sin_t * np.cos(phi_in), sin_t * np.sin(phi_in), np.cos(theta_in)],
        axis=-1,
    )
    R = Rotation.from_rotvec(angle * axis).as_matrix()
    xyz_out = xyz_in @ R.T
    # Check we recovered the output grid.
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    cos_lat = np.cos(lat_g)
    expected = np.stack(
        [cos_lat * np.cos(lon_g), cos_lat * np.sin(lon_g), np.sin(lat_g)],
        axis=-1,
    )
    np.testing.assert_allclose(xyz_out, expected, atol=1e-12)


def test_vector_jacobian_orthogonal():
    """The local-frame Jacobian must be a rotation: ``M·Mᵀ = I`` per point."""
    Nlat, Nlon = 17, 32
    lat = -np.pi / 2.0 + (np.arange(Nlat) + 0.5) * np.pi / Nlat
    lon = np.arange(Nlon) * 2.0 * np.pi / Nlon
    axis, angle = rotation_from_seed(5)

    theta_in, phi_in = back_rotated_thetaphi(lat, lon, axis, angle)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    theta_out = (np.pi / 2.0) - lat_g
    phi_out = lon_g

    M = vector_jacobian(theta_out, phi_out, theta_in, phi_in, axis, angle)
    # M is (Nlat, Nlon, 2, 2). Each 2x2 block should be orthogonal.
    MMt = np.einsum("...ij,...kj->...ik", M, M)
    eye = np.broadcast_to(np.eye(2), MMt.shape)
    np.testing.assert_allclose(MMt, eye, atol=1e-10)


def test_vector_jacobian_preserves_speed():
    """Applying ``M`` to a unit-speed velocity should preserve speed pointwise."""
    Nlat, Nlon = 17, 32
    lat = -np.pi / 2.0 + (np.arange(Nlat) + 0.5) * np.pi / Nlat
    lon = np.arange(Nlon) * 2.0 * np.pi / Nlon
    axis, angle = rotation_from_seed(9)

    theta_in, phi_in = back_rotated_thetaphi(lat, lon, axis, angle)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    theta_out = (np.pi / 2.0) - lat_g
    phi_out = lon_g

    M = vector_jacobian(theta_out, phi_out, theta_in, phi_in, axis, angle)

    rng = np.random.default_rng(0)
    u_phi = rng.standard_normal((Nlat, Nlon))
    u_theta = rng.standard_normal((Nlat, Nlon))
    speed_in = np.sqrt(u_phi ** 2 + u_theta ** 2)
    up_out = M[..., 0, 0] * u_phi + M[..., 0, 1] * u_theta
    ut_out = M[..., 1, 0] * u_phi + M[..., 1, 1] * u_theta
    speed_out = np.sqrt(up_out ** 2 + ut_out ** 2)
    np.testing.assert_allclose(speed_out, speed_in, rtol=1e-10, atol=1e-10)


def test_local_basis_orthonormal():
    """``_local_basis`` should return three pairwise-orthonormal vectors."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.05, math.pi - 0.05, 32)
    phi = rng.uniform(0.0, 2.0 * math.pi, 32)
    F = _local_basis(theta, phi)
    gram = np.einsum("...ji,...jk->...ik", F, F)
    eye = np.broadcast_to(np.eye(3), gram.shape)
    np.testing.assert_allclose(gram, eye, atol=1e-12)
