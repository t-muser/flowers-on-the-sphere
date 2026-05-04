"""IC unit tests for shock_caps — no PDE solves, CPU-only."""

import math

import numpy as np
import pytest

from datagen.shock_caps.geometry import (
    COMPUTATIONAL_LOWER,
    COMPUTATIONAL_UPPER,
    cell_centers_latlon,
    compute_aux,
    subcell_centers_latlon,
)
from datagen.shock_caps.ic import (
    CAP_RADIUS_RANGE,
    H_RANGE,
    VELOCITY_HALFWIDTH,
    cap_label,
    fill_ic,
    sample_caps,
)


NX, NY = 40, 20
K_DEFAULT = 4
DELTA_DEFAULT = 1.0


@pytest.fixture
def grid_params():
    xlower, ylower = COMPUTATIONAL_LOWER
    xupper, yupper = COMPUTATIONAL_UPPER
    dx = (xupper - xlower) / NX
    dy = (yupper - ylower) / NY
    return xlower, ylower, dx, dy


@pytest.fixture
def lat_lon(grid_params):
    xlower, ylower, dx, dy = grid_params
    return cell_centers_latlon(NX, NY, xlower, ylower, dx, dy)


def _call_fill_ic(seed, lat, lon, grid_params, sub_samples=4,
                  K=K_DEFAULT, delta=DELTA_DEFAULT):
    xlower, ylower, dx, dy = grid_params
    h = np.empty((NX, NY))
    mx = np.empty((NX, NY))
    my = np.empty((NX, NY))
    mz = np.empty((NX, NY))
    fill_ic(h, mx, my, mz, seed=seed, K=K, delta=delta,
            lat_centers=lat, lon_centers=lon,
            xlower=xlower, ylower=ylower, dx=dx, dy=dy,
            sub_samples=sub_samples)
    return h, mx, my, mz


# ---------------------------------------------------------------------------
# sample_caps
# ---------------------------------------------------------------------------

def test_sample_caps_determinism():
    a = sample_caps(42, K_DEFAULT, DELTA_DEFAULT)
    b = sample_caps(42, K_DEFAULT, DELTA_DEFAULT)
    states_a, centers_a, radii_a, bg_a = a
    states_b, centers_b, radii_b, bg_b = b
    assert states_a == states_b
    assert bg_a == bg_b
    assert radii_a == radii_b
    for ca, cb in zip(centers_a, centers_b):
        np.testing.assert_array_equal(ca, cb)


def test_sample_caps_different_seeds():
    a = sample_caps(0, K_DEFAULT, DELTA_DEFAULT)
    b = sample_caps(1, K_DEFAULT, DELTA_DEFAULT)
    assert a[0] != b[0]   # state lists differ


@pytest.mark.parametrize("K", [1, 4, 16])
@pytest.mark.parametrize("delta", [0.0, 0.5, 1.0])
def test_sample_caps_state_ranges(K, delta):
    h_lo, h_hi = H_RANGE
    v_bound = delta * VELOCITY_HALFWIDTH
    for seed in range(5):
        states, _, radii, background = sample_caps(seed, K, delta)
        assert len(states) == K
        for s in states + [background]:
            assert h_lo <= s["h"] <= h_hi
            assert -v_bound <= s["u"] <= v_bound
            assert -v_bound <= s["v"] <= v_bound
        for r in radii:
            assert CAP_RADIUS_RANGE[0] <= r <= CAP_RADIUS_RANGE[1]


def test_sample_caps_delta_zero_hard_zeros_velocity():
    """delta=0 must produce exactly-zero u, v (no float slack)."""
    for seed in range(5):
        for K in (1, 4, 16):
            states, _, _, background = sample_caps(seed, K, delta=0.0)
            for s in states + [background]:
                assert s["u"] == 0.0
                assert s["v"] == 0.0


def test_sample_caps_h_strictly_positive():
    for seed in range(20):
        states, _, _, background = sample_caps(seed, K_DEFAULT, DELTA_DEFAULT)
        for s in states + [background]:
            assert s["h"] > 0.0


def test_sample_caps_centers_unit_norm():
    _, centers, _, _ = sample_caps(0, K_DEFAULT, DELTA_DEFAULT)
    for c in centers:
        np.testing.assert_allclose(np.linalg.norm(c), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# cap_label
# ---------------------------------------------------------------------------

def test_cap_label_painter_order():
    """Higher-index cap wins in overlap regions (painter's algorithm)."""
    centers = [
        np.array([0.0, 0.0, 1.0]),   # cap 0: north pole
        np.array([0.0, 0.0, 1.0]),   # cap 1: same place, smaller radius
    ]
    radii = [1.0, 0.3]   # cap 1 strictly inside cap 0
    # A point inside both caps:
    lat = np.array([math.pi / 2 - 0.1])   # very close to north pole
    lon = np.array([0.0])
    label = cap_label(lat, lon, centers, radii)
    assert label[0] == 1   # higher-index cap wins


def test_cap_label_background_outside_caps():
    centers = [np.array([0.0, 0.0, 1.0])]   # north pole
    radii = [0.1]
    lat = np.array([0.0])   # equator — way outside the cap
    lon = np.array([0.0])
    label = cap_label(lat, lon, centers, radii)
    assert label[0] == 1   # K = 1 ⇒ background label = 1


def test_cap_label_inside_cap_assigned():
    centers = [np.array([0.0, 0.0, 1.0])]
    radii = [0.5]   # ~28°
    lat = np.array([math.pi / 2])   # exactly north pole
    lon = np.array([0.0])
    label = cap_label(lat, lon, centers, radii)
    assert label[0] == 0


# ---------------------------------------------------------------------------
# fill_ic
# ---------------------------------------------------------------------------

def test_fill_ic_h_strictly_positive(lat_lon, grid_params):
    lat, lon = lat_lon
    h, _, _, _ = _call_fill_ic(7, lat, lon, grid_params)
    assert np.all(h > 0.0)
    assert np.all(np.isfinite(h))


def test_fill_ic_momentum_finite(lat_lon, grid_params):
    lat, lon = lat_lon
    _, mx, my, mz = _call_fill_ic(3, lat, lon, grid_params)
    for arr in (mx, my, mz):
        assert np.all(np.isfinite(arr))


def test_fill_ic_determinism(lat_lon, grid_params):
    lat, lon = lat_lon
    h1, m1, _, _ = _call_fill_ic(3, lat, lon, grid_params)
    h2, m2, _, _ = _call_fill_ic(3, lat, lon, grid_params)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(m1, m2)


def test_fill_ic_different_seeds_differ(lat_lon, grid_params):
    lat, lon = lat_lon
    h0, _, _, _ = _call_fill_ic(0, lat, lon, grid_params)
    h1, _, _, _ = _call_fill_ic(1, lat, lon, grid_params)
    assert not np.array_equal(h0, h1)


def test_fill_ic_momentum_tangent_to_sphere(lat_lon, grid_params):
    """Cell-centre 3-D momentum must be tangent: ⟨(mx, my, mz), r̂⟩ ≈ 0."""
    lat, lon = lat_lon
    _, mx, my, mz = _call_fill_ic(5, lat, lon, grid_params)
    cos_lat = np.cos(lat)
    rx = cos_lat * np.cos(lon)
    ry = cos_lat * np.sin(lon)
    rz = np.sin(lat)
    radial_component = mx * rx + my * ry + mz * rz
    np.testing.assert_allclose(radial_component, 0.0, atol=1e-12)


def test_fill_ic_aa_softens_boundary(lat_lon, grid_params):
    """Sub-cell AA produces more unique h values than cell-centre sampling.

    With ``S=1`` every cell takes one of ``K + 1`` cap states. With
    ``S=4`` the boundary cells get fractional means → strictly more
    unique values.
    """
    lat, lon = lat_lon
    h_aa, _, _, _ = _call_fill_ic(11, lat, lon, grid_params, sub_samples=4)
    h_nn, _, _, _ = _call_fill_ic(11, lat, lon, grid_params, sub_samples=1)
    assert len(np.unique(h_nn)) <= K_DEFAULT + 1
    assert len(np.unique(h_aa)) > len(np.unique(h_nn))


def test_fill_ic_mass_balance(lat_lon, grid_params):
    """Area-weighted mean h equals the AA cell-mean of cap+background h_i."""
    lat, lon = lat_lon
    xlower, ylower, dx, dy = grid_params

    sub_samples = 4
    h, _, _, _ = _call_fill_ic(0, lat, lon, grid_params, sub_samples=sub_samples)

    aux = compute_aux(NX, NY, xlower, ylower, dx, dy)
    kappa = aux[0]
    h_mean_actual = np.sum(h * kappa) / np.sum(kappa)

    states, centers, radii, background = sample_caps(
        0, K_DEFAULT, DELTA_DEFAULT,
    )
    state_table = states + [background]
    lat_s, lon_s = subcell_centers_latlon(
        NX, NY, xlower, ylower, dx, dy, sub_samples,
    )
    label = cap_label(lat_s, lon_s, centers, radii)
    h_sub = np.empty(label.shape)
    for k, s in enumerate(state_table):
        h_sub[label == k] = s["h"]
    h_expected = h_sub.mean(axis=(-2, -1))
    h_mean_expected = np.sum(h_expected * kappa) / np.sum(kappa)

    np.testing.assert_allclose(h_mean_actual, h_mean_expected, rtol=1e-10)
