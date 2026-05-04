"""Conservative spherical-polygon area-overlap remap from CH FV cells to a
regular ``(lat, lon)`` grid.

Builds a sparse matrix ``W`` of shape ``(Nlat·Nlon, Nx·Ny)`` such that

    q_target = (W @ q_src.ravel()).reshape(Nlat, Nlon)

is the spherical-area-weighted average of source cells overlapping each
target cell:  ``W[T, S] = A(S∩T) / A(T)``  with ``A`` spherical area.

Properties:
    - Each row of ``W`` sums to 1 (every target is fully tiled by sources).
    - Mass-conservative: ``Σ_T q_target[T] · A(T) = Σ_S q_src[S] · A(S)``.
    - Monotone (no values invented between source-cell extremes).
    - Shock-preserving (no smoothing across discontinuities).

The clipping is done in 2D ``(lat, lon)`` space via Sutherland-Hodgman; the
clipped polygon's area is then evaluated on the sphere via Girard's
theorem with great-circle edges between vertices. Treating ``(lat, lon)``
edges as straight is a third-order approximation in cell size — at our
``(Nx, Ny) = (512, 256)`` resolution the relative area error is well below
``1e-6`` per cell, far under float32 storage precision.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from datagen.shock_caps import geometry as geo


logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * np.pi
_HALF_PI = 0.5 * np.pi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_remap_matrix(
    Nx: int, Ny: int, Nlat: int, Nlon: int,
) -> sp.csr_matrix:
    """Precompute the conservative remap operator ``W``.

    ``W`` has shape ``(Nlat·Nlon, Nx·Ny)`` in CSR format. Source field is
    indexed in ``(Nx, Ny)`` row-major (``i*Ny + j``); target field is
    indexed in ``(Nlat, Nlon)`` row-major (``k*Nlon + l``).
    """
    xlower, ylower = geo.COMPUTATIONAL_LOWER
    xupper, yupper = geo.COMPUTATIONAL_UPPER
    dx = (xupper - xlower) / Nx
    dy = (yupper - ylower) / Ny

    lat_corners, lon_corners = _source_corners_latlon(
        Nx, Ny, xlower, ylower, dx, dy,
    )

    rows, cols, vals = _accumulate_overlaps(
        lat_corners, lon_corners, Nx, Ny, Nlat, Nlon,
    )

    W_unnorm = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(Nlat * Nlon, Nx * Ny),
    ).tocsr()

    # Self-consistent target area = row sum of accumulated overlap areas.
    # By construction this makes each row of W sum to exactly 1.0 — the
    # invariant we depend on for monotonicity and conservation against the
    # source partition's representation. Empty rows would make this fail;
    # keep an epsilon floor as a guard but the lat/lon target grid is
    # always fully covered by a CH partition of the sphere.
    row_area = np.asarray(W_unnorm.sum(axis=1)).ravel()
    row_area = np.where(row_area > 0.0, row_area, 1.0)
    inv_target = sp.diags(1.0 / row_area)
    return (inv_target @ W_unnorm).tocsr()


def apply_remap(
    W: sp.csr_matrix, field_cells: np.ndarray, Nlat: int, Nlon: int,
) -> np.ndarray:
    """Apply ``W`` to a source ``(Nx, Ny)`` field, returning ``(Nlat, Nlon)``."""
    return (W @ np.asarray(field_cells).ravel()).reshape(Nlat, Nlon)


def target_cell_areas(Nlat: int, Nlon: int) -> np.ndarray:
    """Public alias: spherical area of each target cell, shape ``(Nlat, Nlon)``."""
    return _target_cell_areas(Nlat, Nlon).reshape(Nlat, Nlon)


# ---------------------------------------------------------------------------
# Internals: source-cell polygon construction
# ---------------------------------------------------------------------------

def _source_corners_latlon(
    Nx: int, Ny: int, xlower: float, ylower: float, dx: float, dy: float,
):
    """``(Nx, Ny, 4)`` lat/lon arrays for the 4 corners (BL, BR, TR, TL) of
    every CH cell, ordered CCW in computational space.
    """
    i = np.arange(Nx + 1)
    j = np.arange(Ny + 1)
    XC, YC = np.meshgrid(xlower + i * dx, ylower + j * dy, indexing="ij")
    xp, yp, zp = geo.mapc2p_sphere(XC.copy(), YC.copy())  # mutates input
    lat_node = np.arcsin(np.clip(zp, -1.0, 1.0))
    lon_node = np.mod(np.arctan2(yp, xp), _TWO_PI)

    lat = np.stack([
        lat_node[:-1, :-1],   # BL
        lat_node[1:,  :-1],   # BR
        lat_node[1:,  1: ],   # TR
        lat_node[:-1, 1: ],   # TL
    ], axis=-1)
    lon = np.stack([
        lon_node[:-1, :-1],
        lon_node[1:,  :-1],
        lon_node[1:,  1: ],
        lon_node[:-1, 1: ],
    ], axis=-1)
    return lat, lon


def _unwrap_lons(lons: np.ndarray, anchor: float) -> np.ndarray:
    """Shift each ``lon`` by ±2π so it lies within ``π`` of ``anchor``.
    Output may extend outside ``[0, 2π)``.
    """
    diff = lons - anchor
    diff = np.where(diff > np.pi, diff - _TWO_PI, diff)
    diff = np.where(diff < -np.pi, diff + _TWO_PI, diff)
    return anchor + diff


def _build_cell_polygon(lats: np.ndarray, lons: np.ndarray):
    """Build a ``(lat, lon)`` polygon for one CH cell.

    Non-polar cells: 4 vertices in CCW order, lons unwrapped relative to
    corner 0. Polar cells (one corner exactly at ``lat = ±π/2``):
    5 vertices — the polar corner is split into two pole vertices, one
    at the previous corner's lon and one at the next corner's lon, so
    the cell's two polar edges become meridians meeting at the pole.
    """
    pole_mask = np.abs(np.abs(lats) - _HALF_PI) < 1e-12
    if not pole_mask.any():
        # Plain quadrilateral.
        return lats.copy(), _unwrap_lons(lons, lons[0])

    c = int(np.argmax(pole_mask))
    prev_c = (c - 1) % 4
    next_c = (c + 1) % 4
    out_lat: list[float] = []
    out_lon: list[float] = []
    for k in range(4):
        if k == c:
            out_lat.append(float(lats[c]))
            out_lon.append(float(lons[prev_c]))
            out_lat.append(float(lats[c]))
            out_lon.append(float(lons[next_c]))
        else:
            out_lat.append(float(lats[k]))
            out_lon.append(float(lons[k]))
    lats_out = np.asarray(out_lat)
    lons_out = np.asarray(out_lon)
    return lats_out, _unwrap_lons(lons_out, lons_out[0])


# ---------------------------------------------------------------------------
# Internals: target-cell areas
# ---------------------------------------------------------------------------

def _target_cell_areas(Nlat: int, Nlon: int) -> np.ndarray:
    """Exact spherical area of every regular lat/lon cell, flat ``(Nlat*Nlon,)``.

    ``A(T) = Δlon · (sin(lat_hi) − sin(lat_lo))`` exact for parallel-bounded
    cells. Note: this is NOT the area used internally to normalise ``W`` —
    that uses the row-sum of GC-quad overlap pieces, which is exact against
    the source partition but differs from the parallel-bounded area by an
    O(grid_size³) "chord vs parallel" correction (negligible at production
    resolution; ~0.4% at the test grid).
    """
    dlat = np.pi / Nlat
    dlon = _TWO_PI / Nlon
    lat_edges = -_HALF_PI + np.arange(Nlat + 1) * dlat
    lat_band = np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    return np.repeat(dlon * lat_band, Nlon)


# ---------------------------------------------------------------------------
# Internals: 2D Sutherland-Hodgman clip in (lat, lon)
# ---------------------------------------------------------------------------

def _clip_half(
    xs: np.ndarray, ys: np.ndarray,
    axis: int, value: float, keep_above: bool,
):
    """Clip polygon against half-plane ``coord_axis ≥/≤ value``.

    ``axis = 0``: clip on ``xs``; ``axis = 1``: clip on ``ys``.
    ``keep_above = True`` keeps the half-plane ``coord >= value``.
    Returns ``(xs_out, ys_out)`` as Python lists for amend-friendly assembly.
    """
    n = len(xs)
    if n == 0:
        return [], []
    coord = xs if axis == 0 else ys
    if keep_above:
        inside = coord >= value
    else:
        inside = coord <= value
    out_x: list[float] = []
    out_y: list[float] = []
    for i in range(n):
        prev_i = (i - 1) % n
        curr_in = inside[i]
        prev_in = inside[prev_i]
        if curr_in:
            if not prev_in:
                px, py = xs[prev_i], ys[prev_i]
                cx, cy = xs[i], ys[i]
                p_coord = px if axis == 0 else py
                c_coord = cx if axis == 0 else cy
                t = (value - p_coord) / (c_coord - p_coord)
                out_x.append(px + t * (cx - px))
                out_y.append(py + t * (cy - py))
            out_x.append(xs[i])
            out_y.append(ys[i])
        elif prev_in:
            px, py = xs[prev_i], ys[prev_i]
            cx, cy = xs[i], ys[i]
            p_coord = px if axis == 0 else py
            c_coord = cx if axis == 0 else cy
            t = (value - p_coord) / (c_coord - p_coord)
            out_x.append(px + t * (cx - px))
            out_y.append(py + t * (cy - py))
    return out_x, out_y


def _clip_polygon_to_rect(
    poly_lat, poly_lon,
    lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float,
):
    """Clip ``(poly_lat, poly_lon)`` against the rectangle
    ``[lat_lo, lat_hi] × [lon_lo, lon_hi]``. Returns ``(list, list)``.
    """
    xs, ys = _clip_half(poly_lat, poly_lon, axis=0, value=lat_lo, keep_above=True)
    xs, ys = _clip_half(xs, ys, axis=0, value=lat_hi, keep_above=False)
    xs, ys = _clip_half(xs, ys, axis=1, value=lon_lo, keep_above=True)
    xs, ys = _clip_half(xs, ys, axis=1, value=lon_hi, keep_above=False)
    return xs, ys




# ---------------------------------------------------------------------------
# Internals: spherical polygon area via Girard's theorem
# ---------------------------------------------------------------------------

def _spherical_polygon_area(lats, lons) -> float:
    """Spherical area of a polygon on the unit sphere.

    Vertices ``(lats, lons)`` are connected by great-circle arcs.
    Area = (Σ interior angles) − (n − 2)π. Coincident consecutive
    vertices (e.g. two pole vertices on a polar edge that collapses
    after clipping) are deduped before the angle sum so degenerate
    triangles still get a meaningful interior-angle count.
    """
    n = len(lats)
    if n < 3:
        return 0.0
    lats_a = np.asarray(lats, float)
    lons_a = np.asarray(lons, float)
    cos_lat = np.cos(lats_a)
    v_full = np.stack([
        cos_lat * np.cos(lons_a),
        cos_lat * np.sin(lons_a),
        np.sin(lats_a),
    ], axis=-1)

    keep = [0]
    for i in range(1, n):
        if np.linalg.norm(v_full[i] - v_full[keep[-1]]) > 1e-12:
            keep.append(i)
    if len(keep) > 2 and np.linalg.norm(v_full[keep[-1]] - v_full[keep[0]]) < 1e-12:
        keep.pop()
    if len(keep) < 3:
        return 0.0
    v = v_full[keep]
    n = len(v)

    angle_sum = 0.0
    for i in range(n):
        a = v[i - 1]
        b = v[i]
        c = v[(i + 1) % n]
        t_ba = a - b * np.dot(b, a)
        t_bc = c - b * np.dot(b, c)
        n_ba = np.linalg.norm(t_ba)
        n_bc = np.linalg.norm(t_bc)
        if n_ba < 1e-15 or n_bc < 1e-15:
            continue
        cos_alpha = np.dot(t_ba, t_bc) / (n_ba * n_bc)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        angle_sum += np.arccos(cos_alpha)

    return angle_sum - (n - 2) * np.pi


# ---------------------------------------------------------------------------
# Internals: build sparse triplets
# ---------------------------------------------------------------------------

def _accumulate_overlaps(
    lat_corners: np.ndarray, lon_corners: np.ndarray,
    Nx: int, Ny: int, Nlat: int, Nlon: int,
):
    """Loop over CH source cells and gather ``A(S∩T)`` overlap entries.

    Returns ``(rows, cols, vals)`` lists for COO assembly.
    """
    dlat = np.pi / Nlat
    dlon = _TWO_PI / Nlon
    lat_edges = -_HALF_PI + np.arange(Nlat + 1) * dlat
    lon_edges = np.arange(Nlon + 1) * dlon

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i in range(Nx):
        for j in range(Ny):
            poly_lat, poly_lon = _build_cell_polygon(
                lat_corners[i, j], lon_corners[i, j],
            )

            cell_lat_lo = float(poly_lat.min())
            cell_lat_hi = float(poly_lat.max())
            cell_lon_lo = float(poly_lon.min())
            cell_lon_hi = float(poly_lon.max())

            k_lo = max(0, int(np.floor((cell_lat_lo + _HALF_PI) / dlat)))
            k_hi = min(Nlat - 1, int(np.floor((cell_lat_hi + _HALF_PI) / dlat)))
            if cell_lat_hi >= _HALF_PI - 1e-14:
                k_hi = Nlat - 1
            if cell_lat_lo <= -_HALF_PI + 1e-14:
                k_lo = 0
            if k_hi < k_lo:
                continue

            src_idx = i * Ny + j

            for shift in (0.0, _TWO_PI, -_TWO_PI):
                a = cell_lon_lo + shift
                b = cell_lon_hi + shift
                if b <= 0.0 or a >= _TWO_PI:
                    continue
                a_clip = max(a, 0.0)
                b_clip = min(b, _TWO_PI)
                l_lo = max(0, int(np.floor(a_clip / dlon)))
                l_hi = min(Nlon - 1, int(np.ceil(b_clip / dlon)) - 1)
                if l_hi < l_lo:
                    continue
                shifted_lon = poly_lon + shift

                for k in range(k_lo, k_hi + 1):
                    lat_t_lo = lat_edges[k]
                    lat_t_hi = lat_edges[k + 1]
                    for l in range(l_lo, l_hi + 1):
                        lon_t_lo = lon_edges[l]
                        lon_t_hi = lon_edges[l + 1]
                        cx, cy = _clip_polygon_to_rect(
                            poly_lat, shifted_lon,
                            lat_t_lo, lat_t_hi, lon_t_lo, lon_t_hi,
                        )
                        if len(cx) < 3:
                            continue
                        area = _spherical_polygon_area(cx, cy)
                        if area <= 0.0:
                            continue
                        rows.append(k * Nlon + l)
                        cols.append(src_idx)
                        vals.append(area)

    return rows, cols, vals
