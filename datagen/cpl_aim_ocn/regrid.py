"""Optional cs32 → regular lat/lon regridder for downstream consumers.

The canonical Zarr produced by :mod:`datagen.cpl_aim_ocn.zarr_writer`
keeps the cubed-sphere grid native (dims ``face, j, i``) — this avoids
the smoothing and cube-edge artefacts that interpolation introduces.
Downstream tooling that expects a regular lat/lon grid can use this
utility to materialise such a view from a cs32 Zarr.

Implementation
--------------
Pure-scipy: build a ``cKDTree`` on the cs32 cell centres
(``XC``, ``YC`` aux coords already stored in the Zarr) and look up
either:

* the **nearest source cell** (``method="nearest"``, default) — fast,
  exact for piecewise-constant fields, gives small staircase artefacts
  on smooth fields;
* the **k=4 inverse-distance weighted average** (``method="idw"``) —
  smoother, comparable in quality to bilinear at ~3.75° → 2.8°
  (cs32 → 128×64) without needing xesmf or ESMF at all.

The mapping is computed in 3-D Cartesian coordinates (lon/lat → xyz on
the unit sphere) so the great-circle distance metric is correct at the
poles. The KDTree is built once and the same weights are applied to
every variable / time step.

This module imports ``scipy`` lazily — the rest of the package does
not depend on it for run-time operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

# ─── Defaults ────────────────────────────────────────────────────────────────

#: Default regular-lat-lon target grid (matches the existing
#: :mod:`datagen.resample` convention).
DEFAULT_NLAT: int = 64
DEFAULT_NLON: int = 128


# ─── Geometry helpers ────────────────────────────────────────────────────────

def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert (lon, lat) in degrees to unit-sphere Cartesian coords.

    Returns an array with shape ``input_shape + (3,)``. Operating in
    xyz space lets a Euclidean k-NN query stand in for a great-circle
    nearest-neighbour search (correct everywhere including poles, no
    seams at the dateline).
    """
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def _build_target_grid(
    nlat: int, nlon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equispaced target grid (cell centres) covering the full sphere.

    lat ∈ (-90 + δ/2, 90 − δ/2) with δ = 180/nlat;
    lon ∈ ( -180 + δ/2,  180 − δ/2) with δ = 360/nlon.

    Both 1-D arrays. Use the same ``regular_lat_grid`` /
    ``regular_lon_grid`` convention as :mod:`datagen.resample` (cell
    centres, not edges).
    """
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lat = np.linspace(-90.0 + dlat / 2, 90.0 - dlat / 2, nlat)
    lon = np.linspace(-180.0 + dlon / 2, 180.0 - dlon / 2, nlon)
    return lat, lon


# ─── Weight precomputation ───────────────────────────────────────────────────

@dataclass(frozen=True)
class RegridWeights:
    """Precomputed neighbour indices + weights mapping cs32 → (lat, lon).

    ``indices`` shape: ``(nlat, nlon, k)`` — flat-index into the cs32
    source array (so a 6×32×32 grid is indexed 0..6143 in
    face-major C order).

    ``weights`` shape: ``(nlat, nlon, k)`` — sums to 1 along axis -1.
    For ``method="nearest"`` the weights are ``[1.0]`` and ``k=1``.
    """
    indices: np.ndarray   # int64
    weights: np.ndarray   # float64
    nlat: int
    nlon: int
    method: str

    @property
    def k(self) -> int:
        return int(self.weights.shape[-1])


def build_weights(
    xc_face_j_i: np.ndarray,
    yc_face_j_i: np.ndarray,
    *,
    nlat: int = DEFAULT_NLAT,
    nlon: int = DEFAULT_NLON,
    method: Literal["nearest", "idw"] = "nearest",
    k: int = 4,
) -> RegridWeights:
    """Build the index + weight arrays for the cs32 → lat/lon mapping.

    Parameters
    ----------
    xc_face_j_i, yc_face_j_i
        cs32 cell-centre lon, lat in degrees, shape ``(face=6, j=32, i=32)``.
        These are the ``XC``, ``YC`` arrays stored in the canonical
        Zarr's aux coords by :mod:`xmitgcm`.
    nlat, nlon
        Target grid resolution. Default 64 × 128 (~2.8°).
    method
        ``"nearest"`` (default): single-neighbour assignment.
        ``"idw"``: inverse-distance-weighted average over ``k`` nearest.
    k
        Number of neighbours for IDW. Ignored for ``method="nearest"``.

    Returns
    -------
    A :class:`RegridWeights` instance, reusable across variables and
    time steps via :func:`apply_weights`.
    """
    if method not in ("nearest", "idw"):
        raise ValueError(f"Unknown regrid method: {method!r}")

    from scipy.spatial import cKDTree

    src_xc = np.asarray(xc_face_j_i, dtype=np.float64).reshape(-1)
    src_yc = np.asarray(yc_face_j_i, dtype=np.float64).reshape(-1)
    if src_xc.shape != src_yc.shape:
        raise ValueError(
            f"XC and YC must have matching shape (got {src_xc.shape} vs {src_yc.shape})"
        )

    src_xyz = _lonlat_to_xyz(src_xc, src_yc)
    tree = cKDTree(src_xyz)

    lat, lon = _build_target_grid(nlat, nlon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)            # both (nlat, nlon)
    tgt_xyz = _lonlat_to_xyz(lon_grid, lat_grid).reshape(-1, 3)

    if method == "nearest":
        _, idx = tree.query(tgt_xyz, k=1)
        idx = idx.reshape(nlat, nlon, 1).astype(np.int64)
        w = np.ones_like(idx, dtype=np.float64)
        return RegridWeights(indices=idx, weights=w,
                             nlat=nlat, nlon=nlon, method="nearest")

    # IDW
    if k < 2:
        raise ValueError(f"IDW requires k ≥ 2; got {k}")
    dist, idx = tree.query(tgt_xyz, k=k)
    # Guard against zero distance (target == source) — set weight to 1.
    zero_mask = dist <= 1e-12
    eps = 1e-12
    w = 1.0 / (dist + eps)
    w[zero_mask] = 0.0
    # If any row had a zero-distance hit, give that neighbour all the weight.
    has_zero = zero_mask.any(axis=-1)
    w[has_zero] = zero_mask[has_zero].astype(np.float64)
    # Normalise rows to sum to 1.
    w = w / w.sum(axis=-1, keepdims=True)

    idx = idx.reshape(nlat, nlon, k).astype(np.int64)
    w = w.reshape(nlat, nlon, k).astype(np.float64)
    return RegridWeights(indices=idx, weights=w,
                         nlat=nlat, nlon=nlon, method="idw")


# ─── Apply weights ───────────────────────────────────────────────────────────

def apply_weights(
    field_face_j_i: np.ndarray,
    weights: RegridWeights,
) -> np.ndarray:
    """Regrid a single source field to the target lat/lon grid.

    ``field_face_j_i`` may be shape ``(face, j, i)`` or have any number
    of leading dims (e.g. ``(time, face, j, i)`` or
    ``(time, sigma, face, j, i)``); the trailing three are the cs32
    horizontal dims and are flattened together internally. The output
    has the same leading dims with the trailing ``(face, j, i)``
    replaced by ``(nlat, nlon)``.
    """
    src = np.asarray(field_face_j_i)
    if src.shape[-3:-2] == (1,):
        # In case the caller has already flattened face into a singleton
        # somehow — the algorithm only cares about the flattened source
        # length matching the weights.
        pass
    n_horiz = int(np.prod(src.shape[-3:]))
    expected = weights.indices.max() + 1
    if n_horiz < expected:
        raise ValueError(
            f"Field horizontal size {n_horiz} smaller than expected "
            f"(weights index up to {expected})"
        )
    leading_shape = src.shape[:-3]
    src_flat = src.reshape(*leading_shape, n_horiz)        # (..., n_horiz)
    # Gather the k-neighbour values: (..., nlat, nlon, k)
    gathered = src_flat[..., weights.indices]
    # Multiply by weights and sum over k.
    return (gathered * weights.weights).sum(axis=-1)


# ─── High-level convenience ──────────────────────────────────────────────────

def regrid_to_latlon(
    cs32_zarr,
    *,
    nlat: int = DEFAULT_NLAT,
    nlon: int = DEFAULT_NLON,
    method: Literal["nearest", "idw"] = "nearest",
    k: int = 4,
    out_path: Path | None = None,
    variables: Iterable[str] | None = None,
):
    """Materialise a regular lat/lon view of a cs32 Zarr (or in-memory ds).

    Parameters
    ----------
    cs32_zarr
        Either a path to a Zarr written by ``write_cs32_zarr``, or an
        already-loaded ``xarray.Dataset`` with that schema.
    nlat, nlon
        Target grid resolution.
    method, k
        See :func:`build_weights`.
    out_path
        If supplied, the regridded Dataset is also written to a new
        Zarr at this path (mode ``"w"``).
    variables
        Optional subset of data-variable names to regrid (everything
        else is dropped). By default, every var with cs32 horizontal
        dims is regridded.

    Returns
    -------
    The regridded ``xarray.Dataset``.
    """
    import xarray as xr

    if isinstance(cs32_zarr, (str, Path)):
        ds = xr.open_zarr(cs32_zarr)
    else:
        ds = cs32_zarr

    # Locate the horizontal cs32 dims. xmitgcm's cs reader uses
    # ``face`` × ``j`` × ``i``; the writer keeps that. Be defensive
    # against future renames.
    face_dim = next(
        (d for d in ds.dims if d.lower() in ("face", "tile")), None
    )
    if face_dim is None:
        raise ValueError(
            "Source dataset has no `face` dimension — is this a cs32 Zarr?"
        )
    j_dim = next((d for d in ds.dims if d.lower() == "j"), None)
    i_dim = next((d for d in ds.dims if d.lower() == "i"), None)
    if j_dim is None or i_dim is None:
        raise ValueError(
            "Source dataset missing `j`/`i` cs32 dims — is this a cs32 Zarr?"
        )
    if "XC" not in ds.coords or "YC" not in ds.coords:
        raise ValueError(
            "Source dataset missing XC/YC cell-centre coords — "
            "regridder cannot infer source positions"
        )
    horizontal_dims = (face_dim, j_dim, i_dim)

    xc = ds["XC"].values    # (face, j, i)
    yc = ds["YC"].values
    weights = build_weights(xc, yc, nlat=nlat, nlon=nlon,
                            method=method, k=k)

    target_lat, target_lon = _build_target_grid(nlat, nlon)

    if variables is not None:
        data_vars_to_regrid = list(variables)
    else:
        data_vars_to_regrid = [
            v for v in ds.data_vars
            if all(d in ds[v].dims for d in horizontal_dims)
        ]

    new_vars = {}
    for v in data_vars_to_regrid:
        arr = ds[v]
        # Move (face, j, i) to the last three positions.
        other_dims = [d for d in arr.dims if d not in horizontal_dims]
        arr = arr.transpose(*other_dims, *horizontal_dims)
        regridded = apply_weights(arr.values, weights)
        new_vars[v] = (tuple(other_dims) + ("lat", "lon"), regridded,
                        {**arr.attrs})

    # Bring forward only the coords that survive regridding: time-like
    # and vertical 1-D coords (any nD coord on the cs32 horizontal grid
    # is dropped — XC/YC/XG/YG don't apply post-regrid).
    cs_coords = {"XC", "YC", "XG", "YG", face_dim, j_dim, i_dim}
    forwarded_coords = {
        c: ds.coords[c] for c in ds.coords
        if c not in cs_coords and ds[c].ndim <= 1
    }
    out = xr.Dataset(
        data_vars=new_vars,
        coords={
            "lat": ("lat", target_lat),
            "lon": ("lon", target_lon),
            **forwarded_coords,
        },
        attrs={**ds.attrs, "regrid_method": method, "regrid_k": k},
    )

    if out_path is not None:
        out_path = Path(out_path)
        if out_path.exists():
            import shutil
            shutil.rmtree(out_path)
        out.to_zarr(out_path, mode="w", consolidated=True)

    return out
