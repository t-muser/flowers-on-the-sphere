"""cs32 → lat/lon regridding for the MITgcm global-ocean dataset.

Wraps :mod:`datagen.cpl_aim_ocn.regrid` (which already does the spatial
k-NN/IDW math) and adds:

* **Vector rotation** for ``(u_k2, v_k2)`` from face-aligned coordinates
  to geographic ``(east, north)`` using the ``angle_cs / angle_sn`` arrays
  in ``grid.zarr``. The rotation is a local linear combine; doing it on
  the native cs grid before scalar regrid is correct in the limit where
  the source neighbourhood spans a single face. At cs32 resolution that
  approximation is adequate everywhere except near face corners (8
  cells), and even there the residual error is dominated by the
  underlying coarse resolution rather than the rotation.

* **Per-field static appliers** that regrid ``depth``, ``mask_k1``,
  ``mask_k2``, ``mask_eta`` once into lat/lon (cached in the dataloader).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from datagen.cpl_aim_ocn.regrid import (
    DEFAULT_NLAT,
    DEFAULT_NLON,
    RegridWeights,
    apply_weights,
    build_weights as _build_weights_scalar,
)


def rotate_uv_to_geographic(
    u: np.ndarray,
    v: np.ndarray,
    angle_cs: np.ndarray,
    angle_sn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate face-aligned ``(u, v)`` into geographic ``(east, north)``.

    MITgcm's ``AngleCS = cos(α)``, ``AngleSN = sin(α)`` where ``α`` is the
    rotation angle from local face axes to geographic east/north. Inverse
    of MITgcm's own ``rotate_uv2en.F`` convention used by the diagnostics
    package.

    Inputs may be (..., face, y, x); ``angle_cs`` and ``angle_sn`` are
    ``(face, y, x)`` and broadcast over the leading dims.
    """
    u_east  = angle_cs * u - angle_sn * v
    v_north = angle_sn * u + angle_cs * v
    return u_east, v_north


@dataclass(frozen=True)
class GlobalOceanLatLon:
    """Static lat/lon-side grid info needed by the dataloader.

    All arrays are on the lat/lon target grid built by :func:`build`.
    """
    nlat: int
    nlon: int
    weights: RegridWeights
    angle_cs: np.ndarray   # (face, y, x), source-side, used pre-regrid
    angle_sn: np.ndarray   # (face, y, x), source-side
    depth_ll: np.ndarray   # (nlat, nlon)
    mask_k1_ll: np.ndarray # (nlat, nlon) bool
    mask_k2_ll: np.ndarray # (nlat, nlon) bool
    mask_eta_ll: np.ndarray# (nlat, nlon) bool
    # Optional per-level masks for the 3-D variant. None when the grid.zarr
    # was produced by an older extract_grid that didn't write them.
    mask_c_3d_ll: np.ndarray | None = None  # (Nr, nlat, nlon) bool
    mask_w_3d_ll: np.ndarray | None = None  # (Nr, nlat, nlon) bool
    # Source-side per-level masks (cs grid). Used by ``apply_dynamic_3d`` to
    # impute land cells before regridding so the IDW kernel doesn't bleed
    # MITgcm's 0.0 land-fill into wet cells across continental margins.
    mask_c_3d_src: np.ndarray | None = None  # (Nr, face, y, x) bool
    mask_w_3d_src: np.ndarray | None = None  # (Nr, face, y, x) bool


def build(
    grid_zarr: str | Path,
    *,
    nlat: int = DEFAULT_NLAT,
    nlon: int = DEFAULT_NLON,
    method: Literal["nearest", "idw"] = "idw",
    k: int = 4,
    mask_threshold: float = 0.5,
) -> GlobalOceanLatLon:
    """Open ``grid.zarr``, build cs → lat/lon weights and regrid the
    static fields (depth, masks). Cheap; the dataloader calls this once.
    """
    grid = xr.open_zarr(str(grid_zarr))
    xc = grid["xc"].values
    yc = grid["yc"].values
    weights = _build_weights_scalar(xc, yc, nlat=nlat, nlon=nlon,
                                    method=method, k=k)

    depth_ll = apply_weights(grid["depth"].values, weights)
    # Regrid bool masks as floats in [0,1] then threshold.
    mask_k1_ll  = apply_weights(grid["mask_k1"].values.astype(np.float32),
                                weights) > mask_threshold
    mask_k2_ll  = apply_weights(grid["mask_k2"].values.astype(np.float32),
                                weights) > mask_threshold
    mask_eta_ll = apply_weights(grid["mask_eta"].values.astype(np.float32),
                                weights) > mask_threshold

    # Optional 3-D per-level masks (only present in grid.zarr files written
    # after the 3-D extract_grid extension).
    mask_c_3d_ll: np.ndarray | None = None
    mask_w_3d_ll: np.ndarray | None = None
    mask_c_3d_src: np.ndarray | None = None
    mask_w_3d_src: np.ndarray | None = None
    if "mask_c_3d" in grid:
        mask_c_3d_src = grid["mask_c_3d"].values.astype(bool)
        mask_c_3d_ll = apply_weights(
            mask_c_3d_src.astype(np.float32), weights
        ) > mask_threshold
    if "mask_w_3d" in grid:
        mask_w_3d_src = grid["mask_w_3d"].values.astype(bool)
        mask_w_3d_ll = apply_weights(
            mask_w_3d_src.astype(np.float32), weights
        ) > mask_threshold

    return GlobalOceanLatLon(
        nlat=nlat, nlon=nlon, weights=weights,
        angle_cs=grid["angle_cs"].values.astype(np.float32),
        angle_sn=grid["angle_sn"].values.astype(np.float32),
        depth_ll=depth_ll.astype(np.float32),
        mask_k1_ll=mask_k1_ll, mask_k2_ll=mask_k2_ll, mask_eta_ll=mask_eta_ll,
        mask_c_3d_ll=mask_c_3d_ll, mask_w_3d_ll=mask_w_3d_ll,
        mask_c_3d_src=mask_c_3d_src, mask_w_3d_src=mask_w_3d_src,
    )


# Default field order in the global-ocean zarrs' `data` variable.
FIELD_ORDER: tuple[str, ...] = (
    "theta_k1", "salt_k1", "u_k2", "v_k2", "eta",
)


def apply_dynamic(
    data: np.ndarray,
    grid_ll: GlobalOceanLatLon,
    *,
    field_order: tuple[str, ...] = FIELD_ORDER,
) -> np.ndarray:
    """Regrid the (time, field, face, y, x) data tensor to (time, field, nlat, nlon).

    Velocities ``u_k2`` and ``v_k2`` are rotated to geographic east/north
    on the native cs grid before scalar regrid; theta, salt, eta are
    treated as scalars.
    """
    if data.ndim != 5:
        raise ValueError(f"expected (time, field, face, y, x); got {data.shape}")
    if data.shape[1] != len(field_order):
        raise ValueError(
            f"field axis size {data.shape[1]} != len(field_order) {len(field_order)}"
        )

    try:
        iu = field_order.index("u_k2")
        iv = field_order.index("v_k2")
    except ValueError as e:
        raise ValueError(f"field_order must contain u_k2 and v_k2; got {field_order}") from e

    src = data.copy()
    u_east, v_north = rotate_uv_to_geographic(
        src[:, iu], src[:, iv],
        grid_ll.angle_cs, grid_ll.angle_sn,
    )
    src[:, iu] = u_east
    src[:, iv] = v_north

    # apply_weights flattens trailing 3 dims (face, y, x) → (n_horiz),
    # gathers, sums. Works for any leading-dims shape.
    return apply_weights(src, grid_ll.weights).astype(np.float32)


# Field-name → mask-array mapping, in lat/lon space.
def field_masks_ll(grid_ll: GlobalOceanLatLon) -> dict[str, np.ndarray]:
    """Map each field to its bool mask on the lat/lon target grid."""
    return {
        "theta_k1": grid_ll.mask_k1_ll,
        "salt_k1":  grid_ll.mask_k1_ll,
        "u_k2":     grid_ll.mask_k2_ll,
        "v_k2":     grid_ll.mask_k2_ll,
        "eta":      grid_ll.mask_eta_ll,
    }


# Per-variable schema written by the 3-D solver branch. ``u`` is the
# geographic east-component after rotation; ``v`` is north.
FIELDS_3D: tuple[str, ...] = ("theta", "salt", "u", "v")
FIELDS_2D: tuple[str, ...] = ("eta",)


def field_names_3d(level_idx: np.ndarray) -> tuple[str, ...]:
    """Channel names for the level-squashed 3-D output.

    Order: every 3-D variable's levels in ``FIELDS_3D`` order, followed by
    each 2-D field. E.g. for levels ``[1, 5]``:
    ``('theta_k01', 'theta_k05', 'salt_k01', 'salt_k05',
       'u_k01',     'u_k05',     'v_k01',    'v_k05', 'eta')``.
    """
    names: list[str] = []
    for var in FIELDS_3D:
        for lvl in level_idx:
            names.append(f"{var}_k{int(lvl):02d}")
    names.extend(FIELDS_2D)
    return tuple(names)


def apply_dynamic_3d(
    fields_3d: dict[str, np.ndarray],
    fields_2d: dict[str, np.ndarray],
    grid_ll: GlobalOceanLatLon,
    *,
    level_idx: np.ndarray | None = None,
    impute_land: bool = True,
) -> np.ndarray:
    """Regrid the per-variable 3-D global-ocean output to a level-squashed
    lat/lon tensor ``(time, C, nlat, nlon)``.

    Mirrors the 2-D :func:`apply_dynamic` interface: depth levels are
    folded into the channel axis so downstream models see a flat channel
    stack. Channel order is given by :func:`field_names_3d`.

    Inputs match the schema of the 3-D ``run.zarr`` written by
    :func:`datagen.mitgcm.global_ocean.solver.write_cubed_sphere_zarr_3d`:

    * ``fields_3d``: ``{'theta', 'salt', 'u', 'v'}``, each
      ``(time, level, face, y, x)``.
    * ``fields_2d``: ``{'eta'}``, ``(time, face, y, x)``.

    The ``(u, v)`` pair is rotated to geographic east/north on the native
    cs grid at every level before regrid; ``theta`` / ``salt`` / ``eta``
    are treated as scalars.

    When ``impute_land`` is True (default) and the source-side per-level
    masks are available in ``grid_ll``, MITgcm's 0.0 land-fill is
    replaced with each variable's per-level wet-cell mean before regrid.
    This stops the IDW kernel from bleeding 0-valued land into wet cells
    near continental margins — particularly visible at deep levels
    where the wet-cell variance is tiny (e.g. salt_k15 ≈ 0.04 psu).
    Land cells are zeroed back out post-regrid via the mask, so this
    only affects values *inside* the wet region.

    ``level_idx`` is the 1-indexed model level for each entry along the
    ``level`` axis of the inputs. When omitted, defaults to
    ``range(1, Nlevel+1)`` (i.e. the inputs are assumed to be the
    first ``Nlevel`` levels).
    """
    if "u" not in fields_3d or "v" not in fields_3d:
        raise ValueError(
            f"fields_3d must contain 'u' and 'v'; got {sorted(fields_3d)}"
        )

    src_3d = {name: np.asarray(arr) for name, arr in fields_3d.items()}
    for name, arr in src_3d.items():
        if arr.ndim != 5:
            raise ValueError(
                f"fields_3d[{name!r}] must be (time, level, face, y, x); "
                f"got shape {arr.shape}"
            )
    Nt, Nlevel = next(iter(src_3d.values())).shape[:2]
    for name, arr in src_3d.items():
        if arr.shape[:2] != (Nt, Nlevel):
            raise ValueError(
                f"fields_3d[{name!r}] has shape {arr.shape}; "
                f"expected leading dims ({Nt}, {Nlevel})"
            )

    if level_idx is None:
        level_idx = np.arange(1, Nlevel + 1, dtype=np.int64)
    levels0 = np.asarray(level_idx, dtype=int) - 1

    # Rotate u/v at every (time, level) — angle_{cs,sn} have shape (face, y, x)
    # and broadcast across the leading two axes.
    u_east, v_north = rotate_uv_to_geographic(
        src_3d["u"], src_3d["v"], grid_ll.angle_cs, grid_ll.angle_sn,
    )
    src_3d = {**src_3d, "u": u_east, "v": v_north}

    # Land-imputation: replace 0-fill with per-variable per-level wet-cell
    # mean before the IDW regrid. We take the mean over wet cells AND time,
    # which is robust when a single window has near-uniform values (e.g. an
    # identity batch). Only runs when the source-side per-level masks are
    # present (i.e. grid.zarr was built by the 3-D extract_grid).
    have_3d_src = (
        grid_ll.mask_c_3d_src is not None
        and grid_ll.mask_w_3d_src is not None
    )
    if impute_land and have_3d_src:
        for var in FIELDS_3D:
            src_mask_full = (
                grid_ll.mask_c_3d_src if var in ("theta", "salt")
                else grid_ll.mask_w_3d_src
            )
            # Copy before mutating so callers' input tensors are never
            # touched. ``u`` / ``v`` came back from ``rotate_uv_to_geographic``
            # as fresh arrays already; ``theta`` / ``salt`` may still alias.
            arr = src_3d[var].copy()
            for k in range(Nlevel):
                wet = src_mask_full[levels0[k]]  # (face, y, x) bool
                if not wet.any():
                    continue
                slab = arr[:, k]  # (time, face, y, x); writable view of arr
                mu = float(slab[..., wet].mean())
                slab[..., ~wet] = mu
            src_3d[var] = arr

    nlat, nlon = grid_ll.nlat, grid_ll.nlon
    n_3d_chans = len(FIELDS_3D) * Nlevel
    n_2d_chans = len(FIELDS_2D)
    out = np.empty((Nt, n_3d_chans + n_2d_chans, nlat, nlon), dtype=np.float32)

    for var_idx, var in enumerate(FIELDS_3D):
        regridded = apply_weights(src_3d[var], grid_ll.weights)  # (Nt, Nlevel, nlat, nlon)
        out[:, var_idx * Nlevel:(var_idx + 1) * Nlevel] = regridded.astype(np.float32)

    for off, name in enumerate(FIELDS_2D):
        arr2 = np.asarray(fields_2d[name])
        # eta wet mask on cs grid = "any level has water" = OR over levels.
        if impute_land and have_3d_src and name == "eta":
            wet_eta = grid_ll.mask_c_3d_src.any(axis=0)  # (face, y, x)
            if wet_eta.any():
                arr2 = arr2.copy()
                mu = float(arr2[..., wet_eta].mean())
                arr2[..., ~wet_eta] = mu
        regridded = apply_weights(arr2, grid_ll.weights)
        out[:, n_3d_chans + off] = regridded.astype(np.float32)

    return out


def field_masks_3d_ll(
    grid_ll: GlobalOceanLatLon,
    level_idx: np.ndarray,
) -> np.ndarray:
    """Build a per-channel mask stack ``(C, nlat, nlon)`` matching
    :func:`apply_dynamic_3d`'s output.

    Channel order follows :func:`field_names_3d`:
    ``theta`` and ``salt`` use the C-grid mask at each level;
    ``u`` and ``v`` use the W-grid mask at each level (the same
    "both-edges-wet" convention as the 2-D ``mask_k2``);
    ``eta`` uses ``mask_eta`` (column-has-water).

    Requires ``grid_ll.mask_c_3d_ll`` / ``mask_w_3d_ll`` to be present —
    rebuild ``grid.zarr`` with the updated ``extract_grid.py`` if the older
    surface-only file is what's on disk.
    """
    if grid_ll.mask_c_3d_ll is None or grid_ll.mask_w_3d_ll is None:
        raise ValueError(
            "field_masks_3d_ll requires per-level masks in grid_ll; "
            "rebuild grid.zarr with the 3-D extract_grid extension."
        )

    nlat, nlon = grid_ll.nlat, grid_ll.nlon
    Nlevel = int(level_idx.size)
    levels0 = np.asarray(level_idx, dtype=int) - 1
    if levels0.min() < 0 or levels0.max() >= grid_ll.mask_c_3d_ll.shape[0]:
        raise ValueError(
            f"level_idx {level_idx!r} out of range "
            f"1..{grid_ll.mask_c_3d_ll.shape[0]}"
        )

    n_3d_chans = len(FIELDS_3D) * Nlevel
    n_2d_chans = len(FIELDS_2D)
    out = np.empty((n_3d_chans + n_2d_chans, nlat, nlon), dtype=bool)

    mask_c = grid_ll.mask_c_3d_ll[levels0]  # (Nlevel, nlat, nlon)
    mask_w = grid_ll.mask_w_3d_ll[levels0]
    var_to_mask = {"theta": mask_c, "salt": mask_c, "u": mask_w, "v": mask_w}
    for var_idx, var in enumerate(FIELDS_3D):
        out[var_idx * Nlevel:(var_idx + 1) * Nlevel] = var_to_mask[var]

    out[n_3d_chans] = grid_ll.mask_eta_ll  # eta
    return out
