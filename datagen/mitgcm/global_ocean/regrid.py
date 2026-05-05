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

    return GlobalOceanLatLon(
        nlat=nlat, nlon=nlon, weights=weights,
        angle_cs=grid["angle_cs"].values.astype(np.float32),
        angle_sn=grid["angle_sn"].values.astype(np.float32),
        depth_ll=depth_ll.astype(np.float32),
        mask_k1_ll=mask_k1_ll, mask_k2_ll=mask_k2_ll, mask_eta_ll=mask_eta_ll,
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
