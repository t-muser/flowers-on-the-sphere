"""Canonical channel layout for the cpl_aim+ocn dataset.

Both the finalize/stats step and the fots-side dataloader need to agree
on which scalar 2D fields make up the channel axis, in what order. This
module is the single source of truth.

Layout
------
The cs32 Zarr written by :mod:`datagen.cpl_aim_ocn.zarr_writer` exposes
13 atm streams (9 surface, 4 full-3D on 5 σ-levels) and 6 ocn streams
(4 already sliced to surface in the namelist + 2 native 2D). After
expansion the canonical channel axis has 35 entries::

    9 atm 2D + (4 × 5) atm 3D + 6 ocn 2D = 35

3D atm streams are split per σ-level so every output channel is a 2D
field on ``(face, j, i)``. The σ index runs ``1..5`` from top of
atmosphere down (``s1`` = TOA, ``s5`` = surface), matching MITgcm's
pressure-coordinate convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

# ─── Stream inventory (matches `zarr_writer.{ATM,OCN}_STREAMS`) ─────────────

ATM_2D_STREAMS: tuple[str, ...] = (
    "atm_TS", "atm_QS",
    "atm_PRECON", "atm_PRECLS",
    "atm_WINDS",
    "atm_UFLUX", "atm_VFLUX",
    "atm_SI_Fract", "atm_SI_Thick",
)

ATM_3D_STREAMS: tuple[str, ...] = (
    "atm_UVEL", "atm_VVEL", "atm_THETA", "atm_SALT",
)

OCN_2D_STREAMS: tuple[str, ...] = (
    "ocn_THETA", "ocn_SALT", "ocn_UVEL", "ocn_VVEL",
    "ocn_ETAN", "ocn_MXLDEPTH",
)

#: Number of σ-levels in the AIM atmosphere (``code_atm/SIZE.h`` Nr=5).
N_SIGMA: int = 5

#: Vertical-dim name on the atm side after ``zarr_writer._rename_atm_vertical``.
ATM_VERTICAL_DIM: str = "Zsigma"


def channel_names() -> tuple[str, ...]:
    """Canonical 35-tuple of channel names in fixed order."""
    out: list[str] = list(ATM_2D_STREAMS)
    for v in ATM_3D_STREAMS:
        out.extend(f"{v}_s{k + 1}" for k in range(N_SIGMA))
    out.extend(OCN_2D_STREAMS)
    return tuple(out)


def expand_to_channels(ds: "xr.Dataset") -> "xr.DataArray":
    """Stack every cs32 data-var into a single ``(time, channel, face, j, i)`` array.

    For 2D streams the corresponding data-var is taken as-is. For 3D atm
    streams the σ axis is split into one channel per level. The output's
    ``channel`` coord is :func:`channel_names`.

    Raises
    ------
    KeyError
        If any expected stream is missing from ``ds`` — failing here is
        better than silently producing fewer channels than declared.
    """
    import xarray as xr

    pieces: list[xr.DataArray] = []

    for v in ATM_2D_STREAMS:
        if v not in ds.data_vars:
            raise KeyError(f"Expected atm 2D stream {v!r} not in dataset")
        pieces.append(ds[v])

    for v in ATM_3D_STREAMS:
        if v not in ds.data_vars:
            raise KeyError(f"Expected atm 3D stream {v!r} not in dataset")
        arr = ds[v]
        if ATM_VERTICAL_DIM not in arr.dims:
            raise ValueError(
                f"Atm 3D stream {v!r} missing vertical dim {ATM_VERTICAL_DIM!r} "
                f"(got dims {arr.dims})"
            )
        if arr.sizes[ATM_VERTICAL_DIM] != N_SIGMA:
            raise ValueError(
                f"Atm 3D stream {v!r} has {arr.sizes[ATM_VERTICAL_DIM]} σ levels, "
                f"expected {N_SIGMA}"
            )
        for k in range(N_SIGMA):
            level = arr.isel({ATM_VERTICAL_DIM: k}, drop=True)
            level = level.rename(f"{v}_s{k + 1}")
            pieces.append(level)

    for v in OCN_2D_STREAMS:
        if v not in ds.data_vars:
            raise KeyError(f"Expected ocn 2D stream {v!r} not in dataset")
        pieces.append(ds[v])

    names = channel_names()
    if len(pieces) != len(names):
        raise AssertionError(
            f"Channel count mismatch: built {len(pieces)} pieces "
            f"but channel_names() has {len(names)}"
        )

    stacked = xr.concat(pieces, dim="channel")
    stacked = stacked.assign_coords(channel=list(names))
    stacked.name = "fields"
    return stacked
