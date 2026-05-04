"""Native cs32 Zarr writer for finished cpl_aim+ocn runs.

Reads the per-rank MDS diagnostic output left by a completed two-phase
run (in ``rank_2/`` for the AIM atmosphere, ``rank_1/`` for the ocean),
merges the two component streams into a single ``xarray.Dataset``, and
writes a Zarr store at the requested path.

Output layout (canonical, **no regridding**)
--------------------------------------------
The cs32 cubed-sphere grid is preserved natively, with all six panels
exposed as a leading ``face`` dimension::

    dims      : (time, sigma, face, j, i)   for atm 3-D fields
                (time, face, j, i)          for atm 2-D + ocn surface
    sigma     : 5 atm σ-levels (1..5, top→bottom in MITgcm pressure-coord)
    face, j, i: 6 × 32 × 32  (cs32 cubed sphere)
    coords    : XC, YC (face, j, i)         — cell-centre lon/lat (deg)
                XG, YG (face, j+1, i+1)     — corner   lon/lat
                Z      (sigma)              — atm pressure levels (Pa)

Variable naming
---------------
The atmosphere and ocean both produce ``UVEL``, ``VVEL``, ``THETA``,
``SALT`` diagnostic streams (their dynamical-core names overlap), and
they share the cs32 horizontal grid coordinates ``XC``, ``YC`` etc.
After loading, we **prefix every data variable** with ``atm_`` or
``ocn_`` so the merged Dataset has unambiguous names. Vertical
coordinates are also renamed (``Z`` → ``Zsigma`` on the atm side) to
avoid collision with the ocean's ``Z`` axis.

A separate, optional :mod:`datagen.cpl_aim_ocn.regrid` utility (phase
6) can resample the cs32 output onto a regular lat/lon grid for
downstream consumers — but the canonical dataset stays interpolation-
free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# These imports are deferred at function-call time so importing this
# module never pulls in ``xmitgcm`` / ``xarray`` (some environments may
# only need the renderers from ``namelist.py``). The TYPE_CHECKING /
# import-on-call pattern keeps the package light.

# ─── Stream prefixes by component (matches namelist.py output) ──────────────

#: Output stream filenames our generators emit. Each stream produces an
#: MDS pair ``<prefix>.<iter10>.{data,meta}`` per snapshot in the
#: relevant rank dir.
#:
#: One stream per field is required because ``xmitgcm.open_mdsdataset``
#: with ``geometry='cs'`` cannot consume multi-field MDS files — it
#: assumes one record per file. The field names match the diagnostic
#: IDs registered in MITgcm's ``available_diagnostics.log``.
ATM_STREAMS: tuple[str, ...] = (
    # surface fields (single level)
    "atm_TS", "atm_QS", "atm_PRECON", "atm_PRECLS", "atm_WINDS",
    "atm_UFLUX", "atm_VFLUX", "atm_SI_Fract", "atm_SI_Thick",
    # full-3D atmosphere on 5 σ-levels
    "atm_UVEL", "atm_VVEL", "atm_THETA", "atm_SALT",
)
OCN_STREAMS: tuple[str, ...] = (
    # ocean surface (3-D fields sliced to level 1 in the namelist)
    "ocn_THETA", "ocn_SALT", "ocn_UVEL", "ocn_VVEL",
    # 2-D fields
    "ocn_ETAN", "ocn_MXLDEPTH",
)


# ─── Public API ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZarrWriteResult:
    """Summary returned by :func:`write_cs32_zarr` for caller logging."""
    path: Path
    n_time: int
    atm_vars: tuple[str, ...]
    ocn_vars: tuple[str, ...]


def write_cs32_zarr(
    run_dir: Path,
    out_path: Path,
    *,
    atm_streams: Iterable[str] = ATM_STREAMS,
    ocn_streams: Iterable[str] = OCN_STREAMS,
    grid_dir: Path | None = None,
    delta_t: float | None = None,
    attrs: dict[str, Any] | None = None,
    chunk_time: int = 30,
    overwrite: bool = True,
) -> ZarrWriteResult:
    """Read MDS diagnostics from one finished run and write the canonical Zarr.

    Parameters
    ----------
    run_dir
        The two-phase run dir, containing ``rank_1/`` (ocn) and
        ``rank_2/`` (atm) populated with MDS diagnostic files.
    out_path
        Destination Zarr path (a directory store will be created here).
    atm_streams, ocn_streams
        Which MDS prefixes to load. Defaults match the streams emitted
        by :mod:`datagen.cpl_aim_ocn.namelist` generators.
    grid_dir
        Path to a directory containing the ``grid_cs32.face00?.bin``
        files (xmitgcm needs them with ``geometry='cs'`` to populate
        ``XC``, ``YC`` etc.). Defaults to the rank dir itself, since
        :func:`stage_run` symlinks the grid files into both rank_1 and
        rank_2.
    delta_t
        Optional explicit timestep; if omitted, xmitgcm reads it from
        the rank's ``data`` namelist. The atm and ocn timesteps differ,
        so each call uses its rank's own value — passing one explicit
        value here only applies if the rank dir doesn't have a usable
        ``data`` file.
    attrs
        Extra global attrs to attach (e.g. sweep parameters, run_id).
    chunk_time
        Number of time steps per Zarr chunk. Default 30 (≈1 month of
        daily snapshots) keeps single-chunk reads cheap while writing.
    overwrite
        If True (default), wipe ``out_path`` before writing. If False,
        appends are not supported — the function refuses to overwrite.

    Returns
    -------
    ZarrWriteResult with the produced path, time length, and lists of
    atm/ocn data variables.

    Raises
    ------
    FileNotFoundError
        If a rank dir or required MDS file is missing.
    """
    import xarray as xr     # imported lazily — see module docstring
    import xmitgcm

    run_dir = Path(run_dir).resolve()
    out_path = Path(out_path).resolve()

    rank_atm = run_dir / "rank_2"
    rank_ocn = run_dir / "rank_1"
    for rd in (rank_atm, rank_ocn):
        if not rd.is_dir():
            raise FileNotFoundError(f"Run dir incomplete: {rd} missing")

    atm_grid = grid_dir if grid_dir is not None else rank_atm
    ocn_grid = grid_dir if grid_dir is not None else rank_ocn

    # ── Load atm side ──
    atm = _open_mds_safe(
        rank_atm, prefixes=tuple(atm_streams),
        grid_dir=atm_grid, delta_t=delta_t, xmitgcm=xmitgcm,
    )
    # Disambiguate atm vertical coord (`Z`, `Zl`, `Zp1`) from ocn's.
    atm = _rename_atm_vertical(atm)
    # Prefix data variables: atm_aim_T2m, atm_UVEL, …
    atm = atm.rename({v: f"atm_{v}" for v in list(atm.data_vars)})

    # ── Load ocn side ──
    ocn = _open_mds_safe(
        rank_ocn, prefixes=tuple(ocn_streams),
        grid_dir=ocn_grid, delta_t=delta_t, xmitgcm=xmitgcm,
    )
    ocn = ocn.rename({v: f"ocn_{v}" for v in list(ocn.data_vars)})

    # ── Merge ──
    # We ask xarray to prefer atm's grid coords if they collide with
    # ocn's (they should be identical for cs32).
    merged = xr.merge([atm, ocn], compat="override", join="inner")

    if attrs:
        merged.attrs.update(attrs)
    merged.attrs.setdefault("grid", "cs32 cubed-sphere (192×32 unfolded)")

    # ── Chunking + write ──
    if "time" in merged.dims:
        chunks = {"time": min(chunk_time, merged.sizes["time"])}
        merged = merged.chunk(chunks)

    if out_path.exists() and overwrite:
        import shutil
        shutil.rmtree(out_path)

    merged.to_zarr(out_path, mode="w", consolidated=True)

    return ZarrWriteResult(
        path=out_path,
        n_time=int(merged.sizes.get("time", 0)),
        atm_vars=tuple(v for v in merged.data_vars if v.startswith("atm_")),
        ocn_vars=tuple(v for v in merged.data_vars if v.startswith("ocn_")),
    )


# ─── Internals ──────────────────────────────────────────────────────────────

#: Horizontal cs32 dimensions for ``xmitgcm.open_mdsdataset``.
#:
#: For ``geometry='cs'`` xmitgcm uses **per-face** dims (32×32 for cs32),
#: not the unfolded global array. Passing them explicitly skips
#: ``_guess_model_horiz_dims`` (which would error because the MITgcm
#: runs we orchestrate write per-tile grid files instead of a global
#: ``XC.meta``).
_NX_CS32_PER_FACE: int = 32
_NY_CS32_PER_FACE: int = 32


def _open_mds_safe(
    rank_dir: Path,
    *,
    prefixes: tuple[str, ...],
    grid_dir: Path,
    delta_t: float | None,
    xmitgcm,
):
    """Wrap ``xmitgcm.open_mdsdataset`` with ergonomic error context.

    Filters out prefixes that produced no output files in this rank dir
    so the caller doesn't have to gate by has_diagnostics — useful for
    incremental smoke runs that only enable one stream.
    """
    available: list[str] = []
    for pfx in prefixes:
        if any(rank_dir.glob(f"{pfx}.*.meta")):
            available.append(pfx)
    if not available:
        raise FileNotFoundError(
            f"No MDS output found in {rank_dir} for any of "
            f"prefixes={list(prefixes)} — was useDiagnostics=.TRUE. set?"
        )

    import numpy as np
    kwargs: dict[str, Any] = {
        "prefix": available,
        "geometry": "cs",
        "read_grid": True,
        "grid_dir": str(grid_dir),
        "ignore_unknown_vars": True,
        # Pass the per-face cs32 dimensions explicitly so xmitgcm
        # doesn't try to infer them from a (non-existent) global
        # XC.meta file. With geometry='cs' xmitgcm expects the per-face
        # size, not the unfolded ``Nx = nFaces × face_nx`` figure.
        "nx": _NX_CS32_PER_FACE,
        "ny": _NY_CS32_PER_FACE,
        # When grid files are per-tile, xmitgcm can't infer a global
        # dtype either — provide the writeBinaryPrec=64 setting from
        # the upstream namelists.
        "default_dtype": np.dtype(">f8"),
    }
    if delta_t is not None:
        kwargs["delta_t"] = float(delta_t)

    try:
        return xmitgcm.open_mdsdataset(str(rank_dir), **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"xmitgcm.open_mdsdataset failed for {rank_dir} "
            f"(prefixes={available}): {exc}"
        ) from exc


def _rename_atm_vertical(ds):
    """Rename the atm side's vertical dim to ``sigma`` so it doesn't
    collide with the ocean's depth axis after merge.

    xmitgcm produces ``Z`` (cell centres), ``Zl`` (lower interfaces),
    and sometimes ``Zp1`` (cell edges). All get the ``sigma`` prefix,
    plus the suffix preserved so atm-side fields remain consistently
    coordinated.
    """
    rename = {}
    for old, new in (("Z", "Zsigma"), ("Zl", "Zlsigma"),
                     ("Zp1", "Zp1sigma"), ("Zu", "Zusigma")):
        if old in ds.dims or old in ds.coords:
            rename[old] = new
    return ds.rename(rename) if rename else ds
