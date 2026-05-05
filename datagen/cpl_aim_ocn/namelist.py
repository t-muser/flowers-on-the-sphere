"""Namelist generation for the coupled AIM+ocean wrapper.

Each MITgcm component reads several Fortran namelist files at runtime
(``data``, ``data.pkg``, ``data.aimphys``, etc.). The upstream
``verification/cpl_aim+ocn/input_*/`` directories contain working
defaults — ~570 lines of forcing-file paths, physics constants, and
solver parameters that we deliberately do **not** reproduce in Python.

Instead, those upstream files are vendored verbatim into
``templates/{atm,ocn,cpl}/`` (see ``PROVENANCE.md``) and this module
applies *targeted line-level substitutions* for the handful of keys we
actually vary across phases or ensemble members:

| File                  | Keys we patch                                               |
|-----------------------|-------------------------------------------------------------|
| ``atm/data``          | ``nIter0``, ``nTimeSteps``, ``deltaT``, ``pChkptFreq``,     |
|                       | ``dumpFreq``, ``monitorFreq``, ``pickupSuff`` (phase 2),    |
|                       | ``hydrogThetaFile`` (per-seed perturbation IC)              |
| ``atm/data.aimphys``  | ``aim_select_pCO2``, ``aim_fixed_pCO2`` (CO2 sweep; JSON    |
|                       | stores ppm, AIM expects mole fraction); ``SOLC`` in         |
|                       | ``&AIM_PAR_FOR`` (area-mean solar sweep)                    |
| ``atm/data.pkg``      | ``useDiagnostics`` (off in spin-up, on in data phase)       |
| ``atm/data.diagnostics`` | **New file** — generated from scratch (upstream has none)|
| ``ocn/data``          | same time-related keys as ``atm/data`` plus ``pickupSuff``  |
| ``ocn/data.pkg``      | ``useDiagnostics`` (already on upstream — left alone)       |
| ``ocn/data.gmredi``   | ``GM_background_K``, ``GM_isopycK`` (κ sweep)               |
| ``ocn/data.diagnostics`` | ``frequency`` of the surfDiag/dynDiag streams            |
| ``cpl/data.cpl``      | ``cpl_atmSendFrq`` (kept at upstream default by default)    |
| ``eedata`` (both)     | unchanged                                                   |

Public entry points:

* :func:`render_phase_namelists` — produce the full ``namelists`` dict
  ready to pass to :func:`datagen.cpl_aim_ocn.staging.stage_run`.
* The individual ``render_*`` functions are exposed for unit testing.

The substitution engine (``_set``) is intentionally simple and only
handles single-line scalar keys (the upstream multi-line array values
like ``tRef`` and ``delR`` are not in our patch set). It refuses to
silently no-op: if a target namelist block is missing, or if patching
a key that exists multiple times produces ambiguity, the function raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# ─── Where the templates live ────────────────────────────────────────────────

_PKG = Path(__file__).resolve().parent
_TEMPLATES = _PKG / "templates"


# ─── Fortran value formatting ────────────────────────────────────────────────

def _fmt(value: bool | int | float | str) -> str:
    """Render ``value`` as a Fortran namelist literal (no trailing comma)."""
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Always emit a decimal point so the Fortran lexer treats the
        # token as REAL not INT (e.g. 0 → "0." not "0"). 7-significant-
        # figure precision is more than enough for our parameter ranges.
        s = f"{value:.7g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += "."
        return s
    if isinstance(value, str):
        # Single-quoted strings; we rely on Fortran's permissive lexer
        # (the keys we set are all simple identifiers / paths without
        # embedded apostrophes).
        if "'" in value:
            raise ValueError(f"Cannot embed apostrophe in namelist string: {value!r}")
        return f"'{value}'"
    raise TypeError(f"Unsupported namelist value type: {type(value).__name__}")


# ─── Block-aware key substitution ────────────────────────────────────────────

# Fortran namelist syntax (as used by MITgcm):
#
#     <leading whitespace> & <BLOCK_NAME> [whitespace]
#         key1 = value1,
#         key2 = value2,
#     <leading whitespace> & [or "/"] [whitespace]
#
# Keys are case-insensitive; blocks are conventionally upper-case. Lines
# starting with '#', '!', or a Fortran fixed-form 'C'/'c' in column 1 are
# comments and ignored. We deliberately do not parse continuation lines —
# every key we need to patch is a single-line scalar.

_BLOCK_OPEN_RE  = re.compile(r"^\s*&([A-Za-z][A-Za-z0-9_]*)\s*$")
_BLOCK_CLOSE_RE = re.compile(r"^\s*[&/]\s*$")
_COMMENT_PFX    = ("#", "!")


def _is_comment(line: str) -> bool:
    """True for lines we should ignore when looking for active keys."""
    stripped = line.lstrip()
    if not stripped or stripped.startswith(_COMMENT_PFX):
        return True
    # Fortran fixed-form comment: "C" or "c" in column 1 (no leading ws).
    if line.startswith(("C", "c")) and (len(line) > 1 and not line[1].isalnum()):
        return True
    return False


def _set(text: str, block: str, key: str, value: bool | int | float | str) -> str:
    """Set ``key = value`` inside the ``&block`` namelist of ``text``.

    If ``key`` is already present (uncommented) in the block, replace
    its value, preserving the original indentation. If ``key`` does
    not appear anywhere (active) in the block, insert a new line
    ``  key = value,`` immediately before the block-closing line.

    If the key appears more than once uncommented (a Fortran legality —
    the last write wins), replace **only the last occurrence**, mirroring
    the runtime semantics. That keeps re-runs idempotent.

    Raises:
        ValueError: if no ``&block`` exists in ``text``.
    """
    formatted = _fmt(value)
    lines = text.splitlines(keepends=True)

    # Locate every &block (there may be several with the same name in
    # weird files — we patch the first occurrence which is what every
    # MITgcm namelist does in practice).
    block_starts = [
        i for i, ln in enumerate(lines)
        if (m := _BLOCK_OPEN_RE.match(ln)) and m.group(1).upper() == block.upper()
    ]
    if not block_starts:
        raise ValueError(f"Namelist block &{block} not found in template")

    start = block_starts[0]
    end = next(
        (j for j in range(start + 1, len(lines)) if _BLOCK_CLOSE_RE.match(lines[j])),
        None,
    )
    if end is None:
        raise ValueError(f"Namelist block &{block} not closed (missing terminator)")

    # Find the last active line that sets `key`.
    key_re = re.compile(rf"^(\s*){re.escape(key)}\s*=\s*([^,]*?)(,?\s*)$",
                        re.IGNORECASE)
    last_match: int | None = None
    for j in range(start + 1, end):
        if _is_comment(lines[j]):
            continue
        if key_re.match(lines[j]):
            last_match = j

    if last_match is not None:
        m = key_re.match(lines[last_match])
        assert m is not None
        indent, _old_value, trailer = m.groups()
        # Preserve the trailing comma (or its absence).
        comma = "," if "," in trailer else ""
        lines[last_match] = f"{indent}{key}={formatted}{comma}\n"
    else:
        # Insert before the closing terminator with a sensible indent.
        # Match the indent of an existing keyed line if present, else " ".
        indent = " "
        for j in range(start + 1, end):
            if not _is_comment(lines[j]) and "=" in lines[j]:
                indent = re.match(r"^(\s*)", lines[j]).group(1) or " "
                break
        lines.insert(end, f"{indent}{key}={formatted},\n")

    return "".join(lines)


def _read_template(component: str, fname: str) -> str:
    """Load a vendored namelist template from ``templates/<component>/``."""
    path = _TEMPLATES / component / fname
    if not path.is_file():
        raise FileNotFoundError(f"Missing template: {path}")
    return path.read_text()


# ─── Per-file renderers ──────────────────────────────────────────────────────
#
# Naming convention: ``render_<component>_<filename>(...)``. Each takes
# only the parameters it actually needs (so test cases stay sharp) and
# returns the file content as a string.
# ────────────────────────────────────────────────────────────────────────────

# ── Atmosphere ──

def render_atm_data(
    *,
    n_iter0: int,
    n_timesteps: int,
    delta_t: float = 450.0,
    write_pickup_at_end: bool,
    snapshot_interval_s: float | None,
    pickup_suff: str | None = None,
    hydrog_theta_file: str | None = None,
) -> str:
    """Render ``rank_2/data`` for one phase of one ensemble member.

    Args:
        n_iter0:                 ``nIter0``  (start iteration count).
        n_timesteps:             ``nTimeSteps`` (number of steps to integrate).
        delta_t:                 ``deltaT``  [s]; default 450 s = upstream.
        write_pickup_at_end:     If True, set ``pChkptFreq`` = run length so
                                 a pickup is written at the very last step.
                                 If False, set it large (effectively off).
        snapshot_interval_s:     If not None, set ``dumpFreq`` and
                                 ``monitorFreq`` to this value. If None,
                                 disable both (spin-up phase).
        pickup_suff:             If given, set ``pickupSuff='<value>'`` so
                                 the run restarts from that pickup. Phase
                                 2 uses the iteration string written at the
                                 end of phase 1.
        hydrog_theta_file:       If given, set ``hydrogThetaFile='<value>'``
                                 to load a per-seed atmospheric theta
                                 perturbation. Phase 1 only.
    """
    text = _read_template("atm", "data")

    # I/O consolidation. With useSingleCpuIO=.FALSE. (MITgcm default)
    # the model writes per-tile output files
    # (``atm_2d.0000000010.001.001.data`` …); xmitgcm's geometry='cs'
    # reader expects a single global file. Setting it here in PARM01
    # (NOT eedata — that's a common gotcha) consolidates output on
    # rank 0 before write. The runtime cost is negligible at cs32.
    text = _set(text, "PARM01", "useSingleCpuIO", True)

    text = _set(text, "PARM03", "nIter0",       n_iter0)
    text = _set(text, "PARM03", "nTimeSteps",   n_timesteps)
    text = _set(text, "PARM03", "deltaT",       float(delta_t))

    run_seconds = float(n_timesteps) * float(delta_t)
    if write_pickup_at_end:
        # Set pChkptFreq exactly equal to the run length: MITgcm writes
        # a pickup whenever (current_time mod pChkptFreq) == 0, so this
        # fires once at the final step (and at iter0 if iter0=0, which
        # MITgcm short-circuits — the step-end pickup is what we use).
        text = _set(text, "PARM03", "pChkptFreq", run_seconds)
    else:
        # Effectively disable: a frequency well outside any run we'd do.
        text = _set(text, "PARM03", "pChkptFreq", 1.0e20)

    if snapshot_interval_s is not None:
        text = _set(text, "PARM03", "dumpFreq",    float(snapshot_interval_s))
        text = _set(text, "PARM03", "monitorFreq", float(snapshot_interval_s))
    else:
        text = _set(text, "PARM03", "dumpFreq",    0.0)
        # Keep monitorFreq active in spin-up too — it's useful for
        # diagnosing instability without producing field dumps.
        text = _set(text, "PARM03", "monitorFreq", 86400.0)

    if pickup_suff is not None:
        text = _set(text, "PARM03", "pickupSuff", str(pickup_suff))
    if hydrog_theta_file is not None:
        text = _set(text, "PARM05", "hydrogThetaFile", str(hydrog_theta_file))
    return text


def render_atm_aimphys(
    *,
    co2_ppm: float,
    solar_const_w_m2: float,
) -> str:
    """Render ``rank_2/data.aimphys`` with sweep CO2 + solar values.

    Sets ``aim_select_pCO2 = 1`` (use a fixed value) and
    ``aim_fixed_pCO2 = co2_ppm * 1e-6`` in ``&AIM_PARAMS``. The sweep
    JSON uses human-facing ppm, while AIM's radiation code expects a
    dry-air mole fraction. ``SOLC`` is AIM's area-mean incoming solar
    constant (upstream default 342 W/m²), not top-of-atmosphere TSI.
    All other AIM physics defaults are inherited from the template.
    """
    text = _read_template("atm", "data.aimphys")
    text = _set(text, "AIM_PARAMS", "aim_select_pCO2", 1)
    text = _set(text, "AIM_PARAMS", "aim_fixed_pCO2", float(co2_ppm) * 1.0e-6)
    text = _set(text, "AIM_PAR_FOR", "SOLC", float(solar_const_w_m2))
    return text


def render_atm_pkg(*, use_diagnostics: bool) -> str:
    """Render ``rank_2/data.pkg`` toggling the diagnostics package."""
    text = _read_template("atm", "data.pkg")
    text = _set(text, "PACKAGES", "useDiagnostics", use_diagnostics)
    return text


def render_atm_diagnostics(*, snapshot_interval_s: float) -> str:
    """Generate a fresh ``rank_2/data.diagnostics`` for the AIM atmosphere.

    Two output streams are defined (every diagnostic name verified
    against MITgcm's runtime ``available_diagnostics.log`` for cs32 +
    ``pkg/aim_v23 + thsice``):

    * ``atm_2d`` — surface state at the bottom σ-level + sea ice fields::

        TS       — near-surface air temperature        (K)
        QS       — near-surface specific humidity      (g/kg)
        PRECON   — convective precipitation            (g/m²/s)
        PRECLS   — large-scale precipitation           (g/m²/s)
        WINDS    — surface wind speed                  (m/s)
        UFLUX    — zonal wind surface stress           (N/m²)
        VFLUX    — meridional wind surface stress      (N/m²)
        SI_Fract — sea-ice fraction                    (0..1)
        SI_Thick — sea-ice thickness                   (m)

    * ``atm_3d`` — full 3-D atmosphere on all 5 σ-levels::

        UVEL, VVEL, THETA  (potential T)
        SALT  (humidity tracer; AIM repurposes the salt slot for q)

    Frequency is the snapshot interval in seconds; ``timePhase = 0``
    means snapshots fire on whole multiples of ``frequency`` from the
    start of the run. Negative ``frequency`` gives instantaneous
    snapshots (vs positive = time averaged) — we want instantaneous
    for ML training data, hence the leading minus sign.
    """
    snap_freq = _fmt(-float(snapshot_interval_s))  # negative ⇒ instantaneous

    # We deliberately emit ONE stream per field (instead of one stream
    # per nine fields).  Reason: ``xmitgcm.open_mdsdataset`` with
    # ``geometry='cs'`` reads via ``read_CS_chunks``, which assumes one
    # record (= one field) per file. A multi-field MDS stream produces
    # files of size ``nFlds × Nx × Ny × 8 bytes``, which xmitgcm
    # rejects with a size-mismatch error against its expected
    # single-record size. Splitting per-field side-steps this and is
    # cheap on the small cs32 grid.
    streams_2d = (
        "TS", "QS", "PRECON", "PRECLS", "WINDS",
        "UFLUX", "VFLUX", "SI_Fract", "SI_Thick",
    )
    streams_3d = ("UVEL", "VVEL", "THETA", "SALT")

    blocks: list[str] = []
    n = 0
    for fld in streams_2d:
        n += 1
        blocks.append(
            f"  fields(1,{n}) = '{fld:<8s}',\n"
            f"   fileName({n}) = 'atm_{fld}',\n"
            f"   frequency({n}) = {snap_freq},\n"
            f"   timePhase({n}) = 0.,\n"
        )
    for fld in streams_3d:
        n += 1
        blocks.append(
            f"  fields(1,{n}) = '{fld:<8s}',\n"
            f"   fileName({n}) = 'atm_{fld}',\n"
            f"   frequency({n}) = {snap_freq},\n"
            f"   timePhase({n}) = 0.,\n"
        )

    body = "\n".join(blocks)
    return f"""# Diagnostics for the coupled AIM atmosphere — generated by
# datagen/cpl_aim_ocn/namelist.py::render_atm_diagnostics().
#
# Field names verified against MITgcm's runtime
# available_diagnostics.log for cs32 + aim_v23 + thsice. We use
# ONE stream per field (multi-field MDS streams trip up xmitgcm's
# CS reader, which expects one record per file). Negative
# frequency ⇒ instantaneous snapshots.
 &DIAGNOSTICS_LIST
{body} &

 &DIAG_STATIS_PARMS
 &
"""


def render_atm_land() -> str:
    """``rank_2/data.land`` — verbatim from upstream."""
    return _read_template("atm", "data.land")


def render_atm_ice() -> str:
    """``rank_2/data.ice`` — verbatim."""
    return _read_template("atm", "data.ice")


def render_atm_shap() -> str:
    """``rank_2/data.shap`` — verbatim."""
    return _read_template("atm", "data.shap")


def render_atm_cpl(*, cpl_atm_send_freq_s: float = 3600.0) -> str:
    """``rank_2/data.cpl`` — atm-side coupler interface namelist.

    Block: ``&CPL_ATM_PARAM``. The only key we actively change is
    ``cpl_atmSendFrq`` (the period at which the atm sends fields to the
    coupler — equal to the coupler exchange period). All ``useImport*``
    flags are kept commented (default) so the standard set of fields
    is exchanged.
    """
    text = _read_template("atm", "data.cpl")
    text = _set(text, "CPL_ATM_PARAM", "cpl_atmSendFrq",
                float(cpl_atm_send_freq_s))
    return text


def render_atm_eedata() -> str:
    """``rank_2/eedata`` — verbatim (must keep ``useCoupler=.TRUE.``).

    ``useSingleCpuIO`` is set in the ``data`` namelist (PARM01), not
    here — see :func:`render_atm_data`.
    """
    return _read_template("atm", "eedata")


# ── Ocean ──

def render_ocn_data(
    *,
    n_iter0: int,
    n_timesteps: int,
    delta_t: float = 3600.0,
    write_pickup_at_end: bool,
    snapshot_interval_s: float | None,
    pickup_suff: str | None = None,
) -> str:
    """Render ``rank_1/data`` for one phase of one ensemble member.

    Same parameter semantics as :func:`render_atm_data`, but no
    ``hydrogThetaFile`` knob (the ocean IC is supplied by the
    pickup file or the template's Levitus T/S files).
    """
    text = _read_template("ocn", "data")

    # See render_atm_data for rationale (I/O consolidation for cs32).
    text = _set(text, "PARM01", "useSingleCpuIO", True)

    text = _set(text, "PARM03", "nIter0",       n_iter0)
    text = _set(text, "PARM03", "nTimeSteps",   n_timesteps)
    text = _set(text, "PARM03", "deltaTmom",    float(delta_t))
    text = _set(text, "PARM03", "deltaTtracer", float(delta_t))
    text = _set(text, "PARM03", "deltaTClock",  float(delta_t))

    run_seconds = float(n_timesteps) * float(delta_t)
    if write_pickup_at_end:
        text = _set(text, "PARM03", "pChkptFreq", run_seconds)
    else:
        text = _set(text, "PARM03", "pChkptFreq", 1.0e20)

    if snapshot_interval_s is not None:
        text = _set(text, "PARM03", "dumpFreq",    float(snapshot_interval_s))
        text = _set(text, "PARM03", "monitorFreq", float(snapshot_interval_s))
    else:
        text = _set(text, "PARM03", "dumpFreq",    0.0)
        text = _set(text, "PARM03", "monitorFreq", 86400.0)

    if pickup_suff is not None:
        text = _set(text, "PARM03", "pickupSuff", str(pickup_suff))
    return text


def render_ocn_pkg(*, use_diagnostics: bool) -> str:
    """``rank_1/data.pkg`` — flip ``useDiagnostics`` per phase."""
    text = _read_template("ocn", "data.pkg")
    text = _set(text, "PACKAGES", "useDiagnostics", use_diagnostics)
    return text


def render_ocn_gmredi(*, gm_kappa: float) -> str:
    """``rank_1/data.gmredi`` with the swept GM-Redi κ.

    Sets both ``GM_background_K`` (constant background) and
    ``GM_isopycK`` (isopycnal) to ``gm_kappa``. Upstream defaults
    ``GM_background_K = 800``; ``GM_isopycK`` defaults to
    ``GM_background_K`` if unset, so we pin it explicitly to make the
    sweep value exact.
    """
    text = _read_template("ocn", "data.gmredi")
    text = _set(text, "GM_PARM01", "GM_background_K", float(gm_kappa))
    text = _set(text, "GM_PARM01", "GM_isopycK",      float(gm_kappa))
    return text


def render_ocn_diagnostics(*, snapshot_interval_s: float) -> str:
    """Generate a fresh ``rank_1/data.diagnostics`` for the ocean.

    Writes one output stream, ``ocn_surf``, containing instantaneous
    snapshots of the surface ocean state at ``snapshot_interval_s``::

      ETAN     — sea-surface height anomaly (free surface)
      THETA    — sea-surface temperature  (level 1 only via levels(1,1)=1.)
      SALT     — sea-surface salinity
      UVEL     — surface zonal velocity
      VVEL     — surface meridional velocity
      MXLDEPTH — KPP/oceMxL mixed-layer depth diagnostic

    We do **not** write the upstream's full-3D ``dynDiag`` stream — at
    cs32×15z that's ~5 MB per snapshot per run, dominating the dataset
    size. The user spec called for surface ocean only (the 3-D part of
    the dataset is the AIM atmosphere on its 5 σ-levels).
    """
    snap_freq = _fmt(-float(snapshot_interval_s))   # negative ⇒ instantaneous

    # One stream per field — see render_atm_diagnostics() for the
    # rationale (xmitgcm CS reader limitation). For the surface fields
    # we use levels(1,N) = 1. to slice the topmost ocean level out of
    # the 3-D fields THETA/SALT/UVEL/VVEL.
    streams_3d_surface = ("THETA", "SALT", "UVEL", "VVEL")
    streams_2d         = ("ETAN", "MXLDEPTH")

    blocks: list[str] = []
    n = 0
    for fld in streams_3d_surface:
        n += 1
        blocks.append(
            f"  fields(1,{n}) = '{fld:<8s}',\n"
            f"   levels(1,{n}) = 1.,\n"
            f"   fileName({n}) = 'ocn_{fld}',\n"
            f"   frequency({n}) = {snap_freq},\n"
            f"   timePhase({n}) = 0.,\n"
        )
    for fld in streams_2d:
        n += 1
        blocks.append(
            f"  fields(1,{n}) = '{fld:<8s}',\n"
            f"   fileName({n}) = 'ocn_{fld}',\n"
            f"   frequency({n}) = {snap_freq},\n"
            f"   timePhase({n}) = 0.,\n"
        )

    body = "\n".join(blocks)
    return f"""# Ocean diagnostics — generated by
# datagen/cpl_aim_ocn/namelist.py::render_ocn_diagnostics().
#
# One stream per field (xmitgcm-CS reader friendly).  Surface ocean
# only — the 3-D fields are level-1 sliced with levels(1,N) = 1.
# Negative frequency ⇒ instantaneous snapshots.
 &DIAGNOSTICS_LIST
{body} &

 &DIAG_STATIS_PARMS
 &
"""


def render_ocn_cpl() -> str:
    """``rank_1/data.cpl`` (ocn-side coupler imports) — verbatim."""
    return _read_template("ocn", "data.cpl")


def render_ocn_eedata() -> str:
    """``rank_1/eedata`` — verbatim. ``useSingleCpuIO`` is set in
    ``data`` PARM01, not here."""
    return _read_template("ocn", "eedata")


# ── Coupler ──

def render_cpl_data() -> str:
    """``rank_0/data.cpl`` — coupler-side exchange & runoff config.

    Block: ``&COUPLER_PARAMS``. Verbatim from upstream — the atm-side
    ``cpl_atmSendFrq`` (set in ``rank_2/data.cpl`` via
    :func:`render_atm_cpl`) drives the exchange cadence; this file just
    declares which fields the coupler routes between the components,
    plus the runoff routing map. None of these vary per ensemble member.
    """
    return _read_template("cpl", "data.cpl")


# The coupler also needs an `eedata`. Upstream uses the atmos one
# (run_cpl_test symlinks input_atm/eedata into rank_0/), but since we
# stage rank_0 from inputs/cpl/ only — and inputs/cpl/ contains no
# eedata — we re-use the atm-side file via a thin wrapper.
def render_cpl_eedata() -> str:
    """Coupler-side ``eedata`` — same content as the atm-side file."""
    return _read_template("atm", "eedata")


# ─── Phase orchestration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhaseTimeConfig:
    """Time-stepping config for a single phase (spin-up or data).

    Defaults follow the cs32 verification: atm step = 450 s,
    ocn step = 3600 s, daily snapshots in the data phase.
    """
    n_atm_steps: int
    n_ocn_steps: int
    delta_t_atm: float = 450.0
    delta_t_ocn: float = 3600.0
    snapshot_interval_s: float | None = None  # None → no diagnostics
    pickup_suff_ocn: str | None = None
    pickup_suff_atm: str | None = None
    write_pickup_at_end: bool = True
    hydrog_theta_file: str | None = None       # atm theta perturb (phase 1)
    cpl_atm_send_freq_s: float = 3600.0


@dataclass(frozen=True)
class SweepParams:
    """Physical (per-ensemble-member) sweep parameters.

    Stored exactly as the JSON sweep config provides them, so that the
    same dataclass round-trips cleanly through generate_sweep.py.
    """
    co2_ppm: float
    solar_scale: float        # multiplier applied to AIM SOLC=342 W/m²
    gm_kappa: float           # m²/s
    seed: int

    SOLAR_REF: float = field(default=342.0, init=False, repr=False)

    @property
    def solar_const_w_m2(self) -> float:
        return float(self.solar_scale) * float(self.SOLAR_REF)


# ─── Master entry point ─────────────────────────────────────────────────────

def render_phase_namelists(
    *,
    time_cfg: PhaseTimeConfig,
    sweep: SweepParams,
) -> dict[str, dict[str, str]]:
    """Produce the full ``namelists`` dict for one phase of one run.

    The returned dict's shape matches what
    :func:`datagen.cpl_aim_ocn.staging.stage_run` expects::

        {
          "cpl": {"data.cpl": "...", "eedata": "..."},
          "ocn": {"data": "...", "data.pkg": "...", ...},
          "atm": {"data": "...", "data.aimphys": "...", ...},
        }

    The diagnostics files are only emitted when
    ``time_cfg.snapshot_interval_s`` is not None (i.e. data phase, not
    spin-up). The ``useDiagnostics`` package toggle is set accordingly.
    """
    has_diag = time_cfg.snapshot_interval_s is not None
    snap_s = time_cfg.snapshot_interval_s

    atm: dict[str, str] = {
        "data": render_atm_data(
            n_iter0=0 if time_cfg.pickup_suff_atm is None else _iter_from_suff(time_cfg.pickup_suff_atm),
            n_timesteps=time_cfg.n_atm_steps,
            delta_t=time_cfg.delta_t_atm,
            write_pickup_at_end=time_cfg.write_pickup_at_end,
            snapshot_interval_s=snap_s,
            pickup_suff=time_cfg.pickup_suff_atm,
            hydrog_theta_file=time_cfg.hydrog_theta_file,
        ),
        "data.aimphys": render_atm_aimphys(
            co2_ppm=sweep.co2_ppm,
            solar_const_w_m2=sweep.solar_const_w_m2,
        ),
        "data.pkg": render_atm_pkg(use_diagnostics=has_diag),
        "data.land": render_atm_land(),
        "data.ice": render_atm_ice(),
        "data.shap": render_atm_shap(),
        "data.cpl": render_atm_cpl(
            cpl_atm_send_freq_s=time_cfg.cpl_atm_send_freq_s,
        ),
        "eedata": render_atm_eedata(),
    }
    if has_diag:
        atm["data.diagnostics"] = render_atm_diagnostics(
            snapshot_interval_s=snap_s
        )

    ocn: dict[str, str] = {
        "data": render_ocn_data(
            n_iter0=0 if time_cfg.pickup_suff_ocn is None else _iter_from_suff(time_cfg.pickup_suff_ocn),
            n_timesteps=time_cfg.n_ocn_steps,
            delta_t=time_cfg.delta_t_ocn,
            write_pickup_at_end=time_cfg.write_pickup_at_end,
            snapshot_interval_s=snap_s,
            pickup_suff=time_cfg.pickup_suff_ocn,
        ),
        "data.pkg": render_ocn_pkg(use_diagnostics=has_diag),
        "data.gmredi": render_ocn_gmredi(gm_kappa=sweep.gm_kappa),
        "data.cpl": render_ocn_cpl(),
        "eedata": render_ocn_eedata(),
    }
    if has_diag:
        ocn["data.diagnostics"] = render_ocn_diagnostics(
            snapshot_interval_s=snap_s
        )

    cpl = {
        "data.cpl": render_cpl_data(),
        "eedata":   render_cpl_eedata(),
    }

    return {"cpl": cpl, "ocn": ocn, "atm": atm}


def _iter_from_suff(suff: str) -> int:
    """Convert a pickup suffix like ``"0000072000"`` into the integer
    iteration to use as ``nIter0`` on restart.

    MITgcm's ``pickupSuff`` mechanism is independent of ``nIter0``, but
    we set ``nIter0`` to match for cleanliness so the model's reported
    iteration count matches the pickup-suffix metadata.
    """
    try:
        return int(suff)
    except ValueError:
        # Non-numeric suffix (e.g. "_phase1") — leave nIter0 at 0.
        return 0
