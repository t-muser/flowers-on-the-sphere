"""Per-run staging: build the ``rank_0/`` (cpl), ``rank_1/`` (ocn),
``rank_2/`` (atm) directory layout that the MPMD ``mpirun`` invocation
expects.

Background
----------
The MITgcm in-house coupler launches three binaries as a single MPMD job::

    mpirun -np 1 build_cpl/mitgcmuv \
         : -np 1 build_ocn/mitgcmuv \
         : -np 1 build_atm/mitgcmuv

inside a *run dir* that contains three sibling sub-directories
``rank_0/``, ``rank_1/``, ``rank_2/``.  After ``MPI_Init`` each component
calls ``setdir(my_id)`` (in MITgcm's ``compon_communic`` package) which
``chdir``-s into ``rank_<my_id>/`` so all subsequent file I/O (namelists,
forcing data, STDOUT, STDERR) is per-component.

The mapping between MPI rank → component is fixed by the order of the
colon-separated ``mpirun`` MPMD form:

    rank 0 → coupler  → ``rank_0/`` (reads ``data.cpl``, ``eedata``)
    rank 1 → ocean    → ``rank_1/`` (reads ``data``, ``data.pkg``, …)
    rank 2 → atmosphere → ``rank_2/`` (reads ``data``, ``data.aimphys``, …)

This module is responsible only for *staging* — populating each rank
dir with the right symlinks and namelist files. The actual launch
(constructing and running the mpirun command) lives in ``solver.py``.

Inputs that get symlinked into each rank dir come from the package's
``inputs/`` tree (populated by ``scripts/build.py``)::

    inputs/atm/   13 ``.cpl_FM.bin``, ``albedo_cs32.bin``, etc.
    inputs/ocn/   12 ``.bin`` + ocean ``pickup.0000072000{,.meta}``
    inputs/grid/   6 ``grid_cs32.face00?.bin`` (shared between atm and ocn)
    inputs/cpl/    ``RA.bin``, ``runOff_cs32_3644.bin``

Symlinks are absolute so the rank dir remains valid even if the run dir
itself is moved (only the package dir must stay put).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

# ─── Package-relative defaults (resolved lazily) ─────────────────────────────

_PKG = Path(__file__).resolve().parent
_DEFAULT_INPUTS = _PKG / "inputs"
_DEFAULT_BUILD_DIRS = {
    "cpl": _PKG / "build_cpl",
    "ocn": _PKG / "build_ocn",
    "atm": _PKG / "build_atm",
}

# ─── Rank layout ─────────────────────────────────────────────────────────────

#: Names of the three components, in the order they MUST appear in the
#: MPMD ``mpirun`` command (= MPI rank index they will receive).
COMPONENTS: tuple[str, ...] = ("cpl", "ocn", "atm")


@dataclass(frozen=True)
class RankLayout:
    """Static description of one rank's input requirements + success criteria.

    ``log_filename`` is the file each component writes its main runtime
    log to inside its rank dir. ``log_marker`` is a substring that, when
    present in that file, indicates the binary finished cleanly. The
    coupler is the special case: ``pkg/atm_ocn_coupler/coupler.F`` ends
    with a bare Fortran ``STOP``, which prints nothing — so the coupler
    layout uses ``log_marker = ""`` (empty), interpreted as "log file
    exists and is non-empty".
    """
    component: str            # "cpl", "ocn", or "atm"
    rank_idx: int             # MPI rank index (matches setdir output)
    input_subdirs: tuple[str, ...]  # subdirs of inputs/ to symlink wholesale
    log_filename: str         # name of the main runtime log file in the rank dir
    log_marker: str           # success substring (empty ⇒ existence-only check)

    @property
    def rank_dirname(self) -> str:
        return f"rank_{self.rank_idx}"


#: Per-component layout. Order matches ``COMPONENTS`` and the MPMD launch.
#:
#: atm and ocn are the standard MITgcm dynamical core: STDOUT.0000 ends
#: with ``"PROGRAM MAIN: Execution ended Normally"``.
#:
#: cpl is the in-tree coupler binary; it does not write STDOUT.0000 and
#: its log file ``Coupler.0000.clog`` has no end-of-run marker, so we
#: only check that the file exists and is non-empty.
LAYOUTS: tuple[RankLayout, ...] = (
    RankLayout(
        component="cpl", rank_idx=0, input_subdirs=("cpl",),
        log_filename="Coupler.0000.clog", log_marker="",
    ),
    RankLayout(
        component="ocn", rank_idx=1, input_subdirs=("ocn", "grid"),
        log_filename="STDOUT.0000", log_marker="ended Normally",
    ),
    RankLayout(
        component="atm", rank_idx=2, input_subdirs=("atm", "grid"),
        log_filename="STDOUT.0000", log_marker="ended Normally",
    ),
)
_LAYOUT_BY_COMPONENT: dict[str, RankLayout] = {l.component: l for l in LAYOUTS}


def layout_for(component: str) -> RankLayout:
    """Look up the ``RankLayout`` for a component name (cpl|ocn|atm)."""
    try:
        return _LAYOUT_BY_COMPONENT[component]
    except KeyError as exc:
        raise ValueError(
            f"Unknown component {component!r}; expected one of {COMPONENTS}"
        ) from exc


# ─── Symlink helper ──────────────────────────────────────────────────────────

def _symlink_force(src: Path, dst: Path) -> None:
    """Create or replace a symlink at ``dst`` pointing at ``src.resolve()``.

    Pre-existing symlinks are removed so the staging step is idempotent
    across re-runs (e.g. spin-up phase → data phase keeps the same
    rank dir but might want to re-point a file).
    """
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _symlink_dir_contents(src_dir: Path, dst_dir: Path) -> list[str]:
    """Symlink every regular file in ``src_dir`` into ``dst_dir``.

    Returns the sorted list of basenames staged, so callers can record
    or verify what landed in each rank dir.
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"Input source dir does not exist: {src_dir} — "
            f"run `scripts/build.py` to populate `inputs/`."
        )
    names: list[str] = []
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        _symlink_force(src, dst_dir / src.name)
        names.append(src.name)
    return names


# ─── Public API ──────────────────────────────────────────────────────────────

def stage_run(
    run_dir: Path,
    *,
    namelists: Mapping[str, Mapping[str, str]],
    inputs_root: Path | None = None,
) -> dict[str, Path]:
    """Materialise the full ``rank_0/``, ``rank_1/``, ``rank_2/`` layout.

    Parameters
    ----------
    run_dir
        The output directory. Will be created if it does not exist;
        existing rank dirs inside are preserved and updated in-place
        (existing symlinks are replaced; existing namelist files are
        overwritten with the new content).
    namelists
        ``{component_name: {filename: file_content}}``. Outer keys must
        be a subset of ``COMPONENTS`` (``"cpl"``, ``"ocn"``, ``"atm"``).
        Inner keys are the namelist filenames as MITgcm expects them
        in cwd (``"data"``, ``"data.cpl"``, ``"eedata"``, …). Values
        are the *full text* of the namelist file.
    inputs_root
        Path to the package's staged input bundle. Defaults to
        ``<package>/inputs/``. Override for tests with synthetic data.

    Returns
    -------
    Mapping ``{component_name: rank_dir_path}`` for the three created
    rank dirs.

    Notes
    -----
    Only namelists supplied for a given component are written. This lets
    callers stage a phase-1 spin-up (with no diagnostics) and then a
    phase-2 data run (with diagnostics) by re-calling ``stage_run`` with
    a different ``namelists`` mapping but the same ``run_dir`` — files
    are overwritten rather than appended.
    """
    inputs_root = Path(inputs_root) if inputs_root is not None else _DEFAULT_INPUTS
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    bad_keys = set(namelists) - set(COMPONENTS)
    if bad_keys:
        raise ValueError(
            f"Unknown component(s) in `namelists`: {sorted(bad_keys)}; "
            f"expected subset of {COMPONENTS}"
        )

    rank_dirs: dict[str, Path] = {}
    for layout in LAYOUTS:
        rank_dir = run_dir / layout.rank_dirname
        rank_dir.mkdir(parents=True, exist_ok=True)

        # 1. Symlink every file from each required inputs/<sub>/ into the rank dir.
        for sub in layout.input_subdirs:
            _symlink_dir_contents(inputs_root / sub, rank_dir)

        # 2. Write any caller-provided namelists for this component.
        for fname, content in namelists.get(layout.component, {}).items():
            (rank_dir / fname).write_text(content)

        rank_dirs[layout.component] = rank_dir

    return rank_dirs


def mpmd_command(
    run_dir: Path,
    *,
    n_mpi: Mapping[str, int] | None = None,
    build_dirs: Mapping[str, Path] | None = None,
    mpirun: str = "mpirun",
) -> list[str]:
    """Build the colon-separated MPMD ``mpirun`` argv list.

    The resulting command should be invoked with ``cwd=run_dir`` so the
    ``mitgcmuv`` binaries find ``rank_0/``, ``rank_1/``, ``rank_2/`` as
    siblings (their `setdir()` chdir's into them by MPI rank).

    Parameters
    ----------
    run_dir
        Run directory containing the rank dirs. Used only to assert that
        the rank dirs exist before launch — the actual cwd is the
        caller's responsibility (``subprocess.run(cwd=run_dir)``).
    n_mpi
        ``{component: nranks}``. Defaults to one rank per component.
        The coupler is hard-locked to 1 rank: increasing it requires
        rebuilding ``code_cpl/`` with ``exch2``, which the upstream
        package intentionally omits — so we assert ``n_mpi['cpl'] == 1``.
        Increasing atm/ocn ranks requires a matching ``SIZE.h`` rebuild
        (``nPx * nPy = nranks``); not enforced here, but flagged.
    build_dirs
        ``{component: path_to_build_dir}``. Defaults to the package's
        own ``build_cpl/``, ``build_ocn/``, ``build_atm/``. Override for
        tests or to point at a sibling checkout.
    mpirun
        Name (or full path) of the mpirun launcher; default ``"mpirun"``.

    Returns
    -------
    A list of strings ready to pass to ``subprocess.run`` / ``Popen``.

    Raises
    ------
    FileNotFoundError
        If any of the three rank dirs or executables is missing.
    ValueError
        If ``n_mpi['cpl']`` is anything other than 1.
    """
    n_mpi = dict(n_mpi) if n_mpi is not None else {c: 1 for c in COMPONENTS}
    for c in COMPONENTS:
        n_mpi.setdefault(c, 1)
    if n_mpi["cpl"] != 1:
        raise ValueError(
            f"n_mpi['cpl'] must be 1 (the coupler binary is built without "
            f"exch2 and is intentionally serial); got {n_mpi['cpl']}"
        )

    build_dirs = (
        {c: Path(p) for c, p in build_dirs.items()}
        if build_dirs is not None else dict(_DEFAULT_BUILD_DIRS)
    )
    for c in COMPONENTS:
        build_dirs.setdefault(c, _DEFAULT_BUILD_DIRS[c])

    run_dir = Path(run_dir)
    for layout in LAYOUTS:
        rank_dir = run_dir / layout.rank_dirname
        if not rank_dir.is_dir():
            raise FileNotFoundError(
                f"Rank dir missing: {rank_dir}; call `stage_run` first."
            )
        exe = build_dirs[layout.component] / "mitgcmuv"
        if not exe.is_file():
            raise FileNotFoundError(
                f"Executable missing: {exe}; build it via "
                f"`scripts/build.py --components {layout.component}`."
            )

    # Assemble the colon-separated MPMD argv. OpenMPI and MPICH-Hydra
    # both accept this form. `setdir()` inside MITgcm chdir's by rank,
    # so we don't need `-wdir`.
    cmd: list[str] = [mpirun]
    for i, layout in enumerate(LAYOUTS):
        if i > 0:
            cmd.append(":")
        cmd += [
            "-np", str(n_mpi[layout.component]),
            str((build_dirs[layout.component] / "mitgcmuv").resolve()),
        ]
    return cmd


# ─── Convenience: per-rank "did it finish cleanly?" check ───────────────────

#: The textual end-of-run marker MITgcm's main dycore prints at the bottom
#: of ``STDOUT.0000``. Used as the success criterion for the atm and ocn
#: ranks (the coupler has no analogue — see ``LAYOUTS``).
SUCCESS_MARKER: str = "ended Normally"

def check_run_completed(run_dir: Path) -> dict[str, bool]:
    """Inspect each rank's runtime log for its success criterion.

    For atm and ocn the criterion is the ``"ended Normally"`` substring
    in ``STDOUT.0000``. For cpl, the upstream coupler binary terminates
    via a bare Fortran ``STOP`` and prints no end-of-run marker — its
    only positive signal is the existence of its log file with non-empty
    content. (Combine with the ``mpirun`` exit code for a full pass/fail
    verdict: if mpirun returned 0 and atm+ocn finished, the coupler is
    guaranteed to have reached ``MPI_Finalize`` cleanly, otherwise
    ``mpirun`` would have aborted the job.)

    Returns ``{component: bool}``. Missing log files are reported as
    ``False``.
    """
    out: dict[str, bool] = {}
    for layout in LAYOUTS:
        log = run_dir / layout.rank_dirname / layout.log_filename
        if not log.is_file():
            out[layout.component] = False
            continue
        if layout.log_marker:
            out[layout.component] = layout.log_marker in log.read_text(
                errors="replace"
            )
        else:
            # No textual marker (coupler) → existence + non-emptiness.
            out[layout.component] = log.stat().st_size > 0
    return out
