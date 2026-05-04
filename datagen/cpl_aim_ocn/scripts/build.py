"""One-time build of the coupled AIM+ocean MITgcm executables.

This script does two things:

1. **Stages the static input bundle** under ``datagen/cpl_aim_ocn/inputs/``
   by copying every ``.bin`` file the runs need from three sibling
   verification experiments inside an MITgcm checkout:

      inputs/atm/   ← <MITgcm>/verification/aim.5l_cs/input.thSI/*.bin
      inputs/ocn/   ← <MITgcm>/verification/global_ocean.cs32x15/input/*
      inputs/grid/  ← <MITgcm>/verification/tutorial_held_suarez_cs/input/grid_cs32.face00?.bin
      inputs/cpl/   ← <MITgcm>/verification/cpl_aim+ocn/input_cpl/{RA,runOff_*}.bin

   Files are *copied* (not symlinked) so a finished build is portable
   away from the MITgcm checkout.

2. **Builds three MPI executables** by running ``genmake2`` from inside
   each of ``build_atm/``, ``build_ocn/``, ``build_cpl/`` (each ships a
   ``genmake_local`` that supplies the right ``MODS=`` directive), then
   ``make depend && make -j``. End state: three ``mitgcmuv`` binaries,
   one per build dir.

The three binaries are launched together via MPMD ``mpirun`` at run time:

    mpirun -np 1 build_cpl/mitgcmuv \
         : -np 1 build_ocn/mitgcmuv \
         : -np 1 build_atm/mitgcmuv

A ``build_info.json`` summary is written to
``datagen/cpl_aim_ocn/build_info.json`` after a successful build for
reproducibility (timestamps, MITgcm git hash, optfile path, executable
sizes, and SHA1 hashes of a few representative input files).

Usage::

    uv run --project datagen python -m datagen.cpl_aim_ocn.scripts.build \
        --mitgcm-root /path/to/MITgcm \
        [--optfile /path/to/build_options/linux_amd64_gfortran] \
        [--jobs 8]                          # parallelism for `make`
        [--skip-stage] [--skip-compile]     # incremental rebuilds

If ``--mitgcm-root`` is omitted, the ``$MITGCM_ROOT`` environment variable
is used. Building the three binaries serially takes a few minutes on a
modern workstation; the staging step copies ~10 MB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# ─── Paths (relative to this file) ───────────────────────────────────────────

_HERE = Path(__file__).parent              # datagen/cpl_aim_ocn/scripts/
_PKG  = _HERE.parent                       # datagen/cpl_aim_ocn/
_INPUTS = _PKG / "inputs"
_BUILD_DIRS = {
    "atm": _PKG / "build_atm",
    "ocn": _PKG / "build_ocn",
    "cpl": _PKG / "build_cpl",
}
_BUILD_INFO = _PKG / "build_info.json"

# ─── Names of the three components to build (also rank-dir indices) ──────────
COMPONENTS: tuple[str, ...] = ("cpl", "ocn", "atm")


# ─── Subdirs of the upstream MITgcm verification tree we depend on ───────────
# Relative to <MITgcm>/verification/. Each entry's pattern field can be a
# single glob string OR a tuple of globs; the staged set is the union.
_SIBLING_DIRS = {
    # source dir, target inputs/<sub>, glob pattern(s), description
    "atm":  ("aim.5l_cs/input.thSI",          "atm",  ("*.bin",),
             "atm forcing/IC: orography, albedo, vegetation, climatologies, land IC"),
    # Ocean: pull binary forcing/IC files AND the pickup pair from the
    # parent global_ocean.cs32x15 spin-up (`pickup.0000072000{,.meta}`).
    # We deliberately do NOT pull the upstream namelist files (`data`,
    # `data.diagnostics`, etc.) — our `namelist.py` regenerates them.
    "ocn":  ("global_ocean.cs32x15/input",    "ocn",  ("*.bin", "pickup.*"),
             "ocn forcing/IC: bathy, Levitus T/S, Trenberth wind stress, ocean pickup"),
    "grid": ("tutorial_held_suarez_cs/input", "grid", ("grid_cs32.face00?.bin",),
             "shared cs32 horizontal grid (6 cube faces)"),
    "cpl":  ("cpl_aim+ocn/input_cpl",         "cpl",  ("*.bin",),
             "coupler-side: RA.bin (atm cell area), runoff routing map"),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run(cmd: Sequence[str | os.PathLike], cwd: Path, env: dict | None = None) -> None:
    """Echo and run a shell command; abort the script on non-zero exit."""
    print(f"$ {' '.join(str(c) for c in cmd)}  (cwd={cwd})", flush=True)
    rc = subprocess.run(list(cmd), cwd=cwd, env=env).returncode
    if rc != 0:
        sys.exit(f"Command failed with exit code {rc}: {cmd[0]}")


def _sha1_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Return the hex SHA1 of a file, streamed in 1 MB chunks."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _git_head(repo: Path) -> str:
    """Return the current git HEAD hash of ``repo``, or ``"unknown"``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ─── Stage forcing files (copy, not symlink) ─────────────────────────────────

def stage_inputs(mitgcm_root: Path) -> dict[str, list[str]]:
    """Copy every required ``.bin`` file from MITgcm's verification tree.

    Returns:
        A dict mapping target subdir name → list of copied filenames.
        Raises ``SystemExit`` if a required source dir is missing.
    """
    verif = mitgcm_root / "verification"
    if not verif.is_dir():
        sys.exit(f"Not an MITgcm checkout: {verif} does not exist.")

    staged: dict[str, list[str]] = {}
    for key, (src_subdir, dst_sub, patterns, desc) in _SIBLING_DIRS.items():
        src = verif / src_subdir
        dst = _INPUTS / dst_sub
        if not src.is_dir():
            sys.exit(
                f"Required upstream dir missing: {src}\n"
                f"  Reason: {desc}\n"
                f"  Cannot proceed — clone MITgcm with verification/ included."
            )
        dst.mkdir(parents=True, exist_ok=True)

        # Patterns may be a single glob (str) or an iterable of globs;
        # take the union and drop any directories.
        glob_list = (patterns,) if isinstance(patterns, str) else tuple(patterns)
        matched: set[Path] = set()
        for pat in glob_list:
            matched.update(p for p in src.glob(pat) if p.is_file())
        files = sorted(matched, key=lambda p: p.name)
        if not files:
            sys.exit(f"No files matched {src}/{glob_list}")

        copied: list[str] = []
        for f in files:
            target = dst / f.name
            # Always overwrite — staging is cheap and deterministic; this
            # makes re-running the script after an upstream re-vendor pick
            # up the new files.
            shutil.copy2(f, target)
            copied.append(f.name)
        print(f"  staged {len(copied):3d} files into {dst} (from {src})", flush=True)
        staged[dst_sub] = copied

    return staged


# ─── Build one component ─────────────────────────────────────────────────────

def build_component(
    component: str,
    mitgcm_root: Path,
    optfile: Path | None,
    jobs: int,
) -> Path:
    """Run ``genmake2 → make depend → make`` inside ``build_<component>/``.

    Each ``build_<component>/`` ships a ``genmake_local`` that already
    sets ``MODS="../code_<component> ../shared_code"`` (and, for the
    coupler, ``STANDARDDIRS=""``), so we only pass ``-mpi`` and the
    optfile to ``genmake2``.

    Returns:
        Path to the produced ``mitgcmuv`` executable.
    """
    build_dir = _BUILD_DIRS[component]
    if not (build_dir / "genmake_local").is_file():
        sys.exit(f"Missing {build_dir}/genmake_local — vendor it from upstream.")

    genmake2 = mitgcm_root / "tools" / "genmake2"
    if not genmake2.is_file():
        sys.exit(f"genmake2 not found at {genmake2} — bad --mitgcm-root?")

    # genmake2 reads `genmake_local` automatically when run inside the
    # build dir; we only have to pass `-mpi` (forced for coupled runs)
    # and `-rootdir`, plus optionally `-of`.
    cmd: list[str] = [
        str(genmake2.resolve()),
        f"-rootdir={mitgcm_root.resolve()}",
        "-mpi",
    ]
    if optfile is not None:
        cmd.append(f"-of={optfile.resolve()}")

    print(f"\n──── Building component '{component}' in {build_dir} ────", flush=True)
    _run(cmd, cwd=build_dir)
    _run(["make", "depend"], cwd=build_dir)
    _run(["make", f"-j{jobs}"], cwd=build_dir)

    exe = build_dir / "mitgcmuv"
    if not exe.is_file():
        sys.exit(f"Build of '{component}' finished but {exe} is missing.")
    print(f"  → {exe} ({exe.stat().st_size:,} bytes)", flush=True)
    return exe


# ─── Build-info JSON ─────────────────────────────────────────────────────────

def write_build_info(
    mitgcm_root: Path,
    optfile: Path | None,
    executables: dict[str, Path],
    staged: dict[str, list[str]],
) -> None:
    """Write a JSON summary of the build for reproducibility."""
    # Sample one binary file from each staged dir (deterministic: the
    # alphabetically-first .bin), to detect upstream input drift cheaply.
    provenance: dict[str, dict[str, str]] = {}
    for sub, names in staged.items():
        bin_names = [n for n in names if n.endswith(".bin")]
        if not bin_names:
            continue
        sample = sorted(bin_names)[0]
        sample_path = _INPUTS / sub / sample
        provenance[sub] = {
            "sample_file": sample,
            "sample_sha1": _sha1_file(sample_path),
            "n_files": str(len(names)),
        }

    info = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "mitgcm_root": str(mitgcm_root.resolve()),
        "mitgcm_git_hash": _git_head(mitgcm_root),
        "optfile": str(optfile.resolve()) if optfile is not None else None,
        "executables": {
            comp: {
                "path": str(p.resolve()),
                "size_bytes": p.stat().st_size,
            }
            for comp, p in executables.items()
        },
        "staged_inputs": {sub: sorted(names) for sub, names in staged.items()},
        "input_provenance": provenance,
    }
    _BUILD_INFO.write_text(json.dumps(info, indent=2) + "\n")
    print(f"\nWrote {_BUILD_INFO}", flush=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build the three coupled-AIM-ocean MITgcm executables.",
    )
    ap.add_argument(
        "--mitgcm-root", type=Path,
        default=Path(os.environ.get("MITGCM_ROOT", "")),
        help="Path to the MITgcm source tree (overrides $MITGCM_ROOT).",
    )
    ap.add_argument(
        "--optfile", type=Path, default=None,
        help="genmake2 optfile (e.g. <MITgcm>/tools/build_options/linux_amd64_gfortran). "
             "If omitted, genmake2 auto-detects.",
    )
    ap.add_argument(
        "--jobs", "-j", type=int, default=os.cpu_count() or 4,
        help="Parallelism for `make` (default: $(nproc)).",
    )
    ap.add_argument(
        "--skip-stage", action="store_true",
        help="Skip the input-staging step (assume inputs/ already populated).",
    )
    ap.add_argument(
        "--skip-compile", action="store_true",
        help="Skip the compile step (only stage inputs).",
    )
    ap.add_argument(
        "--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS),
        help="Subset of components to build (default: all three).",
    )
    args = ap.parse_args(argv)

    mitgcm_root = args.mitgcm_root.resolve() if args.mitgcm_root else None
    if not args.skip_stage or not args.skip_compile:
        if mitgcm_root is None or not mitgcm_root.is_dir():
            sys.exit(
                "Provide --mitgcm-root or set $MITGCM_ROOT to a clone of MITgcm "
                "(must contain tools/genmake2 and verification/)."
            )

    staged: dict[str, list[str]] = {}
    if not args.skip_stage:
        print("\n──── Staging input bundle ────", flush=True)
        staged = stage_inputs(mitgcm_root)
    else:
        # Re-derive staged from existing files on disk so build_info is
        # still consistent on a stage-skipping rebuild.
        if _INPUTS.is_dir():
            for sub in ("atm", "ocn", "grid", "cpl"):
                p = _INPUTS / sub
                if p.is_dir():
                    staged[sub] = sorted(f.name for f in p.iterdir() if f.is_file())

    executables: dict[str, Path] = {}
    if not args.skip_compile:
        for component in args.components:
            executables[component] = build_component(
                component, mitgcm_root, args.optfile, args.jobs,
            )

    # Always pick up any pre-existing binaries on disk too, so
    # build_info.json reflects the full state — rebuilding just `atm`
    # must not drop the cpl/ocn entries from a previous build.
    for comp in COMPONENTS:
        if comp not in executables:
            exe = _BUILD_DIRS[comp] / "mitgcmuv"
            if exe.is_file():
                executables[comp] = exe

    # Refresh build_info.json whenever we did either staging or
    # compilation. (Both skipped → script is a no-op, no need to write.)
    if not (args.skip_stage and args.skip_compile):
        write_build_info(mitgcm_root, args.optfile, executables, staged)

    print("\nBuild script complete.")
    if executables:
        print(f"Executables: {', '.join(f'{c}={p}' for c, p in executables.items())}")


if __name__ == "__main__":
    main()
