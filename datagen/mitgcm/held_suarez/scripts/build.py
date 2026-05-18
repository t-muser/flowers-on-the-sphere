"""One-time MITgcm compilation script for the Held-Suarez configuration.

This script:
1. Generates a resolution-specific ``SIZE.h`` into a per-build ``mods/``
   directory by overlaying the canonical ``code/`` tree.
2. Creates ``datagen/mitgcm/held_suarez/build_<Nlon>x<Nlat>x<Nr>/`` (or the
   legacy ``build/`` path for the historical 128×64×20 layout) and runs
   ``genmake2`` with ``-mods=`` pointing at the per-build mods tree so our
   custom SIZE.h, packages.conf, and apply_forcing.F override the defaults.
3. Runs ``make depend`` and ``make -j$(nproc)``.
4. Generates the static input directory (bathymetry, build-info JSON)
   symlinked into every run directory:
   ``datagen/mitgcm/held_suarez/input/`` for 128×64, or
   ``datagen/mitgcm/held_suarez/input_<Nlon>x<Nlat>/`` otherwise.

Usage (from the repo root)::

    # Legacy default (128×64×20, build/):
    uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.build \\
        --mitgcm-root /path/to/MITgcm \\
        [--optfile /path/to/build_options/linux_amd64_gfortran]

    # High-resolution build (256×128×30, build_256x128x30/):
    uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.build \\
        --mitgcm-root /path/to/MITgcm \\
        --nlon 256 --nlat 128 --nr 30 --n-mpi 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from datagen.mitgcm.held_suarez._size_h import write_size_h
from datagen.mitgcm.held_suarez.solver import (
    default_build_dir,
    default_input_dir,
)

_HERE = Path(__file__).parent        # datagen/mitgcm/held_suarez/scripts/
_PKG  = _HERE.parent                 # datagen/mitgcm/held_suarez/
_CODE = _PKG / "code"


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a shell command and raise on failure."""
    print(f"$ {' '.join(str(c) for c in cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def _stage_mods_dir(
    build_dir: Path,
    *,
    Nlon: int,
    Nlat: int,
    Nr: int,
    n_mpi: int,
) -> Path:
    """Copy ``code/`` into a per-build ``mods/`` and overwrite ``SIZE.h``.

    Returns the path to the staged mods dir, ready to pass to ``genmake2``.
    """
    mods_dir = build_dir / "mods"
    if mods_dir.exists():
        shutil.rmtree(mods_dir)
    shutil.copytree(_CODE, mods_dir)
    write_size_h(
        mods_dir / "SIZE.h",
        Nlon=Nlon, Nlat=Nlat, Nr=Nr, n_mpi=n_mpi,
    )
    return mods_dir


def compile_mitgcm(
    mitgcm_root: Path,
    optfile: Path | None,
    *,
    Nlon: int,
    Nlat: int,
    Nr: int,
    n_mpi: int,
) -> Path:
    """Run genmake2 + make to produce the mitgcmuv executable.

    Returns the path to the resulting build directory.
    """
    build_dir = default_build_dir(Nlon, Nlat, Nr)
    build_dir.mkdir(parents=True, exist_ok=True)

    mods_dir = _stage_mods_dir(
        build_dir, Nlon=Nlon, Nlat=Nlat, Nr=Nr, n_mpi=n_mpi,
    )

    genmake2 = mitgcm_root / "tools" / "genmake2"
    if not genmake2.exists():
        sys.exit(f"genmake2 not found at {genmake2}")

    genmake_cmd = [
        str(genmake2.resolve()),
        f"-mods={mods_dir.resolve()}",
        f"-rootdir={mitgcm_root.resolve()}",
        "-mpi",
    ]
    if optfile is not None:
        genmake_cmd.append(f"-optfile={optfile.resolve()}")

    _run(genmake_cmd, cwd=build_dir)
    _run(["make", "depend"],                    cwd=build_dir)
    _run(["make", f"-j{os.cpu_count() or 4}"], cwd=build_dir)

    exe = build_dir / "mitgcmuv"
    if not exe.exists():
        sys.exit("Build succeeded but mitgcmuv not found — check make output.")
    print(f"Executable: {exe}")
    return build_dir


def generate_static_inputs(*, Nlon: int, Nlat: int) -> Path:
    """Write static input files shared across all runs at this (Nlon, Nlat).

    Returns the input directory path so the build-info JSON can record it.
    """
    input_dir = default_input_dir(Nlon, Nlat)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Bathymetry: all zeros (flat aqua-planet).
    from datagen.mitgcm.held_suarez.ic import write_bathymetry
    write_bathymetry(input_dir / "bathyFile.bin", Nlon=Nlon, Nlat=Nlat)
    print(f"Wrote {input_dir / 'bathyFile.bin'}")
    return input_dir


def write_build_info(
    build_dir: Path,
    mitgcm_root: Path,
    *,
    Nlon: int,
    Nlat: int,
    Nr: int,
    n_mpi: int,
) -> None:
    """Save a JSON record of the build configuration."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=mitgcm_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    info = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "mitgcm_root": str(mitgcm_root),
        "mitgcm_git_hash": git_hash,
        "executable": str(build_dir / "mitgcmuv"),
        "Nlon": Nlon,
        "Nlat": Nlat,
        "Nr": Nr,
        "n_mpi": n_mpi,
    }
    out = build_dir / "build_info.json"
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Build info: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mitgcm-root", type=Path,
        default=Path(os.environ.get("MITGCM_ROOT", "")),
        help="Path to the MITgcm source tree. Falls back to $MITGCM_ROOT.",
    )
    ap.add_argument(
        "--optfile", type=Path, default=None,
        help="genmake2 optfile (e.g. $MITGCM_ROOT/tools/build_options/linux_amd64_gfortran).",
    )
    ap.add_argument(
        "--skip-compile", action="store_true",
        help="Skip compilation; only regenerate input files.",
    )
    ap.add_argument(
        "--nlon", type=int, default=128, dest="Nlon",
        help="Number of longitude points (default: 128).",
    )
    ap.add_argument(
        "--nlat", type=int, default=64, dest="Nlat",
        help="Number of latitude points (default: 64).",
    )
    ap.add_argument(
        "--nr", type=int, default=20, dest="Nr",
        help="Number of vertical levels (default: 20).",
    )
    ap.add_argument(
        "--n-mpi", type=int, default=4, dest="n_mpi",
        help="MPI rank count used to set nPy (default: 4). Must divide Nlat.",
    )
    args = ap.parse_args()

    if not args.skip_compile:
        if not args.mitgcm_root or not args.mitgcm_root.is_dir():
            sys.exit(
                "Provide --mitgcm-root or set $MITGCM_ROOT to the MITgcm source tree."
            )
        build_dir = compile_mitgcm(
            args.mitgcm_root, args.optfile,
            Nlon=args.Nlon, Nlat=args.Nlat, Nr=args.Nr, n_mpi=args.n_mpi,
        )
        write_build_info(
            build_dir, args.mitgcm_root,
            Nlon=args.Nlon, Nlat=args.Nlat, Nr=args.Nr, n_mpi=args.n_mpi,
        )

    generate_static_inputs(Nlon=args.Nlon, Nlat=args.Nlat)
    print("Build and input generation complete.")


if __name__ == "__main__":
    main()
