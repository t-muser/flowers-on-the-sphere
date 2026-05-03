"""One-time MITgcm compilation script for the Held-Suarez configuration.

This script:
1. Creates ``datagen/mitgcm/build/`` and runs ``genmake2`` with the
   ``-mods datagen/mitgcm/code/`` flag so our custom SIZE.h, packages.conf
   and apply_forcing.F override the defaults.
2. Runs ``make depend`` and ``make -j$(nproc)``.
3. Generates the static ``datagen/mitgcm/input/`` files that are symlinked
   into every run directory: ``bathyFile.bin`` and a build-info JSON.

The compiled executable lives at ``datagen/mitgcm/build/mitgcmuv``.

Usage (from the repo root)::

    uv run --project datagen python -m datagen.mitgcm.scripts.build \\
        --mitgcm-root /path/to/MITgcm \\
        [--optfile /path/to/build_options/linux_amd64_gfortran]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent        # datagen/mitgcm/scripts/
_PKG  = _HERE.parent                 # datagen/mitgcm/
_CODE = _PKG / "code"
_BUILD = _PKG / "build"
_INPUT = _PKG / "input"


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a shell command and raise on failure."""
    print(f"$ {' '.join(str(c) for c in cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def compile_mitgcm(mitgcm_root: Path, optfile: Path | None) -> None:
    """Run genmake2 + make to produce the mitgcmuv executable."""
    _BUILD.mkdir(parents=True, exist_ok=True)
    for src in _CODE.iterdir():
        dst = _BUILD / src.name
        if dst.is_symlink():
            dst.unlink()

    genmake2 = mitgcm_root / "tools" / "genmake2"
    if not genmake2.exists():
        sys.exit(f"genmake2 not found at {genmake2}")

    genmake_cmd = [
        str(genmake2),
        f"-mods={_CODE.resolve()}",
        f"-rootdir={mitgcm_root.resolve()}",
        "-mpi",
    ]
    if optfile is not None:
        genmake_cmd.append(f"-optfile={optfile.resolve()}")

    _run(genmake_cmd, cwd=_BUILD)
    _run(["make", "depend"],           cwd=_BUILD)
    _run(["make", f"-j{os.cpu_count() or 4}"], cwd=_BUILD)

    exe = _BUILD / "mitgcmuv"
    if not exe.exists():
        sys.exit("Build succeeded but mitgcmuv not found — check make output.")
    print(f"Executable: {exe}")


def generate_static_inputs(Nlon: int = 128, Nlat: int = 64) -> None:
    """Write static input files shared across all runs."""
    _INPUT.mkdir(parents=True, exist_ok=True)

    # Bathymetry: all zeros (flat aqua-planet).
    from datagen.mitgcm.ic import write_bathymetry
    write_bathymetry(_INPUT / "bathyFile.bin", Nlon=Nlon, Nlat=Nlat)
    print(f"Wrote {_INPUT / 'bathyFile.bin'}")


def write_build_info(mitgcm_root: Path) -> None:
    """Save a JSON record of the build configuration."""
    # Try to capture MITgcm git hash.
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
        "executable": str(_BUILD / "mitgcmuv"),
    }
    out = _BUILD / "build_info.json"
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
    args = ap.parse_args()

    if not args.skip_compile:
        if not args.mitgcm_root or not args.mitgcm_root.is_dir():
            sys.exit(
                "Provide --mitgcm-root or set $MITGCM_ROOT to the MITgcm source tree."
            )
        compile_mitgcm(args.mitgcm_root, args.optfile)
        write_build_info(args.mitgcm_root)

    generate_static_inputs()
    print("Build and input generation complete.")


if __name__ == "__main__":
    main()
