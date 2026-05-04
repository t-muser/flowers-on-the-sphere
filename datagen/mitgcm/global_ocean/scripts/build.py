"""Build the MITgcm global ocean tutorial executable.

This compiles the vendored MITgcm
``verification/tutorial_global_oce_latlon`` code customizations into an
executable at ``datagen/mitgcm/global_ocean/build/mitgcmuv``.

Run from the repository root::

    uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.build \\
        --mitgcm-root mitgcm \\
        --optfile mitgcm/tools/build_options/linux_amd64_gfortran

The runtime driver in :mod:`datagen.mitgcm.global_ocean` stages the tutorial
binary inputs directly from the vendored MITgcm tree, so this script only
compiles the executable and records build provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
_REPO_ROOT = _PKG.parents[2]
_TUTORIAL_CODE = _REPO_ROOT / "mitgcm" / "verification" / "tutorial_global_oce_latlon" / "code"
_BUILD = _PKG / "build"


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def compile_mitgcm(
    *,
    mitgcm_root: Path,
    mods_dir: Path,
    optfile: Path | None,
    use_mpi: bool,
) -> None:
    """Run genmake2 and make for the global ocean tutorial."""
    _BUILD.mkdir(parents=True, exist_ok=True)

    genmake2 = mitgcm_root / "tools" / "genmake2"
    if not genmake2.exists():
        sys.exit(f"genmake2 not found at {genmake2}")
    if not mods_dir.is_dir():
        sys.exit(f"Global-ocean mods directory not found: {mods_dir}")

    genmake_cmd = [
        str(genmake2.resolve()),
        f"-mods={mods_dir.resolve()}",
        f"-rootdir={mitgcm_root.resolve()}",
    ]
    if use_mpi:
        genmake_cmd.append("-mpi")
    if optfile is not None:
        genmake_cmd.append(f"-optfile={optfile.resolve()}")

    _run(genmake_cmd, cwd=_BUILD)
    _run(["make", "depend"], cwd=_BUILD)
    _run(["make", f"-j{os.cpu_count() or 4}"], cwd=_BUILD)

    exe = _BUILD / "mitgcmuv"
    if not exe.exists():
        sys.exit("Build completed but mitgcmuv was not produced.")
    print(f"Executable: {exe}")


def write_build_info(*, mitgcm_root: Path, mods_dir: Path, use_mpi: bool) -> None:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=mitgcm_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    info = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "tutorial_global_oce_latlon",
        "mitgcm_root": str(mitgcm_root),
        "mitgcm_git_hash": git_hash,
        "mods_dir": str(mods_dir),
        "use_mpi": use_mpi,
        "executable": str(_BUILD / "mitgcmuv"),
    }
    out = _BUILD / "build_info.json"
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Build info: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mitgcm-root",
        type=Path,
        default=Path(os.environ.get("MITGCM_ROOT", "mitgcm")),
        help="Path to the MITgcm source tree. Defaults to $MITGCM_ROOT or ./mitgcm.",
    )
    ap.add_argument(
        "--mods-dir",
        type=Path,
        default=_TUTORIAL_CODE,
        help="Code customizations directory for the global ocean tutorial.",
    )
    ap.add_argument(
        "--optfile",
        type=Path,
        default=None,
        help="genmake2 optfile, e.g. MITgcm/tools/build_options/linux_amd64_gfortran.",
    )
    ap.add_argument(
        "--serial",
        action="store_true",
        help="Build without MPI. The default is MPI with one runtime rank.",
    )
    args = ap.parse_args()

    if not args.mitgcm_root.is_dir():
        sys.exit(
            "Provide --mitgcm-root or set $MITGCM_ROOT to the MITgcm source tree."
        )

    use_mpi = not args.serial
    compile_mitgcm(
        mitgcm_root=args.mitgcm_root,
        mods_dir=args.mods_dir,
        optfile=args.optfile,
        use_mpi=use_mpi,
    )
    write_build_info(
        mitgcm_root=args.mitgcm_root,
        mods_dir=args.mods_dir,
        use_mpi=use_mpi,
    )
    print("Global ocean build complete.")


if __name__ == "__main__":
    main()
