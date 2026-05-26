"""Finalize the ClimaAtmos held-suarez-clima dataset.

The actual finalize logic (manifest copy, stratified splits, symlinks,
stats) is grid-agnostic and lives in
``datagen.mitgcm.held_suarez.scripts.finalize_3d``. We just reuse it here
with paths pointed at the ClimaAtmos producer's output tree, and pre-flight
that every run produced a ``run.zarr`` under ``runs/run_XXXX/`` (the
ClimaAtmos producer writes Zarr alongside the raw ClimaAtmos NetCDF
output, mirroring the MITgcm layout).

Run from the repo root::

    uv run --project datagen python -m datagen.clima_atmos.held_suarez.scripts.finalize_dataset \\
        --root /scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-clima \\
        --manifest datagen/clima_atmos/held_suarez/configs/manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


from datagen.mitgcm.held_suarez.scripts.finalize_3d import (
    compute_stats, link_runs_into_splits, write_splits,
)


_DEFAULT_MANIFEST = Path("datagen/clima_atmos/held_suarez/configs/manifest.json")


def _check_runs_exist(root: Path) -> list[int]:
    """Sanity-check that ``runs/run_XXXX/run.zarr`` is present for every
    run that completed. Returns the sorted list of available run-ids."""
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"runs/ directory missing under {root}")
    avail: list[int] = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("run_"):
            continue
        zarr_path = sub / "run.zarr"
        if zarr_path.is_dir():
            try:
                avail.append(int(sub.name.split("_")[1]))
            except ValueError:
                continue
    return avail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root (contains runs/ from the production array).")
    ap.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST,
                    help="Sweep manifest from generate_sweep.")
    ap.add_argument("--stratify-on", default="tau_drag_days",
                    help="Param to stratify the split on (default: tau_drag_days).")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.1)
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--skip-stats", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")

    print(f"Finalising {root}")
    avail = _check_runs_exist(root)
    print(f"  found {len(avail)} run zarrs under runs/")

    import shutil
    print("\n[1/4] copy manifest")
    out_manifest = root / "manifest.json"
    shutil.copyfile(args.manifest, out_manifest)
    print(f"  copied {args.manifest} → {out_manifest}")

    print("\n[2/4] write splits.json")
    splits = write_splits(
        args.manifest, root / "splits.json",
        (args.train, args.val, args.test),
        args.stratify_on, args.seed,
    )

    print("\n[3/4] symlink runs into split dirs")
    link_runs_into_splits(root, splits)

    if not args.skip_stats:
        print("\n[4/4] compute stats over train split")
        stats = compute_stats(root, splits["train"])
        out_stats = root / "stats.json"
        import json
        out_stats.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"  wrote {out_stats}")
    else:
        print("\n[4/4] stats skipped")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
