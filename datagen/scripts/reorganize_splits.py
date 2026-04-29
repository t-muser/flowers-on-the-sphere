"""Move processed zarr runs into train/val/test subdirectories.

Reads ``<root>/manifest.json``, draws a random split, and moves each
``<root>/processed/run_XXXX.zarr`` into ``<root>/{train,val,test}/``.
Run once per dataset before uploading to HuggingFace.

Usage::

    uv run datagen/scripts/reorganize_splits.py --root /path/to/dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _make_split(
    run_ids: list[int],
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    shuffled = rng.permutation(run_ids).tolist()
    n = len(shuffled)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)
    return {
        "train": sorted(shuffled[:n_train]),
        "val":   sorted(shuffled[n_train : n_train + n_val]),
        "test":  sorted(shuffled[n_train + n_val :]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root containing manifest.json and processed/.")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.1)
    ap.add_argument("--seed",  type=int,   default=42)
    args = ap.parse_args()

    if not abs(args.train + args.val + args.test - 1.0) < 1e-9:
        ap.error(f"Split ratios must sum to 1.0, got {args.train + args.val + args.test}")

    processed_dir = args.root / "processed"
    if not processed_dir.is_dir():
        ap.error(f"Expected {processed_dir} to exist; has the dataset already been reorganized?")

    with open(args.root / "manifest.json") as f:
        manifest = json.load(f)
    run_ids = [int(r["run_id"]) for r in manifest["runs"]]

    rng = np.random.default_rng(args.seed)
    splits = _make_split(run_ids, (args.train, args.val, args.test), rng)

    for split_name, ids in splits.items():
        split_dir = args.root / split_name
        split_dir.mkdir(exist_ok=True)
        for run_id in ids:
            src = processed_dir / f"run_{run_id:04d}.zarr"
            dst = split_dir / f"run_{run_id:04d}.zarr"
            shutil.move(str(src), str(dst))
        print(f"  {split_name}: {len(ids)} runs → {split_dir}")

    remaining = list(processed_dir.iterdir())
    if not remaining:
        processed_dir.rmdir()
        print(f"Removed empty {processed_dir}")
    else:
        print(f"Warning: {len(remaining)} files remain in {processed_dir}: {remaining[:3]}")


if __name__ == "__main__":
    main()
