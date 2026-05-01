"""Move processed zarr runs into train/val/test subdirectories.

Reads ``<root>/splits.json`` (produced by ``generate_split.py``) and moves each
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root containing splits.json and processed/.")
    ap.add_argument("--splits", type=Path, default=None,
                    help="Path to splits.json (defaults to <root>/splits.json).")
    args = ap.parse_args()

    processed_dir = args.root / "processed"
    if not processed_dir.is_dir():
        ap.error(f"Expected {processed_dir} to exist; has the dataset already been reorganized?")

    splits_path = args.splits or (args.root / "splits.json")
    with open(splits_path) as f:
        splits_data = json.load(f)

    print(f"Loaded splits from {splits_path}")
    print(f"  strategy: {splits_data.get('strategy')}  seed: {splits_data.get('seed')}")
    print(f"  counts:   {splits_data['counts']}")

    for split_name in ("train", "val", "test"):
        ids = splits_data[split_name]
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
