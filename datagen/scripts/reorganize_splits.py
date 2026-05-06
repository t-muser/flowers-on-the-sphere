"""Move processed zarr runs into train/val/test subdirectories.

Reads ``<root>/splits.json`` (produced by ``generate_split.py``) and moves each
source zarr into ``<root>/{train,val,test}/run_XXXX.zarr``. Two source layouts
are supported:

* **flat** (default): ``<root>/<src-dir>/run_XXXX.zarr``
* **nested**: ``<root>/<src-dir>/run_XXXX/run.zarr`` (used by the MITgcm
  pipelines, where each run dir also holds raw solver artifacts)

Run once per dataset before uploading to HuggingFace.

Usage::

    uv run datagen/scripts/reorganize_splits.py --root /path/to/dataset
    uv run datagen/scripts/reorganize_splits.py --root /path/to/dataset --src-dir sweep --nested
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _move_with_optional_trim(src: Path, dst: Path, trim_first: int) -> None:
    """Move ``src`` zarr to ``dst``. If ``trim_first > 0``, slice the time
    axis on the way out and rebase ``time`` so ``time[0] == 0``."""
    if trim_first <= 0:
        shutil.move(str(src), str(dst))
        return

    import xarray as xr

    ds = xr.open_zarr(str(src))
    if "time" not in ds.dims:
        raise ValueError(f"{src}: --trim-first requires a 'time' dimension")
    if trim_first >= ds.sizes["time"]:
        raise ValueError(
            f"{src}: --trim-first {trim_first} >= time size {ds.sizes['time']}"
        )
    trimmed = ds.isel(time=slice(trim_first, None))
    trimmed = trimmed.assign_coords(time=trimmed["time"] - trimmed["time"][0])
    # Load into memory before we can delete the source.
    trimmed = trimmed.load()
    ds.close()
    trimmed.to_zarr(str(dst), mode="w", consolidated=True)
    shutil.rmtree(str(src))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root containing splits.json (and the source dir).")
    ap.add_argument("--splits", type=Path, default=None,
                    help="Path to splits.json (defaults to <root>/splits.json).")
    ap.add_argument("--src-dir", default="processed",
                    help="Source subdir of <root> with run zarrs "
                         "(default: processed).")
    ap.add_argument("--nested", action="store_true",
                    help="Source is <src-dir>/run_XXXX/run.zarr instead of "
                         "<src-dir>/run_XXXX.zarr.")
    ap.add_argument("--trim-first", type=int, default=0,
                    help="Drop the first N snapshots and rebase 'time' to "
                         "start at 0. Use to bake spinup discard into the "
                         "split (default: 0 = no trim).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers for the move+trim step "
                         "(default: 1, sequential). Each move is independent, "
                         "so this scales near-linearly until disk-bound.")
    args = ap.parse_args()

    processed_dir = args.root / args.src_dir
    if not processed_dir.is_dir():
        ap.error(f"Expected {processed_dir} to exist; has the dataset already been reorganized?")

    splits_path = args.splits or (args.root / "splits.json")
    with open(splits_path) as f:
        splits_data = json.load(f)

    print(f"Loaded splits from {splits_path}")
    print(f"  strategy: {splits_data.get('strategy')}  seed: {splits_data.get('seed')}")
    print(f"  counts:   {splits_data['counts']}")
    if args.trim_first:
        print(f"  trim_first: {args.trim_first} snapshots dropped, time rebased to 0")

    jobs: list[tuple[str, Path, Path]] = []
    for split_name in ("train", "val", "test"):
        ids = splits_data[split_name]
        split_dir = args.root / split_name
        split_dir.mkdir(exist_ok=True)
        for run_id in ids:
            if args.nested:
                src = processed_dir / f"run_{run_id:04d}" / "run.zarr"
            else:
                src = processed_dir / f"run_{run_id:04d}.zarr"
            dst = split_dir / f"run_{run_id:04d}.zarr"
            # Skip if the destination is already a complete zarr store
            # (zarr.json for v3; .zgroup for v2). A partial source dir from
            # an interrupted previous move can leave src.is_dir() == True
            # while dst is the canonical, valid copy — so check dst, not src.
            if dst.is_dir() and (
                (dst / "zarr.json").exists() or (dst / ".zgroup").exists()
            ):
                continue
            jobs.append((split_name, src, dst))

    print(f"Total moves to perform: {len(jobs)} (workers={args.workers})")
    if args.workers <= 1:
        for split_name, src, dst in jobs:
            _move_with_optional_trim(src, dst, args.trim_first)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_move_with_optional_trim, src, dst, args.trim_first):
                    (split_name, src, dst)
                for split_name, src, dst in jobs
            }
            done = 0
            for fut in as_completed(futures):
                split_name, src, dst = futures[fut]
                fut.result()  # surface exceptions
                done += 1
                if done % 10 == 0 or done == len(futures):
                    print(f"  moved {done}/{len(futures)}")

    for split_name in ("train", "val", "test"):
        n = len(splits_data[split_name])
        split_dir = args.root / split_name
        print(f"  {split_name}: {n} runs → {split_dir}")

    if args.nested:
        # Per-run dirs likely still hold solver artifacts (MDS files, STDOUTs).
        # Leave them in place — the user can `rm -rf` after verifying the
        # split looks right.
        leftover = sorted(processed_dir.glob("run_*"))
        if leftover:
            print(f"Note: {len(leftover)} per-run dirs remain in {processed_dir} "
                  f"(solver artifacts, ~{leftover[0].name} etc.). "
                  "Delete manually once you've verified the split.")
    else:
        remaining = list(processed_dir.iterdir())
        if not remaining:
            processed_dir.rmdir()
            print(f"Removed empty {processed_dir}")
        else:
            print(f"Warning: {len(remaining)} files remain in {processed_dir}: {remaining[:3]}")


if __name__ == "__main__":
    main()
