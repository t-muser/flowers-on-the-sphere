"""Verify a transferred Zarr dataset by decompressing every chunk.

Checks completeness against splits.json/manifest.json, reports pending
.zarr.tar files separately, and streams chunks instead of slurping whole
arrays into memory.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr


def verify_run(path: Path) -> tuple[Path, str]:
    try:
        g = zarr.open(str(path), mode="r")
        for name in sorted(g.array_keys()):
            arr = g[name]
            # Iterate chunk-by-chunk; force decompression but don't hold the
            # whole array in RAM.
            cdata_shape = tuple(
                -(-s // c) for s, c in zip(arr.shape, arr.chunks)
            )
            for block in np.ndindex(*cdata_shape):
                sl = tuple(
                    slice(b * c, min((b + 1) * c, s))
                    for b, c, s in zip(block, arr.chunks, arr.shape)
                )
                _ = arr[sl]
        return path, "OK"
    except Exception as e:
        return path, f"FAIL: {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    print(f"zarr {zarr.__version__}")
    root: Path = args.root

    # --- sidecar checks -------------------------------------------------
    sidecars = ["manifest.json", "splits.json", "stats.json"]
    for s in sidecars:
        if not (root / s).is_file():
            print(f"MISSING sidecar: {s}")
    grid = root / "grid.zarr"
    if grid.is_dir():
        path, status = verify_run(grid)
        if status != "OK":
            print(f"grid.zarr {status}")
    else:
        print("MISSING sidecar: grid.zarr")

    # --- completeness ---------------------------------------------------
    splits_path = root / "splits.json"
    expected: dict[str, set[int]] = {}
    if splits_path.is_file():
        splits = json.loads(splits_path.read_text())
        for s in ("train", "val", "test"):
            expected[s] = set(splits.get(s, []))
    else:
        print("WARNING: no splits.json — skipping completeness check")

    pending_tars = sorted(root.glob("run_*.zarr.tar"))
    if pending_tars:
        print(f"\n{len(pending_tars)} pending (still tarred):")
        for t in pending_tars:
            print(f"  {t.name}")

    runs: list[Path] = []
    missing: list[str] = []
    for s in ("train", "val", "test"):
        present_paths = sorted((root / s).glob("run_*.zarr"))
        present_ids = {int(p.stem.split("_")[1]) for p in present_paths}
        runs += present_paths
        if s in expected:
            miss = expected[s] - present_ids
            tarred = {int(t.name.split("_")[1].split(".")[0]) for t in pending_tars}
            miss -= tarred  # don't double-count in-flight runs
            for rid in sorted(miss):
                missing.append(f"{s}/run_{rid:04d}.zarr")

    if missing:
        print(f"\n{len(missing)} MISSING run(s):")
        for m in missing:
            print(f"  {m}")

    # --- chunk-decode pass ---------------------------------------------
    print(f"\nverifying {len(runs)} runs with {args.workers} workers")
    ok = fail = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed({ex.submit(verify_run, r): r for r in runs}):
            path, status = fut.result()
            if status == "OK":
                ok += 1
            else:
                fail += 1
                rel = path.relative_to(root)
                print(f"{rel} {status}")
                failures.append(str(rel))

    print(f"\n=== {ok} OK, {fail} FAIL, {len(missing)} MISSING, "
          f"{len(pending_tars)} PENDING ===")
    sys.exit(0 if (fail == 0 and not missing) else 1)


if __name__ == "__main__":
    main()
