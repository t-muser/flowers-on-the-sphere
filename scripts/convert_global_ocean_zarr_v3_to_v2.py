"""Rewrite the global-ocean dataset from zarr v3 to zarr v2.

The MITgcm cs32x15 dataset (243 runs + grid + json metadata files) was
emitted as zarr v3. The aurora_arm64_modulus2412 container ships zarr
2.x on Python 3.10, which cannot read v3 stores and cannot be upgraded
(zarr-python 3.x requires Python >= 3.11). This converter reads each v3
store with a miniforge env (zarr 3.x, Python 3.13) and rewrites to a
sibling tree as v2. Chunks must be re-emitted because v2 and v3 use
different chunk filename conventions, so this is a full
read+decompress+recompress pass — but the dataset is only ~22 GiB and
the runs are independent, so a small process pool finishes in minutes.

Usage:
    /capstor/scratch/cscs/tmuser/miniforge3/bin/python \
        scripts/convert_global_ocean_zarr_v3_to_v2.py \
            --src /iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean \
            --dst /iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/global-ocean-v2 \
            --workers 16
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import xarray as xr

logger = logging.getLogger("convert_v3_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _convert_one(src: Path, dst: Path) -> tuple[Path, float]:
    t0 = time.time()
    ds = xr.open_zarr(str(src)).load()
    # v3 stores encode a `serializer` codec slot that v2 arrays don't
    # have. Drop the source encoding so to_zarr picks v2 defaults.
    for v in list(ds.data_vars) + list(ds.coords):
        ds[v].encoding = {}
    if dst.exists():
        shutil.rmtree(dst)
    ds.to_zarr(str(dst), mode="w", zarr_format=2, consolidated=True)
    return src, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        raise SystemExit(f"src not a dir: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ("splits.json", "manifest.json", "stats.json"):
        sp = src / fname
        if sp.is_file():
            shutil.copy2(sp, dst / fname)
            logger.info("copied %s", fname)

    tasks: list[tuple[Path, Path]] = []
    grid_src = src / "grid.zarr"
    if grid_src.is_dir():
        tasks.append((grid_src, dst / "grid.zarr"))

    for split in ("train", "val", "test"):
        split_src = src / split
        if not split_src.is_dir():
            continue
        split_dst = dst / split
        split_dst.mkdir(exist_ok=True)
        for run in sorted(split_src.iterdir()):
            if run.is_dir() and run.suffix == ".zarr":
                tasks.append((run, split_dst / run.name))

    logger.info("converting %d zarr stores with %d workers", len(tasks), args.workers)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_convert_one, s, d) for s, d in tasks]
        for f in as_completed(futs):
            spath, dt = f.result()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                logger.info("[%d/%d] %s in %.1fs", done, len(tasks), spath.name, dt)
    logger.info("all %d done in %.1fs", len(tasks), time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
