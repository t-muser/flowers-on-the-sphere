"""Re-write each ``<src>/{train,val,test}/run_*.zarr`` from zarr v3 → v2.

The Daint modulus container ships zarr 2.18 / Python 3.10 and can't read
v3 stores (the dataset rsync'd from scicore is v3). We rewrite each store
in place at a parallel ``-v2`` directory, then point ``configs/data/
galewsky.yaml::root`` at it.

Run on the login node from the conda env that has zarr>=3:

    conda activate dedalus
    python scripts/convert_zarr_v3_to_v2.py \
        --src /iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/galewsky-sw \
        --dst /iopsstor/scratch/cscs/tmuser/PDEDatasets/SphericalPDEs/galewsky-sw-v2 \
        --workers 8
"""
from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import xarray as xr


def convert_one(src: Path, dst: Path) -> tuple[Path, str]:
    if dst.exists():
        return src, "skip(exists)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    # load_dataset (eager) is fine for ~350 MB per run and avoids dask
    # serialization edge cases when crossing the v3/v2 codec boundary.
    ds = xr.open_zarr(str(src), zarr_format=3).load()
    # v2 has no `serializer` codec slot. Clearing the read-side encoding
    # forces xarray to pick fresh v2-compatible defaults on write.
    for v in ds.variables:
        ds[v].encoding = {}
    ds.to_zarr(str(dst), zarr_format=2, consolidated=True, mode="w-")
    return src, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="v3 dataset root.")
    ap.add_argument("--dst", type=Path, required=True, help="v2 output root (created).")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    # Copy non-zarr top-level metadata files verbatim.
    for name in ("manifest.json", "splits.json", "stats.json"):
        s = args.src / name
        if s.is_file():
            shutil.copy2(s, args.dst / name)

    jobs: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        for store in sorted((args.src / split).glob("run_*.zarr")):
            jobs.append((store, args.dst / split / store.name))

    print(f"converting {len(jobs)} stores: {args.src} -> {args.dst}")
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_one, s, d): s for s, d in jobs}
        for i, fut in enumerate(as_completed(futures)):
            try:
                src, status = fut.result()
            except Exception as exc:
                src = futures[fut]
                status = f"ERR {exc!r}"
                n_err += 1
            else:
                if status == "ok":
                    n_ok += 1
                elif status.startswith("skip"):
                    n_skip += 1
            if (i + 1) % 50 == 0 or i == len(jobs) - 1:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                eta = (len(jobs) - i - 1) / max(rate, 1e-6)
                print(
                    f"[{i + 1:4d}/{len(jobs)}] last={src.name:18s} {status:12s} "
                    f"({rate:.2f}/s, ETA {eta / 60:.1f} min)"
                )
    dt = time.time() - t0
    print(f"done: ok={n_ok} skip={n_skip} err={n_err} in {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
