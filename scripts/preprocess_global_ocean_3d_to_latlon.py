"""Preprocess the global-ocean-3D dataset onto a lat/lon grid.

The cs32→(nlat, nlon) IDW regrid + uv rotation + land imputation are
deterministic, so doing them once per run (rather than every
__getitem__) eliminates the dataloader bottleneck. Output layout
(per run.zarr):

    theta, salt, u, v: (time, level=Nr, lat, lon) float32
    eta              : (time, lat, lon) float32
    coords: time, level (1..Nr), lat, lon

All Nr levels are kept on disk so level-subset experiments don't
need a re-precompute. Channel-axis stacking happens at training time.

Usage (typically via scripts/preprocess_global_ocean_3d_to_latlon.sbatch):

    /capstor/scratch/cscs/tmuser/miniforge3/bin/python \\
        scripts/preprocess_global_ocean_3d_to_latlon.py \\
            --src /iopsstor/.../global-ocean-3D-v2 \\
            --dst /iopsstor/.../global-ocean-3D-latlon \\
            --nlat 64 --nlon 128 --workers 16
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from datagen.mitgcm.global_ocean.regrid import (
    FIELDS_2D,
    FIELDS_3D,
    apply_dynamic_3d,
    build as build_lat_lon,
)

logger = logging.getLogger("preprocess_latlon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _convert_one(src: Path, dst: Path, grid_ll, level_idx: np.ndarray,
                 chunk_time: int) -> tuple[Path, float]:
    t0 = time.time()
    ds = xr.open_zarr(str(src), consolidated=True)

    fields_3d = {name: ds[name].values for name in FIELDS_3D}
    fields_2d = {name: ds[name].values for name in FIELDS_2D}

    # (time, C=4*Nr+1, nlat, nlon)
    flat = apply_dynamic_3d(
        fields_3d, fields_2d, grid_ll,
        level_idx=level_idx, impute_land=True,
    )

    Nr = int(level_idx.size)
    Nt = flat.shape[0]
    nlat, nlon = grid_ll.nlat, grid_ll.nlon

    theta = flat[:, 0 * Nr:1 * Nr]
    salt  = flat[:, 1 * Nr:2 * Nr]
    u     = flat[:, 2 * Nr:3 * Nr]
    v     = flat[:, 3 * Nr:4 * Nr]
    eta   = flat[:, 4 * Nr]

    time_coord = ds["time"].values
    lat_coord = np.linspace(-90.0 + 90.0 / nlat, 90.0 - 90.0 / nlat, nlat,
                            dtype=np.float32)
    lon_coord = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float32)
    level_coord = np.asarray(level_idx, dtype=np.int64)

    chunks_3d = {"time": chunk_time, "level": Nr, "lat": nlat, "lon": nlon}
    chunks_2d = {"time": chunk_time, "lat": nlat, "lon": nlon}

    out = xr.Dataset(
        {
            "theta": (("time", "level", "lat", "lon"), theta.astype(np.float32)),
            "salt":  (("time", "level", "lat", "lon"), salt.astype(np.float32)),
            "u":     (("time", "level", "lat", "lon"), u.astype(np.float32)),
            "v":     (("time", "level", "lat", "lon"), v.astype(np.float32)),
            "eta":   (("time", "lat", "lon"),          eta.astype(np.float32)),
        },
        coords={
            "time":  ("time",  time_coord),
            "level": ("level", level_coord),
            "lat":   ("lat",   lat_coord),
            "lon":   ("lon",   lon_coord),
        },
        attrs={"source": str(src), "Nt": Nt, "Nr": Nr,
               "nlat": nlat, "nlon": nlon},
    )
    out["theta"].encoding = {"chunks": tuple(chunks_3d[d] for d in out["theta"].dims)}
    out["salt"].encoding  = {"chunks": tuple(chunks_3d[d] for d in out["salt"].dims)}
    out["u"].encoding     = {"chunks": tuple(chunks_3d[d] for d in out["u"].dims)}
    out["v"].encoding     = {"chunks": tuple(chunks_3d[d] for d in out["v"].dims)}
    out["eta"].encoding   = {"chunks": tuple(chunks_2d[d] for d in out["eta"].dims)}

    if dst.exists():
        shutil.rmtree(dst)
    out.to_zarr(str(dst), mode="w", zarr_format=2, consolidated=True)

    return src, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--nlat", type=int, default=64)
    p.add_argument("--nlon", type=int, default=128)
    p.add_argument("--regrid-method", default="idw")
    p.add_argument("--regrid-k", type=int, default=4)
    p.add_argument("--chunk-time", type=int, default=50)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        raise SystemExit(f"src not a dir: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # Copy sidecars + grid.zarr verbatim. Stats keys (theta_k01, ..., eta)
    # match the regridded channel naming, so reuse is safe.
    for fname in ("splits.json", "manifest.json", "stats.json"):
        sp = src / fname
        if sp.is_file():
            shutil.copy2(sp, dst / fname)
            logger.info("copied %s", fname)
    grid_dst = dst / "grid.zarr"
    if grid_dst.exists():
        shutil.rmtree(grid_dst)
    shutil.copytree(src / "grid.zarr", grid_dst)
    logger.info("copied grid.zarr")

    grid_ll = build_lat_lon(
        src / "grid.zarr", nlat=args.nlat, nlon=args.nlon,
        method=args.regrid_method, k=args.regrid_k,
    )
    if grid_ll.mask_c_3d_src is None:
        raise SystemExit(
            "grid.zarr lacks mask_c_3d / mask_w_3d — rebuild via the 3-D "
            "extract_grid before preprocessing."
        )
    Nr = int(grid_ll.mask_c_3d_src.shape[0])
    level_idx = np.arange(1, Nr + 1, dtype=np.int64)
    logger.info("regrid %d→(%d, %d) method=%s k=%d Nr=%d",
                grid_ll.mask_c_3d_src.shape[1],
                args.nlat, args.nlon, args.regrid_method, args.regrid_k, Nr)

    tasks: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        split_src = src / split
        if not split_src.is_dir():
            continue
        split_dst = dst / split
        split_dst.mkdir(exist_ok=True)
        for run in sorted(split_src.iterdir()):
            if run.is_dir() and run.suffix == ".zarr":
                tasks.append((run, split_dst / run.name))

    logger.info("converting %d run.zarrs with %d workers", len(tasks), args.workers)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(_convert_one, s, d, grid_ll, level_idx, args.chunk_time)
            for s, d in tasks
        ]
        for f in as_completed(futs):
            spath, dt = f.result()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                logger.info("[%d/%d] %s in %.1fs", done, len(tasks), spath.name, dt)
    logger.info("all %d done in %.1fs", len(tasks), time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
