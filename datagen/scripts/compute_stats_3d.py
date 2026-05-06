"""Per-channel z-score stats for the 3-D global-ocean dataset.

The 3-D run.zarr stores expose per-variable arrays:

* ``theta``, ``salt``, ``u``, ``v``: ``(time, level, face, y, x)``
* ``eta``: ``(time, face, y, x)``

This script emits one ``stats.json`` entry per channel produced by
:func:`datagen.mitgcm.global_ocean.regrid.field_names_3d` — i.e.
``theta_k01``, ``theta_k02``, ..., ``eta`` — using per-level wet-cell
masks (``mask_c_3d`` for theta/salt; ``mask_w_3d`` for u/v; ``mask_eta``
for eta) from ``grid.zarr``. Stats are computed via streaming Welford
aggregation (one run + one level slab in memory at a time).

The work is embarassingly parallel across runs: each train zarr is
independent. The script splits the run list across ``--workers``
processes and combines per-worker accumulators with the parallel
Welford combine.

Usage::

    uv run --project datagen python datagen/scripts/compute_stats_3d.py \\
        --root /path/to/global-ocean-3D \\
        --grid /path/to/global-ocean-3D/grid.zarr \\
        --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

# Mirrors datagen.mitgcm.global_ocean.regrid.{FIELDS_3D,FIELDS_2D}.
# Hardcoded so this script runs as a plain `python script.py` invocation
# without needing the repo root on sys.path.
FIELDS_3D: tuple[str, ...] = ("theta", "salt", "u", "v")
FIELDS_2D: tuple[str, ...] = ("eta",)


_RUN_RE = re.compile(r"run_(\d+)\.zarr")

_FIELD_TO_MASK_3D = {"theta": "mask_c_3d", "salt": "mask_c_3d",
                     "u": "mask_w_3d",     "v": "mask_w_3d"}


def _scan_zarrs(split_dir: Path) -> list[Path]:
    paths = sorted(p for p in split_dir.iterdir() if _RUN_RE.fullmatch(p.name))
    if not paths:
        raise FileNotFoundError(f"No run_XXXX.zarr files found in {split_dir}")
    return paths


def _combine(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    """Parallel Welford combine for two accumulators."""
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
    return n, mean, M2


def _batch_stats(flat: np.ndarray) -> tuple[int, float, float]:
    n = flat.size
    if n == 0:
        return 0, 0.0, 0.0
    mean = float(flat.mean())
    M2 = float(((flat - mean) ** 2).sum())
    return n, mean, M2


def _process_zarrs(
    paths: list[str],
    mask_c_3d: np.ndarray,
    mask_w_3d: np.ndarray,
    mask_eta: np.ndarray,
    Nr: int,
    worker_id: int,
) -> tuple[dict, dict, list[str]]:
    """Worker entry point. Returns (acc, acc_d, skipped) for the path subset."""
    acc: dict[str, tuple[int, float, float]] = {}
    acc_d: dict[str, tuple[int, float, float]] = {}
    skipped: list[str] = []

    for i, path_str in enumerate(paths):
        path = Path(path_str)
        ds = xr.open_zarr(str(path), consolidated=None)

        run_has_nonfinite = False
        for var in FIELDS_3D:
            data = ds[var].values  # (time, level, face, y, x)
            if not np.all(np.isfinite(data)):
                run_has_nonfinite = True
                break
            delta = np.diff(data, axis=0)
            mask_var = mask_c_3d if _FIELD_TO_MASK_3D[var] == "mask_c_3d" else mask_w_3d
            for k in range(Nr):
                name = f"{var}_k{k + 1:02d}"
                m = mask_var[k]
                flat = data[:, k][..., m].ravel().astype(np.float64)
                flat_d = delta[:, k][..., m].ravel().astype(np.float64)
                acc[name] = _combine(*acc.get(name, (0, 0.0, 0.0)),
                                     *_batch_stats(flat))
                acc_d[name] = _combine(*acc_d.get(name, (0, 0.0, 0.0)),
                                       *_batch_stats(flat_d))

        if run_has_nonfinite:
            skipped.append(path.name)
            continue

        for var in FIELDS_2D:
            data = ds[var].values  # (time, face, y, x)
            if not np.all(np.isfinite(data)):
                run_has_nonfinite = True
                break
            delta = np.diff(data, axis=0)
            flat = data[..., mask_eta].ravel().astype(np.float64)
            flat_d = delta[..., mask_eta].ravel().astype(np.float64)
            acc[var] = _combine(*acc.get(var, (0, 0.0, 0.0)),
                                *_batch_stats(flat))
            acc_d[var] = _combine(*acc_d.get(var, (0, 0.0, 0.0)),
                                  *_batch_stats(flat_d))

        if run_has_nonfinite:
            skipped.append(path.name)
            continue

        if (i + 1) % 5 == 0 or (i + 1) == len(paths):
            print(f"  worker {worker_id}: processed {i + 1}/{len(paths)}", flush=True)

    return acc, acc_d, skipped


def _split_paths(paths: list[Path], n: int) -> list[list[str]]:
    """Round-robin partition so each worker gets a similar mix of run sizes."""
    buckets: list[list[str]] = [[] for _ in range(n)]
    for i, p in enumerate(paths):
        buckets[i % n].append(str(p))
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root containing train/ subdir.")
    ap.add_argument("--split", default="train",
                    help="Which split to compute stats over (default: train).")
    ap.add_argument("--grid", type=Path, default=None,
                    help="Path to grid.zarr (defaults to <root>/grid.zarr). "
                         "Must contain mask_c_3d, mask_w_3d, mask_eta.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (defaults to <root>/stats.json).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers (default: 1, sequential).")
    args = ap.parse_args()

    split_dir = args.root / args.split
    zarr_paths = _scan_zarrs(split_dir)

    grid_path = args.grid or (args.root / "grid.zarr")
    grid = xr.open_zarr(str(grid_path))
    for required in ("mask_c_3d", "mask_w_3d", "mask_eta"):
        if required not in grid:
            ap.error(f"{grid_path} missing {required!r}; "
                     "rebuild grid.zarr with the 3-D extract_grid extension.")
    mask_c_3d = grid["mask_c_3d"].values.astype(bool)  # (Nr, face, y, x)
    mask_w_3d = grid["mask_w_3d"].values.astype(bool)
    mask_eta = grid["mask_eta"].values.astype(bool)
    Nr = int(mask_c_3d.shape[0])

    print(f"Computing 3-D stats over {len(zarr_paths)} runs in {split_dir} "
          f"(workers={args.workers})", flush=True)
    print(f"  grid:   {grid_path}  (Nr={Nr})", flush=True)

    t0 = time.time()
    if args.workers <= 1:
        acc, acc_d, skipped = _process_zarrs(
            [str(p) for p in zarr_paths],
            mask_c_3d, mask_w_3d, mask_eta, Nr, 0,
        )
    else:
        acc = {}
        acc_d = {}
        skipped: list[str] = []
        buckets = _split_paths(zarr_paths, args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_zarrs,
                            buckets[w], mask_c_3d, mask_w_3d, mask_eta, Nr, w): w
                for w in range(args.workers) if buckets[w]
            }
            for fut in as_completed(futures):
                w = futures[fut]
                w_acc, w_acc_d, w_skipped = fut.result()
                for name, (n, mean, M2) in w_acc.items():
                    acc[name] = _combine(*acc.get(name, (0, 0.0, 0.0)),
                                         n, mean, M2)
                for name, (n, mean, M2) in w_acc_d.items():
                    acc_d[name] = _combine(*acc_d.get(name, (0, 0.0, 0.0)),
                                           n, mean, M2)
                skipped.extend(w_skipped)
                print(f"  worker {w} done", flush=True)

    print(f"All workers done in {time.time() - t0:.1f}s", flush=True)

    stats: dict[str, dict] = {}
    # Emit channels in the same order as field_names_3d for stable inspection.
    channel_names: list[str] = []
    for var in FIELDS_3D:
        for k in range(Nr):
            channel_names.append(f"{var}_k{k + 1:02d}")
    channel_names.extend(FIELDS_2D)

    for name in channel_names:
        n, mean, M2 = acc[name]
        std = float(np.sqrt(M2 / n)) if n > 1 else 0.0
        n_d, mean_d, M2_d = acc_d[name]
        std_d = float(np.sqrt(M2_d / n_d)) if n_d > 1 else 0.0
        stats[name] = {
            "mean":       round(float(mean), 6),
            "std":        round(float(std), 6),
            "mean_delta": round(float(mean_d), 6),
            "std_delta":  round(float(std_d), 6),
        }
        print(f"  {name:12s}  mean={mean:.4g}  std={std:.4g}"
              f"  mean_delta={mean_d:.4g}  std_delta={std_d:.4g}",
              flush=True)

    out_path = args.out or (args.root / "stats.json")
    with open(out_path, "w") as fp:
        json.dump(stats, fp, indent=2)
        fp.write("\n")
    print(f"\nWrote {out_path}", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} non-finite runs: {skipped}", flush=True)


if __name__ == "__main__":
    main()
