"""Assemble the published dataset from a finished sweep.

Takes the per-run output of ``scripts/run.py`` (one ``run_XXXX/run.zarr``
under a runs directory) and produces the canonical dataset tree
consumed by :mod:`fots.data.cpl_aim_ocn`::

    <out_root>/
        train/
            run_XXXX.zarr  (symlinks to the source per-run zarrs)
            ...
        val/
        test/
        splits.json   { "train": [...], "val": [...], "test": [...] }
        stats.json    field-first { name: {mean, std, mean_delta, std_delta}}

Splits are random 80/10/10 of the successful runs (any run with a
sibling ``run_XXXX.FAILED`` file is excluded), seeded so re-running
the script produces identical splits.

Stats are computed over the **regridded lat/lon view** of the train
split — not the cs32 cube — because that is what the dataloader feeds
the model. The on-disk artifact stays cs32 native; the regridding
happens in memory here just for the stats pass and again at load time
in the loader.

Usage::

    uv run --project datagen python -m datagen.cpl_aim_ocn.scripts.finalize_dataset \\
        --runs-dir $DATA_ROOT/cpl_aim_ocn/runs \\
        --out-root $DATA_ROOT/cpl_aim_ocn

Idempotent: re-running with the same args replaces the split symlinks
and overwrites ``splits.json`` / ``stats.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from datagen.cpl_aim_ocn.channels import channel_names, expand_to_channels

_RUN_RE = re.compile(r"run_(\d+)$")
logger = logging.getLogger(__name__)


# ─── Run discovery ──────────────────────────────────────────────────────────

def discover_runs(runs_dir: Path) -> list[tuple[int, Path]]:
    """Return ``[(run_id, run_zarr_path), ...]`` for every successful run.

    A run dir is considered successful when its ``run.zarr`` exists
    *and* there is no sibling ``run_XXXX.FAILED`` marker (written by
    ``scripts/run.py`` on exception).
    """
    runs: list[tuple[int, Path]] = []
    for entry in sorted(runs_dir.iterdir()):
        m = _RUN_RE.fullmatch(entry.name)
        if not m or not entry.is_dir():
            continue
        run_id = int(m.group(1))
        run_zarr = entry / "run.zarr"
        failed_marker = runs_dir / f"{entry.name}.FAILED"
        if failed_marker.exists():
            logger.info("skip %s: FAILED marker present", entry.name)
            continue
        if not run_zarr.exists():
            logger.info("skip %s: run.zarr missing", entry.name)
            continue
        runs.append((run_id, run_zarr))
    return runs


# ─── Splits ─────────────────────────────────────────────────────────────────

def split_runs(
    run_ids: Sequence[int],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Random 80/10/10 split of ``run_ids`` with deterministic seed.

    The first split (train) gets ``floor(N * ratios[0])`` runs, the
    second (val) gets ``floor(N * ratios[1])``, and any remainder lands
    in test (so totals sum to N exactly even when N doesn't divide
    evenly). This matches the convention used by other datasets in this
    repo.
    """
    if not (0 < sum(ratios) <= 1.0 + 1e-9):
        raise ValueError(f"ratios must sum to ≤ 1; got {ratios} → {sum(ratios)}")
    if min(ratios) < 0:
        raise ValueError(f"ratios must be non-negative; got {ratios}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(run_ids))
    shuffled = [int(run_ids[i]) for i in perm]

    n = len(shuffled)
    n_train = int(np.floor(n * ratios[0]))
    n_val = int(np.floor(n * ratios[1]))
    return {
        "train": sorted(shuffled[:n_train]),
        "val":   sorted(shuffled[n_train:n_train + n_val]),
        "test":  sorted(shuffled[n_train + n_val:]),
    }


def materialise_splits(
    runs: dict[int, Path],
    splits: dict[str, list[int]],
    out_root: Path,
    *,
    copy: bool = False,
) -> None:
    """Place each run's zarr into ``out_root/<split>/run_XXXX.zarr``.

    Symlinks by default (cheap, reversible). With ``copy=True`` the
    source dir is copied tree-wise instead — use that when the source
    runs dir lives on a different filesystem than the publish root.
    Re-runnable: existing symlinks/directories at the destination are
    removed first.
    """
    for split, ids in splits.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for rid in ids:
            src = runs[rid].resolve()
            dst = split_dir / f"run_{rid:04d}.zarr"
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            if copy:
                shutil.copytree(src, dst)
            else:
                dst.symlink_to(src)


# ─── Welford parallel combine (matches ``datagen/scripts/compute_stats.py``) ─

def _combine(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
    return n, mean, M2


def _accumulate_array(acc, arr: np.ndarray):
    flat = np.asarray(arr, dtype=np.float64).ravel()
    n_r = flat.size
    if n_r == 0:
        return acc
    mean_r = float(flat.mean())
    M2_r = float(((flat - mean_r) ** 2).sum())
    return _combine(*acc, n_r, mean_r, M2_r)


# ─── Stats computation ──────────────────────────────────────────────────────

def compute_stats(
    train_zarrs: list[Path],
    *,
    nlat: int,
    nlon: int,
    method: str = "nearest",
    k: int = 4,
) -> dict[str, dict[str, float]]:
    """Stream per-channel ``mean/std/mean_delta/std_delta`` over the train split.

    Each cs32 run is opened in turn, expanded into the canonical 35
    channels via :func:`expand_to_channels`, regridded onto an
    ``(nlat, nlon)`` grid using ``regrid.build_weights`` /
    ``apply_weights``, and folded into per-channel Welford accumulators
    (one for raw values, one for per-step time differences).

    Regrid weights are built once from the XC/YC of the first run and
    reused for the rest — they're identical across runs because the
    cs32 grid is fixed by ``code_atm/SIZE.h`` / ``code_ocn/SIZE.h``.

    Returns a field-first dict ready to dump as ``stats.json``.
    """
    import xarray as xr

    from datagen.cpl_aim_ocn.regrid import apply_weights, build_weights

    if not train_zarrs:
        raise ValueError("compute_stats: train_zarrs is empty")

    names = channel_names()
    n_ch = len(names)
    acc = [(0, 0.0, 0.0)] * n_ch
    acc_delta = [(0, 0.0, 0.0)] * n_ch

    skipped: list[str] = []
    weights = None

    for i, path in enumerate(train_zarrs):
        ds = xr.open_zarr(str(path), consolidated=None)

        if weights is None:
            xc = ds["XC"].values
            yc = ds["YC"].values
            weights = build_weights(xc, yc, nlat=nlat, nlon=nlon,
                                    method=method, k=k)
            logger.info("built %s regrid weights nlat=%d nlon=%d (k=%d)",
                        method, nlat, nlon, weights.k)

        try:
            stacked = expand_to_channels(ds)         # (time, channel, face, j, i)
        except (KeyError, ValueError) as exc:
            logger.warning("skip %s: %s", path.parent.name, exc)
            skipped.append(path.parent.name)
            continue

        # Regrid this run's full (time, channel, face, j, i) tensor in one
        # shot — `apply_weights` flattens (face, j, i) internally.
        cube = stacked.transpose("time", "channel", "face", "j", "i").values
        if not np.all(np.isfinite(cube)):
            logger.warning("skip %s: non-finite values", path.parent.name)
            skipped.append(path.parent.name)
            continue

        regridded = apply_weights(cube, weights)  # (time, channel, nlat, nlon)
        if regridded.shape[1] != n_ch:
            raise AssertionError(
                f"channel-axis mismatch: regridded {regridded.shape[1]} vs "
                f"channel_names() {n_ch}"
            )

        deltas = np.diff(regridded, axis=0)            # (time-1, channel, nlat, nlon)

        for c in range(n_ch):
            acc[c] = _accumulate_array(acc[c], regridded[:, c])
            acc_delta[c] = _accumulate_array(acc_delta[c], deltas[:, c])

        if (i + 1) % 25 == 0 or (i + 1) == len(train_zarrs):
            logger.info("processed %d/%d", i + 1, len(train_zarrs))

    stats: dict[str, dict[str, float]] = {}
    for c, name in enumerate(names):
        n, mean, M2 = acc[c]
        std = float(np.sqrt(M2 / n)) if n > 1 else 0.0
        n_d, mean_d, M2_d = acc_delta[c]
        std_d = float(np.sqrt(M2_d / n_d)) if n_d > 1 else 0.0
        stats[name] = {
            "mean":       round(float(mean), 6),
            "std":        round(float(std), 6),
            "mean_delta": round(float(mean_d), 6),
            "std_delta":  round(float(std_d), 6),
        }

    if skipped:
        logger.warning("skipped %d non-conforming runs: %s",
                       len(skipped), skipped)
    return stats


# ─── CLI ────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True,
                    help="Directory containing run_XXXX/ subdirs.")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Destination dataset root.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the split (default 42).")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="Train/val/test split ratios (default 0.8 0.1 0.1).")
    ap.add_argument("--copy", action="store_true",
                    help="Copy run dirs instead of symlinking.")
    ap.add_argument("--nlat", type=int, default=64,
                    help="Stats target grid latitudes (default 64).")
    ap.add_argument("--nlon", type=int, default=128,
                    help="Stats target grid longitudes (default 128).")
    ap.add_argument("--regrid-method", choices=("nearest", "idw"),
                    default="nearest",
                    help="Regridder for the stats pass (default nearest).")
    ap.add_argument("--regrid-k", type=int, default=4,
                    help="k for IDW regridder (default 4, ignored for nearest).")
    ap.add_argument("--skip-stats", action="store_true",
                    help="Materialise splits but do not compute stats.json.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    runs_dir: Path = args.runs_dir.resolve()
    out_root: Path = args.out_root.resolve()
    if not runs_dir.is_dir():
        sys.exit(f"runs-dir does not exist: {runs_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    discovered = discover_runs(runs_dir)
    if not discovered:
        sys.exit(f"No successful runs found under {runs_dir}")
    runs_by_id: dict[int, Path] = {rid: p for rid, p in discovered}
    logger.info("found %d successful runs in %s", len(runs_by_id), runs_dir)

    splits = split_runs(sorted(runs_by_id.keys()),
                        ratios=tuple(args.ratios), seed=args.seed)
    logger.info("split: train=%d val=%d test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    materialise_splits(runs_by_id, splits, out_root, copy=args.copy)
    splits_path = out_root / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
        f.write("\n")
    logger.info("wrote %s", splits_path)

    if args.skip_stats:
        return 0

    train_zarrs = [out_root / "train" / f"run_{rid:04d}.zarr"
                   for rid in splits["train"]]
    stats = compute_stats(train_zarrs,
                          nlat=args.nlat, nlon=args.nlon,
                          method=args.regrid_method, k=args.regrid_k)

    stats_path = out_root / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    logger.info("wrote %s", stats_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
