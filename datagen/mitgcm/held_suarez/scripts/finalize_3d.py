"""Finalize the held-suarez-3d dataset: assemble splits, compute stats.

Run *after* the 72-run production array completes. Performs:

1. Copies the sweep manifest from ``configs/safe/manifest.json`` to
   ``<root>/manifest.json``.
2. Generates ``<root>/splits.json`` (train/val/test by run_id, stratified on
   ``tau_drag_days`` so each split sees all 3 drag values).
3. Symlinks each run's zarr from ``runs/run_XXXX/run.zarr`` into
   ``<root>/<split>/run_XXXX.zarr`` so downstream loaders can use the
   conventional layout.
4. Computes per-(var, level) normalisation stats over the train split via
   parallel-Welford and writes ``<root>/stats.json``.

Stats schema matches the existing single-level convention:
``u_<P>hpa, v_<P>hpa, T_<P>hpa`` for the 3-D vars at each requested
pressure level, plus ``ps`` for the 2-D surface pressure.

Run from the repo root::

    uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.finalize_3d \\
        --root /scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-3d
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


_DEFAULT_MANIFEST = Path("datagen/mitgcm/held_suarez/configs/safe/manifest.json")


# ─── splits ──────────────────────────────────────────────────────────────────

def _assign_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1; got {ratios} → {sum(ratios)}")
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _stratified_split(manifest_runs, ratios, stratify_on: str, rng):
    by_stratum: dict[float, list[int]] = {}
    for entry in manifest_runs:
        key = float(entry["params"][stratify_on])
        by_stratum.setdefault(key, []).append(int(entry["run_id"]))

    splits = {"train": [], "val": [], "test": []}
    for ids in by_stratum.values():
        shuffled = rng.permutation(ids).tolist()
        n_train, n_val, _ = _assign_sizes(len(shuffled), ratios)
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])
    return {k: sorted(v) for k, v in splits.items()}


def write_splits(manifest_path: Path, out_path: Path, ratios, stratify_on: str, seed: int):
    with open(manifest_path) as f:
        manifest = json.load(f)
    rng = np.random.default_rng(seed)
    splits = _stratified_split(manifest["runs"], ratios, stratify_on, rng)
    payload = {
        "seed": seed,
        "strategy": "stratified",
        "stratify_on": stratify_on,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "n_runs": len(manifest["runs"]),
        "counts": {k: len(v) for k, v in splits.items()},
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {out_path}  counts={payload['counts']}")
    return splits


# ─── symlinks ────────────────────────────────────────────────────────────────

def link_runs_into_splits(root: Path, splits: dict[str, list[int]]) -> None:
    runs_dir = root / "runs"
    for split, run_ids in splits.items():
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        n_linked = 0
        for rid in run_ids:
            src = runs_dir / f"run_{rid:04d}" / "run.zarr"
            dst = split_dir / f"run_{rid:04d}.zarr"
            if not src.is_dir():
                raise FileNotFoundError(f"missing zarr for run_{rid:04d}: {src}")
            if dst.is_symlink() or dst.exists():
                dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
            dst.symlink_to(src.resolve())
            n_linked += 1
        print(f"  linked {n_linked} runs into {split_dir}")


# ─── stats (parallel Welford) ────────────────────────────────────────────────

def _combine(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
    return n, mean, M2


def _accumulate(arr_flat: np.ndarray, acc):
    """Add a chunk to a Welford accumulator."""
    n_r = arr_flat.size
    if n_r == 0:
        return acc
    mean_r = float(arr_flat.mean())
    M2_r = float(((arr_flat - mean_r) ** 2).sum())
    return _combine(*acc, n_r, mean_r, M2_r)


def _finalise(acc) -> tuple[float, float]:
    n, mean, M2 = acc
    std = float(np.sqrt(M2 / n)) if n > 1 else 0.0
    return float(mean), std


def compute_stats(root: Path, train_run_ids: list[int]) -> dict[str, dict]:
    """Stream through train zarrs and compute per-(var, level) stats."""
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train dir missing: {train_dir}")

    # Discover schema (vars, levels) from the first store.
    first = xr.open_zarr(str(train_dir / f"run_{train_run_ids[0]:04d}.zarr"),
                         consolidated=None)
    vars_3d = [v for v in ("u", "v", "T") if v in first.data_vars]
    vars_2d = [v for v in ("ps",) if v in first.data_vars]
    level_hpa = first.level.values.astype(float).tolist() if "level" in first.coords else []
    print(f"  schema: 3D vars={vars_3d}  2D vars={vars_2d}  levels={level_hpa}")

    # Build per-field accumulators.
    fields: list[str] = []
    for v in vars_3d:
        for L in level_hpa:
            fields.append(f"{v}_{int(round(L))}hpa")
    fields.extend(vars_2d)

    acc = {f: (0, 0.0, 0.0) for f in fields}
    acc_d = {f: (0, 0.0, 0.0) for f in fields}

    skipped: list[str] = []
    for i, rid in enumerate(train_run_ids):
        path = train_dir / f"run_{rid:04d}.zarr"
        ds = xr.open_zarr(str(path), consolidated=None)

        finite_ok = True
        for v in vars_3d + vars_2d:
            if not np.all(np.isfinite(ds[v].values)):
                finite_ok = False
                break
        if not finite_ok:
            print(f"  skip run_{rid:04d}: non-finite values")
            skipped.append(f"run_{rid:04d}")
            continue

        for v in vars_3d:
            arr = ds[v].values  # (time, level, lat, lon)
            for li, L in enumerate(level_hpa):
                name = f"{v}_{int(round(L))}hpa"
                slab = arr[:, li].astype(np.float64)         # (time, lat, lon)
                acc[name] = _accumulate(slab.ravel(), acc[name])
                d_slab = np.diff(slab, axis=0).astype(np.float64)
                acc_d[name] = _accumulate(d_slab.ravel(), acc_d[name])

        for v in vars_2d:
            arr = ds[v].values.astype(np.float64)            # (time, lat, lon)
            acc[v] = _accumulate(arr.ravel(), acc[v])
            d_arr = np.diff(arr, axis=0).astype(np.float64)
            acc_d[v] = _accumulate(d_arr.ravel(), acc_d[v])

        if (i + 1) % 10 == 0 or (i + 1) == len(train_run_ids):
            print(f"  processed {i + 1}/{len(train_run_ids)} train runs")

    stats: dict[str, dict] = {}
    for name in fields:
        mean, std = _finalise(acc[name])
        mean_d, std_d = _finalise(acc_d[name])
        stats[name] = {
            "mean":       round(mean, 6),
            "std":        round(std, 6),
            "mean_delta": round(mean_d, 6),
            "std_delta":  round(std_d, 6),
        }
    if skipped:
        print(f"  WARNING: skipped {len(skipped)} non-finite runs: {skipped}")
    return stats


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root (contains runs/ from the production array).")
    ap.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST,
                    help="Sweep manifest from generate_sweep_safe.")
    ap.add_argument("--stratify-on", default="tau_drag_days",
                    help="Param to stratify the split on (default: tau_drag_days).")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-stats", action="store_true",
                    help="Skip the stats step (useful if rerun-only for splits).")
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")

    print(f"Finalising {root}")

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
        out_stats.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"  wrote {out_stats}")
    else:
        print("\n[4/4] stats skipped")

    print("\nDone.")


if __name__ == "__main__":
    main()
