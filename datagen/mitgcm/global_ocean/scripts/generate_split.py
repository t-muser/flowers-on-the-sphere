"""Emit a train/val/test split of run IDs for the global-ocean sweep.

Reads the manifest emitted by ``generate_sweep.py`` and writes
``splits.json`` next to it. Default strategy is stratified on
``gm_background_k`` so each split sees all three GM/Redi eddy regimes
in the target proportions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _assign_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1; got {ratios} → {sum(ratios)}")
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 0:
        raise ValueError(f"Degenerate split for n={n}, ratios={ratios}")
    return n_train, n_val, n_test


def _random_split(run_ids, ratios, rng):
    shuffled = rng.permutation(run_ids).tolist()
    n_train, n_val, _ = _assign_sizes(len(run_ids), ratios)
    return {
        "train": sorted(shuffled[:n_train]),
        "val":   sorted(shuffled[n_train : n_train + n_val]),
        "test":  sorted(shuffled[n_train + n_val:]),
    }


def _stratified_split(manifest_runs, ratios, stratify_on, rng):
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("datagen/mitgcm/global_ocean/configs/manifest.json"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output splits.json path (defaults to <manifest dir>/splits.json).")
    ap.add_argument("--strategy", choices=("stratified", "random"), default="stratified")
    ap.add_argument("--stratify-on", default="gm_background_k")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratios = (args.train, args.val, args.test)
    with open(args.manifest) as f:
        manifest = json.load(f)
    runs = manifest["runs"]

    rng = np.random.default_rng(args.seed)
    if args.strategy == "stratified":
        splits = _stratified_split(runs, ratios, args.stratify_on, rng)
    else:
        splits = _random_split([int(r["run_id"]) for r in runs], ratios, rng)

    out_path = args.out or (args.manifest.parent / "splits.json")
    payload = {
        "seed": args.seed,
        "strategy": args.strategy,
        "stratify_on": args.stratify_on if args.strategy == "stratified" else None,
        "split_ratios": {"train": args.train, "val": args.val, "test": args.test},
        "n_runs": len(runs),
        "counts": {k: len(v) for k, v in splits.items()},
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")
    print(f"  strategy: {payload['strategy']}  seed: {payload['seed']}")
    print(f"  counts:   {payload['counts']}")


if __name__ == "__main__":
    main()
