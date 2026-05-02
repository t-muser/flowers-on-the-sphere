"""Emit a train/val/test split of shock_caps run IDs.

Reads ``manifest.json`` (produced by ``generate_sweep.py``) and writes a
``splits.json`` next to it with ``train`` / ``val`` / ``test`` lists of
integer run IDs. The split is stratified jointly on ``(K, delta)`` so
each split sees every (cap-count, velocity-strength) cell of the grid
in the target proportions.

With the default 80 / 10 / 10 ratios and 20 seeds per cell this gives
**16 / 2 / 2** runs per cell, **400 / 50 / 50** runs total.

The split is deterministic for a fixed ``--seed``. Re-running this
script overwrites ``splits.json`` in place; the seed is recorded inside
the file so splits are trivially reproducible.

Usage::

    uv run --project datagen python -m datagen.shock_caps.scripts.generate_split
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


def _stratified_split(manifest_runs, ratios, stratify_keys, rng):
    """Group by the tuple of values at ``stratify_keys`` and split each bucket."""
    by_stratum: dict[tuple, list[int]] = {}
    for entry in manifest_runs:
        key = tuple(float(entry["params"][k]) for k in stratify_keys)
        by_stratum.setdefault(key, []).append(int(entry["run_id"]))

    splits = {"train": [], "val": [], "test": []}
    for ids in by_stratum.values():
        shuffled = rng.permutation(ids).tolist()
        n_train, n_val, _ = _assign_sizes(len(shuffled), ratios)
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])
    return {k: sorted(v) for k, v in splits.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("datagen/shock_caps/configs/manifest.json"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output splits.json path (defaults to <manifest dir>/splits.json).")
    ap.add_argument("--stratify-on", default="K,delta",
                    help="Comma-separated parameter keys to stratify jointly on.")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratios = (args.train, args.val, args.test)
    stratify_keys = [s.strip() for s in args.stratify_on.split(",") if s.strip()]

    with open(args.manifest) as f:
        manifest = json.load(f)
    runs = manifest["runs"]

    rng = np.random.default_rng(args.seed)
    splits = _stratified_split(runs, ratios, stratify_keys, rng)

    out_path = args.out or (args.manifest.parent / "splits.json")
    payload = {
        "seed": args.seed,
        "strategy": "stratified",
        "stratify_on": stratify_keys,
        "split_ratios": {"train": args.train, "val": args.val, "test": args.test},
        "n_runs_total": len(runs),
        "counts": {k: len(v) for k, v in splits.items()},
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path}")
    print(f"  stratify_on: {stratify_keys}")
    print(f"  counts:      {payload['counts']}  (of {payload['n_runs_total']} runs)")


if __name__ == "__main__":
    main()
