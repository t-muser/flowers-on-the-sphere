"""Emit a train/val/test split of Mickelin run IDs.

Reads ``manifest.json`` (produced by ``generate_sweep.py``) and writes a
``splits.json`` next to it with ``train`` / ``val`` / ``test`` lists of integer
run IDs. The split is stratified jointly on ``(r_over_lambda, kappa_lambda)``
so each split sees every regime cell in the target proportions.

Use ``--exclude`` to drop runs whose parameters match given ``key=value``
pairs (all listed pairs must match for a run to be excluded). Excluded
run IDs are simply absent from the output, leaving gaps in the run-ID
sequence — that is intentional and downstream code should not assume a
contiguous range.

Example: drop the sub-A-phase ``(r=2, k=0.4)`` row that produces bursty,
low-dimensional dynamics rather than chained turbulence::

    uv run --project datagen python -m datagen.mickelin.scripts.generate_split \\
        --exclude r_over_lambda=2.0,kappa_lambda=0.4
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


def _parse_exclude(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for pair in spec.split(","):
        key, _, val = pair.partition("=")
        if not key or not val:
            raise ValueError(f"Bad --exclude pair {pair!r}; expected key=value.")
        out[key.strip()] = float(val)
    return out


def _matches_exclude(params: dict, exclude: dict[str, float]) -> bool:
    return all(
        math.isclose(float(params[k]), v, abs_tol=1e-9)
        for k, v in exclude.items()
    )


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
                    default=Path("datagen/mickelin/configs/manifest.json"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output splits.json path (defaults to <manifest dir>/splits.json).")
    ap.add_argument("--stratify-on", default="r_over_lambda,kappa_lambda",
                    help="Comma-separated parameter keys to stratify jointly on.")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated key=value pairs; runs matching ALL pairs "
                         "are dropped from the split.")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratios = (args.train, args.val, args.test)
    stratify_keys = [s.strip() for s in args.stratify_on.split(",") if s.strip()]
    exclude = _parse_exclude(args.exclude)

    with open(args.manifest) as f:
        manifest = json.load(f)
    all_runs = manifest["runs"]
    kept_runs = [r for r in all_runs if not _matches_exclude(r["params"], exclude)]
    excluded_ids = sorted(int(r["run_id"]) for r in all_runs
                          if _matches_exclude(r["params"], exclude))

    rng = np.random.default_rng(args.seed)
    splits = _stratified_split(kept_runs, ratios, stratify_keys, rng)

    out_path = args.out or (args.manifest.parent / "splits.json")
    payload = {
        "seed": args.seed,
        "strategy": "stratified",
        "stratify_on": stratify_keys,
        "exclude": exclude or None,
        "excluded_run_ids": excluded_ids,
        "split_ratios": {"train": args.train, "val": args.val, "test": args.test},
        "n_runs_total": len(all_runs),
        "n_runs_split": len(kept_runs),
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
    print(f"  excluded:    {len(excluded_ids)} runs (params match {exclude or '∅'})")
    print(f"  counts:      {payload['counts']}  (of {payload['n_runs_split']} kept)")


if __name__ == "__main__":
    main()
