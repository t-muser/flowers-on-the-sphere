"""Emit one JSON config per Cahn-Hilliard run plus a top-level manifest.

The parameter grid is an explicit tensor product over three axes; runs are
indexed ``run_0000 … run_NNNN`` in row-major order over the tuple
``(epsilon, mean_init, seed)``. A stable short hash of the tuple is stored
in every config so downstream code can detect stale caches.

The held-fixed quantities (``D``, ``a``, ``variance``, ``radius``) are
embedded in every per-run config so the solver only ever reads from
``params``, with no hidden defaults at run time.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

PARAM_GRID: dict[str, tuple[float, ...]] = {
    "epsilon":   (0.5, 1.0, 1.5, 2.0),
    "mean_init": (0.35, 0.50, 0.65),
    "variance":  (0.001, 0.005, 0.01, 0.05),
    "radius":    (5.0, 7.5, 10.0),
    "seed":      (0, 1, 2, 3),
}

FIXED_PARAMS: dict[str, float] = {
    "D": 1.0,
    "a": 1.0,
}


def _hash_params(params: dict) -> str:
    """Stable short hash of the parameter tuple (useful as a cache key)."""
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def iter_grid() -> list[dict]:
    """Enumerate the full parameter grid in row-major order."""
    axes = list(PARAM_GRID.keys())
    values = [PARAM_GRID[a] for a in axes]
    runs: list[dict] = []
    for combo in itertools.product(*values):
        params: dict = {}
        for axis, value in zip(axes, combo):
            # Keep ``seed`` as an int; everything else is a float coordinate.
            params[axis] = int(value) if axis == "seed" else float(value)
        params.update(FIXED_PARAMS)
        runs.append(params)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("datagen/cahn_hilliard/configs"),
                    help="Output directory for per-run JSON configs.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Path for the top-level manifest (defaults to <out>/manifest.json).")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (out_dir / "manifest.json")

    runs = iter_grid()
    manifest_entries: list[dict] = []
    for run_id, params in enumerate(runs):
        entry = {
            "run_id": run_id,
            "run_name": f"run_{run_id:04d}",
            "params": params,
            "param_hash": _hash_params(params),
        }
        manifest_entries.append(entry)

        with open(out_dir / f"run_{run_id:04d}.json", "w") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")

    manifest = {
        "n_runs": len(runs),
        "grid": {k: list(v) for k, v in PARAM_GRID.items()},
        "fixed": FIXED_PARAMS,
        "runs": manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(runs)} configs to {out_dir} and manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
