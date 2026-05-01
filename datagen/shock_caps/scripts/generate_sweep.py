"""Emit one JSON config per run plus a top-level manifest.

Three-axis grid over ``(K, delta, seed)``: 5 cap counts × 5 velocity-
strength values × 20 seeds = 500 runs. Iteration nests
``K → delta → seed`` so each ``(K, delta)`` block is contiguous in the
SLURM array, which keeps any per-block failure pattern easy to spot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PARAM_GRID: dict[str, tuple] = {
    "K": (1, 2, 4, 8, 16),
    "delta": (0.0, 0.25, 0.5, 0.75, 1.0),
    "seed": tuple(range(20)),
}


def _hash_params(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def iter_grid() -> list[dict]:
    runs: list[dict] = []
    for K in PARAM_GRID["K"]:
        for delta in PARAM_GRID["delta"]:
            for seed in PARAM_GRID["seed"]:
                runs.append({
                    "seed": int(seed),
                    "delta": float(delta),
                    "K": int(K),
                })
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("datagen/shock_caps/configs"),
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
        "runs": manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(runs)} configs to {out_dir} and manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
