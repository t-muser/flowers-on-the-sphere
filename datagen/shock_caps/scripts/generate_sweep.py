"""Emit one JSON config per run plus a top-level manifest.

Single-axis sweep over ``seed`` (0–499). Each seed determines the K cap
centres, radii, per-cap states, and background state for the cap-Riemann
IC. Total: 500 runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PARAM_GRID: dict[str, tuple[float, ...]] = {
    "seed": tuple(float(s) for s in range(500)),
}


def _hash_params(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def iter_grid() -> list[dict]:
    runs: list[dict] = []
    for s in PARAM_GRID["seed"]:
        runs.append({"seed": int(s)})
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
