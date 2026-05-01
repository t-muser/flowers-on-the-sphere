"""Emit one JSON config per run plus a top-level manifest.

The parameter grid is an explicit tensor product over five axes; runs are
indexed ``run_0000 … run_NNNN`` in row-major order over the tuple
``(u_max, lat_center, h_hat, H, seed)``. A stable short hash of the tuple
is stored in every config so downstream code can detect stale caches.

The previous ``lon_c`` axis is gone: each run's per-trajectory SO(3) tilt
(keyed on ``run_id`` at postprocess time) supplies the rotational diversity
that lon_c used to provide, plus a non-grid-aligned rotation axis so a
model can't exploit grid alignment as a shortcut. The ``seed`` axis is
just a multiplicity index — it produces 6 distinct ``run_id``s per physics
combination, and so 6 distinct rotations.

Run from anywhere — by default the script writes relative to the repo root
(``configs/`` directory) and does NOT touch the data tree.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

# Parameter grid (5 · 4 · 2 · 4 · 6 = 960 runs). Keep axes in this order
# because the run index is computed row-major over them. The SO(3) tilt
# is keyed on ``run_id`` at postprocess time, so ``seed`` is just a
# rotation-multiplicity index — it gives every physics combination 6
# distinct ``run_id``s, hence 6 distinct rotations.
PARAM_GRID: dict[str, tuple[float, ...]] = {
    "u_max": (60.0, 70.0, 80.0, 90.0, 100.0),
    "lat_center": (30.0, 40.0, 50.0, 60.0),
    "h_hat": (60.0, 240.0),
    "H": (8000.0, 10000.0, 12000.0, 14000.0),
    "seed": tuple(float(s) for s in range(6)),
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
            params[axis] = int(value) if axis == "seed" else float(value)
        runs.append(params)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("datagen/galewsky/configs"),
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
