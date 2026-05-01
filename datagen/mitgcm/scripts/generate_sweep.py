"""Emit one JSON config per run plus a top-level manifest.

The parameter grid is a tensor product over five axes — three Held-Suarez
physical parameters, one infrastructure parameter, and one random seed.
Runs are indexed ``run_0000 … run_0161`` in row-major order over the tuple
``(tau_drag_days, delta_T_y, delta_theta_z, tau_surf_days, seed)``.

A stable short hash of the physical parameters is stored in each config so
downstream scripts can detect stale caches without re-running simulations.

Run from the repo root::

    uv run --project datagen python -m datagen.mitgcm.scripts.generate_sweep \\
        --out datagen/mitgcm/configs
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

# ── Parameter grid ────────────────────────────────────────────────────────────
# Physical axes follow Held & Suarez (1994) naming and are stored in natural
# user-facing units (days for timescales, K for temperature differences).
# Row-major order: tau_drag varies slowest, seed varies fastest.
#   3 × 3 × 3 × 2 × 3 = 162 total runs.
PARAM_GRID: dict[str, tuple] = {
    "tau_drag_days":  (0.5, 1.0, 2.0),    # Surface drag timescale [days]
    "delta_T_y":      (40.0, 60.0, 80.0), # Equator-to-pole ΔT [K]
    "delta_theta_z":  (5.0, 10.0, 20.0),  # Surface-to-tropopause Δθ [K]
    "tau_surf_days":  (4.0, 8.0),          # Surface cooling timescale [days]
    "seed":           (0, 1, 2),           # IC perturbation seed
}

# Fixed parameters not varied in the sweep.
FIXED_PARAMS: dict = {
    "tau_atm_days": 40.0,   # Free-atmosphere cooling timescale [days]
}


def _hash_params(params: dict) -> str:
    """Stable short hash of a parameter dict (useful as a stale-cache key)."""
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
            # seed is stored as int; all other params as float.
            params[axis] = int(value) if axis == "seed" else float(value)
        params.update(FIXED_PARAMS)
        runs.append(params)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=Path("datagen/mitgcm/configs"),
        help="Output directory for per-run JSON configs.",
    )
    ap.add_argument(
        "--manifest", type=Path, default=None,
        help="Path for the top-level manifest (defaults to <out>/manifest.json).",
    )
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (out_dir / "manifest.json")

    runs = iter_grid()
    manifest_entries: list[dict] = []

    for run_id, params in enumerate(runs):
        entry = {
            "run_id":     run_id,
            "run_name":   f"run_{run_id:04d}",
            "params":     params,
            "param_hash": _hash_params(params),
        }
        manifest_entries.append(entry)
        with open(out_dir / f"run_{run_id:04d}.json", "w") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")

    manifest = {
        "n_runs": len(runs),
        "grid":   {k: list(v) for k, v in PARAM_GRID.items()},
        "fixed":  FIXED_PARAMS,
        "runs":   manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(runs)} configs to {out_dir} and manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
