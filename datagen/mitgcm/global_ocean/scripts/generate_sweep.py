"""Emit one JSON config per MITgcm global-ocean sweep run.

The grid varies five physically meaningful ocean controls around the
``tutorial_global_oce_latlon`` defaults:

* ``gm_background_k``: GM/Redi mesoscale eddy diffusivity.
* ``visc_ah``: horizontal Laplacian viscosity.
* ``diff_kr``: vertical diffusivity for temperature and salinity.
* ``tau_theta_relax_days``: surface temperature restoring timescale.
* ``tau_salt_relax_days``: surface salinity restoring timescale.

Run from the repository root::

    uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.generate_sweep \\
        --out datagen/mitgcm/global_ocean/configs
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

# Row-major order: first axis varies slowest, last axis varies fastest.
# 3**5 = 243 runs, comparable to the Held-Suarez 162-run grid while exploring
# weak/standard/strong mixing and weak/standard/strong air-sea restoring.
PARAM_GRID: dict[str, tuple[float, ...]] = {
    "gm_background_k":      (250.0, 1000.0, 2500.0),
    "visc_ah":              (2.0e5, 5.0e5, 1.0e6),
    "diff_kr":              (1.0e-5, 3.0e-5, 1.0e-4),
    "tau_theta_relax_days": (30.0, 60.0, 120.0),
    "tau_salt_relax_days":  (90.0, 180.0, 360.0),
}

FIXED_PARAMS: dict[str, float] = {}


def _hash_params(params: dict) -> str:
    """Stable short hash of a parameter dict."""
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def iter_grid() -> list[dict]:
    """Enumerate the full global-ocean tensor-product grid."""
    axes = list(PARAM_GRID.keys())
    values = [PARAM_GRID[a] for a in axes]
    runs: list[dict] = []
    for combo in itertools.product(*values):
        params = {axis: float(value) for axis, value in zip(axes, combo)}
        params.update(FIXED_PARAMS)
        runs.append(params)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("datagen/mitgcm/global_ocean/configs"),
        help="Output directory for per-run JSON configs.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
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
            "run_id": run_id,
            "run_name": f"run_{run_id:04d}",
            "params": params,
            "param_hash": _hash_params(params),
        }
        manifest_entries.append(entry)
        with open(out_dir / f"run_{run_id:04d}.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")

    manifest = {
        "n_runs": len(runs),
        "grid": {k: list(v) for k, v in PARAM_GRID.items()},
        "fixed": FIXED_PARAMS,
        "runs": manifest_entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(runs)} configs to {out_dir} and manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
