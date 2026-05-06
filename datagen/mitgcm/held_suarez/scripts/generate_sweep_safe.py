"""Emit the 72-run *safe* parameter sweep used by the Held-Suarez-3D dataset.

The original 162-run sweep at Δt=45 s produced a 26 % failure rate, all
in the weak-drag / strong-meridional-forcing corner of the box. This
sweep drops the two riskiest extreme axis values
(``delta_T_y = 80`` and ``delta_theta_z = 5``) and is run at Δt = 30 s
to give a comfortable CFL margin on the remaining grid.

Configs are written to ``configs/safe/run_{0000..0071}.json`` plus a
``manifest.json``, mirroring the layout of ``generate_sweep.py`` so the
existing single-run driver (``scripts/run.py``) can consume them
without modification.

Run from the repo root::

    uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.generate_sweep_safe \\
        --out datagen/mitgcm/held_suarez/configs/safe
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from datagen.mitgcm.held_suarez.scripts.generate_sweep import _hash_params

# Safe parameter grid. 3 × 2 × 2 × 2 × 3 = 72 runs.
SAFE_PARAM_GRID: dict[str, tuple] = {
    "tau_drag_days":  (0.5, 1.0, 2.0),
    "delta_T_y":      (40.0, 60.0),    # drops 80 K (strongest jets)
    "delta_theta_z":  (10.0, 20.0),    # drops 5 K (weak stratification)
    "tau_surf_days":  (4.0, 8.0),
    "seed":           (0, 1, 2),
}

FIXED_PARAMS: dict = {
    "tau_atm_days": 40.0,
}


def iter_grid() -> list[dict]:
    """Enumerate the safe grid in row-major order."""
    axes = list(SAFE_PARAM_GRID.keys())
    values = [SAFE_PARAM_GRID[a] for a in axes]
    runs: list[dict] = []
    for combo in itertools.product(*values):
        params: dict = {}
        for axis, value in zip(axes, combo):
            params[axis] = int(value) if axis == "seed" else float(value)
        params.update(FIXED_PARAMS)
        runs.append(params)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=Path("datagen/mitgcm/held_suarez/configs/safe"),
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
        "grid":   {k: list(v) for k, v in SAFE_PARAM_GRID.items()},
        "fixed":  FIXED_PARAMS,
        "runs":   manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(runs)} safe configs to {out_dir} and manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
