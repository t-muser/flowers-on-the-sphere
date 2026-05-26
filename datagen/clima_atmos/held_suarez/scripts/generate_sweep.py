"""Emit one JSON config per run plus a top-level manifest.

The parameter grid is a tensor product over four axes: three physical
Held-Suarez parameters from ClimaParams plus an IC-perturbation seed.
The MITgcm pipeline's 5-axis grid (tau_drag / tau_surf / tau_atm /
delta_T_y / delta_theta_z / seed) is superseded — three of those axes
were hardcoded inside ClimaAtmos.jl's HS forcing tendency and required
a source patch to vary. The active design swaps them for the planet
rotation rate Omega, which is settable through stock ClimaParams TOML
and produces qualitatively distinct jet structures across its sweep
range (multi-jet → Earth-like → wide single-jet).

Runs are indexed ``run_0000 … run_0161`` in row-major order over
``(omega_factor, delta_T_y, delta_theta_z, seed)``. The same order
keeps a deterministic mapping from SLURM array task ID to physical
regime: omega_factor varies slowest, seed fastest.

A stable short hash of the physical parameters is stored in each config
so downstream scripts can detect stale caches without re-running.

Run from the repo root::

    uv run --project datagen python -m datagen.clima_atmos.held_suarez.scripts.generate_sweep \\
        --out datagen/clima_atmos/held_suarez/configs
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

# Earth's angular velocity in rad/s — taken from
# ClimaParams' default parameters.toml (angular_velocity_planet_rotation).
# Multiplying by OMEGA_EARTH yields the absolute Ω we hand to ClimaAtmos.
OMEGA_EARTH: float = 7.2921159e-5

# ── Parameter grid ────────────────────────────────────────────────────────────
# Three physical axes that map directly onto ClimaParams TOML keys, plus
# an IC seed axis. Levels chosen for maximal regime separation within
# numerical-stability limits at he24ze31:
#
#   omega_factor  → multiplier on Earth's Ω. 0.5 / 1.0 / 2.0 takes the
#                   flow through wide single-jet → midlatitude single
#                   jet → multi-jet regimes.
#   delta_T_y     → equator-to-pole equilibrium ΔT [K]. 40 / 60 / 80
#                   matches the HS literature span; sets baroclinic
#                   forcing strength.
#   delta_theta_z → surface-to-tropopause Δθ_z [K]. 5 / 10 / 20 matches
#                   the MITgcm sweep's validated range; sets static
#                   stability and vertical mode structure.
#   seed          → IC perturbation RNG seed; 6 independent realizations
#                   per regime give variance bars on benchmark metrics.
#
#   3 × 3 × 3 × 6 = 162 total runs.
PARAM_GRID: dict[str, tuple] = {
    "omega_factor":   (0.5, 1.0, 2.0),
    "delta_T_y":      (40.0, 60.0, 80.0),
    "delta_theta_z":  (5.0, 10.0, 20.0),
    "seed":           (0, 1, 2, 3, 4, 5),
}

# Fixed parameters not varied in the sweep.
FIXED_PARAMS: dict = {}


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
            params[axis] = int(value) if axis == "seed" else float(value)
        params.update(FIXED_PARAMS)
        runs.append(params)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=Path("datagen/clima_atmos/held_suarez/configs"),
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
    entries: list[dict] = []
    for i, params in enumerate(runs):
        entry = {
            "run_id":     i,
            "run_name":   f"run_{i:04d}",
            "params":     params,
            "param_hash": _hash_params(params),
        }
        entries.append(entry)
        with open(out_dir / f"run_{i:04d}.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")

    manifest = {
        "n_runs": len(runs),
        "grid":   {k: list(v) for k, v in PARAM_GRID.items()},
        "fixed":  FIXED_PARAMS,
        "omega_earth_rad_s": OMEGA_EARTH,
        "runs":   entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(runs)} run configs + manifest to {out_dir}.")


if __name__ == "__main__":
    main()
