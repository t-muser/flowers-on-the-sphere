"""Emit one JSON config per ensemble member + a top-level manifest.

The cpl_aim+ocn parameter grid is a tensor product over four axes:

* ``co2_ppm`` — atmospheric CO2 concentration in ppm (4 values; converted
  to AIM's mole-fraction namelist value at render time)
* ``solar_scale`` — multiplier on AIM's area-mean 342 W/m² solar constant
  (3 values)
* ``gm_kappa`` — ocean GM-Redi background diffusivity, m²/s (3 values)
* ``seed`` — atmospheric IC perturbation RNG seed (5 values)

3 × 4 × 3 × 5 = 180 runs total. Row-major iteration: ``co2_ppm`` varies
slowest, ``seed`` varies fastest, so the first 5 runs in the sweep are
the same physics replayed under five different IC perturbations.

Each run is also tagged with a stable 12-char SHA1 hash of its parameter
dict (deterministic across machines / Python versions) so downstream
scripts can detect stale caches without re-running simulations.

Run from the repo root::

    uv run --project datagen python -m datagen.cpl_aim_ocn.scripts.generate_sweep \\
        --out datagen/cpl_aim_ocn/configs
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

# ── Parameter grid ────────────────────────────────────────────────────────────
# Order matters: tensor product is enumerated in row-major order, so
# axes listed earlier vary slower in the run index.
PARAM_GRID: dict[str, tuple] = {
    "co2_ppm":     (280.0, 348.0, 560.0, 1120.0),  # pre-industrial → 4×PI
    "solar_scale": (0.97, 1.00, 1.03),             # ±3% on AIM area-mean SOLC
    "gm_kappa":    (500.0, 1000.0, 2000.0),        # m²/s, GM-Redi κ
    "seed":        (0, 1, 2, 3, 4),                # IC perturbation seed
}

# Per-run derived run-time parameters (same for every run in the sweep —
# they live in the run JSON so the single-run driver doesn't need extra
# CLI flags). Override with `scripts/run.py --spinup-days …` etc.
FIXED_PARAMS: dict = {
    "spinup_days":             30.0,    # ~30 d coupled adjustment
    "data_days":               365.0,   # ~1 yr data collection
    "snapshot_interval_days":   1.0,    # daily snapshots
}


def _hash_params(params: dict) -> str:
    """Stable short hash of a parameter dict (useful as a stale-cache key)."""
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def iter_grid() -> list[dict]:
    """Enumerate the full parameter grid in row-major order.

    Returns a list of ``params`` dicts; each contains every PARAM_GRID
    axis plus every FIXED_PARAMS entry. The seed is stored as ``int``,
    everything else as ``float``.
    """
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=Path("datagen/cpl_aim_ocn/configs"),
        help="Output directory for per-run JSON configs.",
    )
    ap.add_argument(
        "--manifest", type=Path, default=None,
        help="Path for the top-level manifest (defaults to <out>/manifest.json).",
    )
    args = ap.parse_args(argv)

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
