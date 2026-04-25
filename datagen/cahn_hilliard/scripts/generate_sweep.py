"""Emit one JSON config per Cahn-Hilliard run plus a top-level manifest.

The parameter grid is an explicit tensor product over five axes; runs are
indexed ``run_0000 … run_NNNN`` in row-major order over the tuple
``(xi, ell, amplitude, omega, seed)``. A stable short hash of the tuple
is stored in every config so downstream code can detect stale caches.

``xi`` is the Krekhov interface width (sets the pattern wavelength); the
bulk control parameter ``epsilon`` is held fixed at 1.0 so all runs sit
in the ordered phase. Forcing harmonic ``m`` is locked to ``ell``
(sectoral mode).

The ``omega`` axis spans three regimes:
  * 0.0      — locked Krekhov pattern (no rotation)
  * 0.005    — slow rotation, near the locking-to-irregular boundary
  * 0.015    — fast rotation, irregular dynamics

Held-fixed quantities are embedded in every per-run config so the solver
only ever reads from ``params``, with no hidden defaults at run time.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

PARAM_GRID: dict[str, tuple[float, ...]] = {
    "xi":        (0.5, 1.0, 2.0),
    "ell":       (6.0, 12.0),
    "amplitude": (0.01, 0.03),
    "omega":     (0.0, 0.005, 0.015),
    "seed":      (0.0, 1.0, 2.0, 3.0),
}

FIXED_PARAMS: dict[str, float] = {
    "epsilon":   1.0,
    "R":         5.0,
    "mean_init": 0.0,
    "variance":  1.0e-2,
    "psi_mean":  0.0,
    "ell_init":  6.0,
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
            if axis in ("seed", "ell"):
                params[axis] = int(value)
            else:
                params[axis] = float(value)
        # Sectoral forcing: m = ell.
        params["m"] = int(params["ell"])
        # Mix in held-fixed scalars; cast ``ell_init`` back to int.
        for k, v in FIXED_PARAMS.items():
            params[k] = int(v) if k == "ell_init" else float(v)
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
