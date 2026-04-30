"""Emit per-run JSON configs + manifest for the Mickelin GNS ensemble.

The parameter grid is a tensor product over three axes; runs are indexed
``run_0000 … run_0479`` in row-major order over the tuple
``(r_over_lambda, kappa_lambda, seed)``. A stable short hash of the tuple is
stored in every config so downstream code can detect stale caches.

Fixed non-dimensional scales: ``R = 1``, ``τ = 1``. Derived per run:
``Λ = R / r_over_lambda`` and ``κ = kappa_lambda / Λ``. The Mickelin A-phase
regime requires ``R⁻¹ < κ < Λ⁻¹`` — always satisfied by the default grid
(``κ·R = kappa_lambda · r_over_lambda ≥ 1.2`` and
``κΛ = kappa_lambda ≤ 0.7``).

Run from anywhere — by default the script writes relative to the repo root
and does NOT touch the data tree::

    uv run --project datagen python -m datagen.mickelin.scripts.generate_sweep

This also emits a small preflight set (8 extreme points of the
``r_over_lambda × kappa_lambda`` box, ``seed = 0``) into
``mickelin/configs/preflight/``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


# Parameter grid (4 · 3 · 40 = 480 runs). Keep axes in this order because the
# run index is computed row-major over them.
PARAM_GRID_MICKELIN: dict[str, tuple[float, ...]] = {
    "r_over_lambda": (2.0, 4.0, 7.0, 10.0),
    "kappa_lambda": (0.2, 0.4, 0.7, 1.0, 1.4, 1.8),
    "seed": tuple(float(s) for s in range(20)),
}


# Fixed non-dimensional scales shared across the whole sweep.
R_SPHERE = 1.0
TAU_TIME = 1.0


def _hash_params(params: dict) -> str:
    """Stable short hash of the parameter tuple (useful as a cache key)."""
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def _make_params(r_over_lambda: float, kappa_lambda: float, seed: int) -> dict:
    """Derive the full Mickelin parameter dict from the three control axes."""
    Lambda = R_SPHERE / float(r_over_lambda)
    kappa = float(kappa_lambda) / Lambda
    return {
        "R": R_SPHERE,
        "tau": TAU_TIME,
        "Lambda": Lambda,
        "kappa": kappa,
        "seed": int(seed),
        "r_over_lambda": float(r_over_lambda),
        "kappa_lambda": float(kappa_lambda),
    }


def iter_grid() -> list[dict]:
    """Enumerate the full 480-run parameter grid in row-major order."""
    axes = list(PARAM_GRID_MICKELIN.keys())
    values = [PARAM_GRID_MICKELIN[a] for a in axes]
    runs: list[dict] = []
    for combo in itertools.product(*values):
        r_over_lambda, kappa_lambda, seed = combo
        runs.append(_make_params(r_over_lambda, kappa_lambda, int(seed)))
    return runs


def iter_preflight() -> list[dict]:
    """Low-res preflight set: 4 ``r_over_lambda`` × 2 extreme ``kappa_lambda``.

    Eight points at ``seed = 0`` sample the extremes of the active-band
    parameter plane, so a preflight pass confirms numerical stability
    across the whole sweep before launching the 480-run array.
    """
    r_over_lambdas = PARAM_GRID_MICKELIN["r_over_lambda"]
    kappa_lambdas = (
        min(PARAM_GRID_MICKELIN["kappa_lambda"]),
        max(PARAM_GRID_MICKELIN["kappa_lambda"]),
    )
    runs: list[dict] = []
    for r_over_lambda in r_over_lambdas:
        for kappa_lambda in kappa_lambdas:
            runs.append(_make_params(r_over_lambda, kappa_lambda, seed=0))
    return runs


def _write_configs(
    runs: list[dict],
    out_dir: Path,
    manifest_path: Path,
    name_pattern: str,
    grid: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    for run_id, params in enumerate(runs):
        run_name = name_pattern.format(run_id=run_id)
        entry = {
            "run_id": run_id,
            "run_name": run_name,
            "params": params,
            "param_hash": _hash_params(params),
        }
        manifest_entries.append(entry)
        with open(out_dir / f"{run_name}.json", "w") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")

    manifest = {"n_runs": len(runs), "runs": manifest_entries}
    if grid is not None:
        manifest["grid"] = {k: list(v) for k, v in grid.items()}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("datagen/mickelin/configs"),
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
    manifest_path = args.manifest or (out_dir / "manifest.json")

    sweep_runs = iter_grid()
    _write_configs(
        sweep_runs,
        out_dir=out_dir,
        manifest_path=manifest_path,
        name_pattern="run_{run_id:04d}",
        grid=PARAM_GRID_MICKELIN,
    )
    print(
        f"Wrote {len(sweep_runs)} sweep configs to {out_dir} "
        f"and manifest to {manifest_path}."
    )

    preflight_runs = iter_preflight()
    preflight_dir = out_dir / "preflight"
    _write_configs(
        preflight_runs,
        out_dir=preflight_dir,
        manifest_path=preflight_dir / "manifest.json",
        name_pattern="corner_{run_id:02d}",
    )
    print(f"Wrote {len(preflight_runs)} preflight configs to {preflight_dir}.")


if __name__ == "__main__":
    main()
