"""Stability sweep over the 32 parameter-box corners at reduced duration.

Generates a corners-only manifest (one config per extreme of the 5-axis
parameter box, 2**5 = 32 runs) and runs each with a shortened spin-up
(30 days) and data-collection window (30 days) rather than the full
200+365-day production run.  Grid resolution (128×64×20) is fixed at
compile time in SIZE.h and cannot be reduced without recompiling MITgcm.

Mirrors the CLI shape of ``generate_sweep.py`` + ``run.py`` so SLURM can
treat preflight and production array jobs the same way.

Two sub-commands (invoke from the repo root so ``datagen.*`` imports
resolve)::

    uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.preflight generate \\
        --out datagen/mitgcm/held_suarez/configs/preflight/

    uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.preflight run \\
        --config datagen/mitgcm/held_suarez/configs/preflight/corner_00.json \\
        --out-dir "$DATA_ROOT/mitgcm/preflight/corner_00/"

On solver failure writes ``<out_dir>.FAILED`` with the exception and params
so the SLURM array keeps running and a failure report can be produced.

The Python driver is serial (no mpi4py); MITgcm manages its own MPI
communicator internally, launched via ``mpirun`` inside ``run_simulation``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
import traceback
from dataclasses import fields
from pathlib import Path

from datagen.mitgcm.held_suarez.scripts.generate_sweep import FIXED_PARAMS, PARAM_GRID
from datagen.mitgcm.held_suarez.solver import RunConfig, run_simulation

# Preflight defaults — short enough to verify solver health without committing
# to a full 565-day production run.
_PREFLIGHT_SPINUP_DAYS: float = 30.0
_PREFLIGHT_DATA_DAYS: float = 30.0


# ─── helpers ─────────────────────────────────────────────────────────────────

def _corner_params() -> list[dict]:
    """All 32 corners: for each axis take the minimum and maximum value."""
    axes = list(PARAM_GRID.keys())
    lo_hi = [(min(vals), max(vals)) for vals in PARAM_GRID.values()]
    runs: list[dict] = []
    for combo in itertools.product(*lo_hi):
        params: dict = {}
        for axis, value in zip(axes, combo):
            params[axis] = int(value) if axis == "seed" else float(value)
        params.update(FIXED_PARAMS)
        runs.append(params)
    return runs


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _hash_params(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


# ─── generate sub-command ────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> int:
    """Emit one JSON config per corner of the parameter box."""
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    corners = _corner_params()
    manifest_entries: list[dict] = []
    for i, params in enumerate(corners):
        entry = {
            "run_id":     i,
            "run_name":   f"corner_{i:02d}",
            "params":     params,
            "param_hash": _hash_params(params),
        }
        manifest_entries.append(entry)
        _write_json(out_dir / f"corner_{i:02d}.json", entry)

    _write_json(
        out_dir / "manifest.json",
        {"n_runs": len(corners), "corners": manifest_entries},
    )
    print(f"Wrote {len(corners)} corner configs to {out_dir}.")
    return 0


# ─── run sub-command ─────────────────────────────────────────────────────────

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run one corner config with preflight (reduced) duration."""
    _setup_logging()
    log = logging.getLogger(__name__)

    with open(args.config, encoding="utf-8") as f:
        entry = json.load(f)
    params = entry["params"]
    log.info("Loaded corner config %s (run_id=%s)", args.config,
             entry.get("run_id"))

    out_dir: Path = args.out_dir
    failed_marker = out_dir.parent / (out_dir.name + ".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    # Collect CLI overrides for RunConfig fields. Asserting membership turns
    # a CLI/RunConfig field-name mismatch into an immediate error rather than
    # silently dropping the override.
    rc_field_names = {f.name for f in fields(RunConfig)}
    overrides: dict = {}
    for dest in ("n_mpi", "spinup_days", "data_days", "snapshot_interval_days",
                 "pressure_hpa", "pressure_levels", "delta_t", "executable"):
        assert dest in rc_field_names, f"Unknown RunConfig field: {dest!r}"
        val = getattr(args, dest, None)
        if val is not None:
            overrides[dest] = tuple(val) if dest == "pressure_levels" else val

    try:
        run_simulation(params, out_dir, **overrides)
    except Exception as exc:
        log.exception("Corner run failed")
        payload = {
            "run_id":      entry.get("run_id"),
            "config_path": str(args.config),
            "params":      params,
            "exception":   repr(exc),
            "traceback":   traceback.format_exc(),
        }
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        _write_json(failed_marker, payload)
        return 1

    log.info("Corner run completed OK.")
    return 0


# ─── entry point ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    ap_gen = sub.add_parser("generate",
                             help="Emit corners-only JSON configs + manifest.")
    ap_gen.add_argument(
        "--out", type=Path,
        default=Path("datagen/mitgcm/held_suarez/configs/preflight"),
        help="Output directory for corner configs.",
    )
    ap_gen.set_defaults(func=cmd_generate)

    # ── run ───────────────────────────────────────────────────────────────────
    ap_run = sub.add_parser("run",
                             help="Run one corner config.")
    ap_run.add_argument(
        "--config", type=Path, required=True,
        help="Path to a corner JSON config emitted by the generate sub-command.",
    )
    ap_run.add_argument(
        "--out-dir", type=Path, required=True,
        help="Root output directory for this corner run.",
    )
    ap_run.add_argument(
        "--executable", type=Path, default=None,
        help="Path to compiled mitgcmuv binary (overrides RunConfig default).",
    )
    ap_run.add_argument(
        "--n-mpi", type=int, default=None, dest="n_mpi",
        help="Number of MPI ranks passed to mpirun.",
    )
    ap_run.add_argument(
        "--spinup-days", type=float, default=_PREFLIGHT_SPINUP_DAYS,
        dest="spinup_days",
        help=f"Spin-up duration [days] (default: {_PREFLIGHT_SPINUP_DAYS}).",
    )
    ap_run.add_argument(
        "--data-days", type=float, default=_PREFLIGHT_DATA_DAYS,
        dest="data_days",
        help=f"Data-collection duration [days] (default: {_PREFLIGHT_DATA_DAYS}).",
    )
    ap_run.add_argument(
        "--snapshot-interval-days", type=float, default=1.0,
        dest="snapshot_interval_days",
        help="Output cadence [days] (default: 1.0).",
    )
    ap_run.add_argument(
        "--pressure-hpa", type=float, default=None, dest="pressure_hpa",
        help="Pressure level to extract for u, v, T [hPa] (single-level mode).",
    )
    ap_run.add_argument(
        "--pressure-levels", type=float, nargs="+", default=None,
        dest="pressure_levels",
        help=(
            "Multiple pressure levels [hPa] for 3-D output. When set, the "
            "writer emits per-variable u/v/T arrays with a `level` axis."
        ),
    )
    ap_run.add_argument(
        "--delta-t", type=float, default=None, dest="delta_t",
        help="Timestep [s] (overrides RunConfig default).",
    )
    ap_run.set_defaults(func=cmd_run)

    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
