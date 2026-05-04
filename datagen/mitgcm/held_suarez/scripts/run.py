"""Single-run driver: load a JSON config and run one MITgcm simulation.

Designed to be launched from the repo root (so that ``datagen.*`` imports
resolve against the filesystem), typically from a SLURM array job::

    uv run --project datagen python -m datagen.mitgcm.held_suarez.scripts.run \\
        --config datagen/mitgcm/held_suarez/configs/run_0000.json \\
        --out-dir $DATA_ROOT/mitgcm/runs/run_0000/

The Python driver process is serial; MITgcm internally manages its own MPI
communicator. ``n_mpi`` controls how many ranks the driver passes to
``mpirun``.

On solver failure, writes ``<out_dir>.FAILED`` with the exception and params
so that the SLURM array keeps running and ``consolidate.py`` can produce a
failure report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import fields
from pathlib import Path

from datagen.mitgcm.held_suarez.solver import RunConfig, run_simulation


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, required=True,
        help="Path to the per-run JSON config emitted by generate_sweep.py.",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Root output directory for this run.",
    )
    ap.add_argument(
        "--executable", type=Path, default=None,
        help="Path to the compiled mitgcmuv binary (overrides RunConfig default).",
    )
    ap.add_argument(
        "--n-mpi", type=int, default=None, dest="n_mpi",
        help="Number of MPI ranks (overrides RunConfig default).",
    )
    ap.add_argument(
        "--spinup-days", type=float, default=None, dest="spinup_days",
        help="Spin-up duration in days (overrides RunConfig default).",
    )
    ap.add_argument(
        "--data-days", type=float, default=None, dest="data_days",
        help="Data-collection duration in days (overrides RunConfig default).",
    )
    ap.add_argument(
        "--snapshot-interval-days", type=float, default=None,
        dest="snapshot_interval_days",
        help="Output cadence in days (overrides RunConfig default).",
    )
    ap.add_argument(
        "--pressure-hpa", type=float, default=None, dest="pressure_hpa",
        help="Pressure level to extract for u, v, T [hPa].",
    )
    ap.add_argument(
        "--delta-t", type=float, default=None, dest="delta_t",
        help="Timestep [s] (overrides RunConfig default).",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging()
    log = logging.getLogger(__name__)

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    params = config["params"]
    log.info("Loaded config %s (run_id=%s)", args.config, config.get("run_id"))

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
                 "pressure_hpa", "delta_t", "executable"):
        assert dest in rc_field_names, f"Unknown RunConfig field: {dest!r}"
        val = getattr(args, dest, None)
        if val is not None:
            overrides[dest] = val

    try:
        run_simulation(params, out_dir, **overrides)
    except Exception as exc:
        log.exception("Simulation failed")
        payload = {
            "run_id":      config.get("run_id"),
            "config_path": str(args.config),
            "params":      params,
            "exception":   repr(exc),
            "traceback":   traceback.format_exc(),
        }
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_marker, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return 1

    log.info("Run completed OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
