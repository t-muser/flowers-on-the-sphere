"""Single-run driver: load a JSON config, run one Dedalus simulation.

Designed to be launched under MPI from SLURM, from the repo root (so that
``datagen.*`` imports resolve against the filesystem)::

    mpirun -n 16 uv run --project datagen python -m datagen.galewsky.scripts.run \\
        --config datagen/galewsky/configs/run_0000.json \\
        --out-dir $DATA_ROOT/raw/run_0000/

On solver failure, writes ``<out_dir>.FAILED`` with the exception and params
so that the SLURM array keeps going and ``consolidate.py`` can produce a
failure report. Only rank 0 writes the marker.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import fields
from pathlib import Path

from mpi4py import MPI

from datagen.galewsky.solver import RunConfig, run_simulation


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True,
                    help="Path to the per-run JSON config emitted by generate_sweep.py.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory for the Dedalus HDF5 snapshots.")
    ap.add_argument("--nphi", type=int, default=512, dest="Nphi")
    ap.add_argument("--ntheta", type=int, default=256, dest="Ntheta")
    ap.add_argument("--stop-sim-time", type=float, default=16 * 86400.0,
                    dest="stop_sim_time")
    ap.add_argument("--snapshot-dt", type=float, default=3600.0,
                    dest="snapshot_dt")
    ap.add_argument("--initial-dt", type=float, default=120.0,
                    dest="initial_dt")
    ap.add_argument("--max-dt", type=float, default=600.0,
                    dest="max_dt")
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

    # Clear any previous FAILED marker from a prior attempt (rank 0 only).
    if MPI.COMM_WORLD.rank == 0 and failed_marker.exists():
        failed_marker.unlink()
    MPI.COMM_WORLD.Barrier()

    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(RunConfig)
        if hasattr(args, f.name)
    }

    try:
        run_simulation(params, out_dir, **overrides)
    except Exception as exc:
        log.exception("Simulation failed")
        if MPI.COMM_WORLD.rank == 0:
            payload = {
                "run_id": config.get("run_id"),
                "config_path": str(args.config),
                "params": params,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
            }
            with open(failed_marker, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
        return 1

    log.info("Run completed OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
