"""Single-run driver for the Mickelin GNS ensemble.

Designed to be launched under MPI from SLURM, from the repo root (so that
``datagen.*`` imports resolve against the filesystem)::

    mpirun -n 16 uv run --project datagen python -m datagen.mickelin.scripts.run \\
        --config datagen/mickelin/configs/run_0000.json \\
        --out-dir $DATA_ROOT/raw/run_0000/

On solver failure, writes ``<out_dir>.FAILED`` with the exception and params
so that the SLURM array keeps going and ``consolidate.py`` can produce a
failure report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

from datagen.mickelin.solver import run_simulation

from mpi4py import MPI

def _setup_logging() -> None:
    # Only configure stdout for the root MPI rank
    if MPI.COMM_WORLD.rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True,
                    help="Path to the per-run JSON config.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory for the Dedalus HDF5 snapshots.")
    ap.add_argument("--nphi", type=int, default=256)
    ap.add_argument("--ntheta", type=int, default=128)
    ap.add_argument("--stop-sim-time", type=float, default=65.0,
                    help="Stop simulation at this many τ (default: 65·τ).")
    ap.add_argument("--snapshot-dt", type=float, default=0.2,
                    help="Snapshot cadence in τ units (default: τ/5).")
    ap.add_argument("--initial-dt", type=float, default=5.0e-3)
    ap.add_argument("--max-dt", type=float, default=5.0e-2)
    ap.add_argument("--ell-init", type=int, default=None,
                    help="Override the default ell_init derived from the unstable band.")
    ap.add_argument("--epsilon", type=float, default=1.0e-3)
    args = ap.parse_args()

    _setup_logging()
    log = logging.getLogger("run_mickelin")

    with open(args.config) as f:
        config = json.load(f)
    params = config["params"]
    log.info("Loaded config %s (run_id=%s)", args.config, config.get("run_id"))

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_marker = out_dir.with_suffix(".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    try:
        run_simulation(
            params,
            out_dir,
            snapshot_dt=args.snapshot_dt,
            stop_sim_time=args.stop_sim_time,
            Nphi=args.nphi,
            Ntheta=args.ntheta,
            initial_dt=args.initial_dt,
            max_dt=args.max_dt,
            ell_init=args.ell_init,
            epsilon=args.epsilon,
        )
    except Exception as exc:
        log.exception("Simulation failed")
        payload = {
            "run_id": config.get("run_id"),
            "config_path": str(args.config),
            "params": params,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
        with open(failed_marker, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return 2

    log.info("Run completed OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
