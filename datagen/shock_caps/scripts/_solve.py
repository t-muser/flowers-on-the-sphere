"""Solver-only entrypoint: write raw simulation fields to a ``.npz`` file.

This is the inner half of the two-stage write: ``run.py`` invokes this
in a subprocess so the Clawpack Fortran extensions never share a heap
with the xarray/zarr write phase. The Fortran teardown corrupts heap
metadata in ways that have been observed to fire mid-process during
the post-simulation ``np.stack`` inside ``write_latlon_zarr``; isolating
the solver in its own process keeps that corruption out of the writer.

The solver process exits via ``os._exit`` to bypass Python destructor
chains (the Fortran extensions also double-free at clean teardown,
turning ``sys.exit`` into ``SIGABRT``). The orchestrator only needs the
``.npz`` to be on disk by the time we ``os._exit(0)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from pathlib import Path

import numpy as np

from datagen.shock_caps.solver import run_simulation


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--raw-out", type=Path, required=True,
                    help="Output .npz path for raw simulation fields.")
    ap.add_argument("--nlat", type=int, default=256)
    ap.add_argument("--nlon", type=int, default=512)
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--snapshot-dt", type=float, default=0.015)
    ap.add_argument("--stop-sim-time", type=float, default=1.5)
    ap.add_argument("--cfl-desired", type=float, default=0.45)
    ap.add_argument("--cfl-max", type=float, default=0.9)
    ap.add_argument("--sub-samples", type=int, default=4)
    args = ap.parse_args()

    _setup_logging()
    log = logging.getLogger("solve_shock_caps")

    with open(args.config) as f:
        config = json.load(f)
    params = config["params"]
    log.info("Loaded config %s (run_id=%s)", args.config, config.get("run_id"))

    raw_out: Path = args.raw_out
    failed_marker = raw_out.with_suffix(".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    try:
        result = run_simulation(
            params,
            Nlat=args.nlat,
            Nlon=args.nlon,
            Nx=args.nx,
            Ny=args.ny,
            snapshot_dt=args.snapshot_dt,
            stop_sim_time=args.stop_sim_time,
            cfl_desired=args.cfl_desired,
            cfl_max=args.cfl_max,
            sub_samples=args.sub_samples,
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
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_marker, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return 2

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        raw_out,
        fields=result["fields"],
        time=result["time"].astype(np.float64),
        field_names=np.array(result["field_names"]),
    )
    log.info("Wrote raw %s", raw_out)
    return 0


if __name__ == "__main__":
    rc = main()
    # See module docstring: Fortran destructors at process teardown
    # double-free, so go through os._exit to preserve rc.
    logging.shutdown()
    os._exit(rc)
