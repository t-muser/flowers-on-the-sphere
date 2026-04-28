"""Single-run driver: load JSON config, run PyClaw simulation, write Zarr.

PyClaw runs in a single process (OMP-only), so this driver does not need
``mpirun``::

    uv run --project datagen python -m datagen.shock_quadrants.scripts.run \\
        --config datagen/shock_quadrants/configs/run_0000.json \\
        --out    $DATASET_ROOT/processed/run_0000.zarr

On solver failure, writes ``<out>.FAILED`` with the exception and params
so the SLURM array can keep going and ``consolidate.py`` can produce a
failure report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from datagen.shock_quadrants.solver import run_simulation
from datagen.resample import regular_lat_grid, regular_lon_grid, write_latlon_zarr


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True,
                    help="Path to the per-run JSON config emitted by generate_sweep.py.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output Zarr store path.")
    ap.add_argument("--nlat", type=int, default=256)
    ap.add_argument("--nlon", type=int, default=512)
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--snapshot-dt", type=float, default=0.015)
    ap.add_argument("--stop-sim-time", type=float, default=1.5)
    ap.add_argument("--cfl-desired", type=float, default=0.45)
    ap.add_argument("--cfl-max", type=float, default=0.9)
    ap.add_argument("--sub-samples", type=int, default=4,
                    help="Sub-cell antialiasing factor (S × S per cell).")
    args = ap.parse_args()

    _setup_logging()
    log = logging.getLogger("run_shock_quadrants")

    with open(args.config) as f:
        config = json.load(f)
    params = config["params"]
    log.info("Loaded config %s (run_id=%s)", args.config, config.get("run_id"))

    out_path: Path = args.out
    failed_marker = out_path.with_suffix(".FAILED")
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_marker, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return 2

    field_names = result["field_names"]
    fields = result["fields"]   # (Nt, 3, Nlat, Nlon) float32
    time_arr = result["time"]
    axis = result["so3_axis_xyz"]
    angle = float(result["so3_angle_rad"])

    lat_target = regular_lat_grid(args.nlat)
    lon_target = regular_lon_grid(args.nlon)

    write_latlon_zarr(
        out_path,
        time_arr=time_arr,
        field_arrays=[np.ascontiguousarray(fields[:, i]) for i in range(len(field_names))],
        field_names=field_names,
        lat_target=lat_target,
        lon_target=lon_target,
        description=(
            "Shallow-water equations on the unit sphere, 4-quadrant Riemann IC "
            "(SO(3)-tilted), Calhoun-Helzel mapped sphere via PyClaw "
            "shallow_sphere_2D Riemann solver."
        ),
        run_id=config.get("run_id"),
        params=params,
        time_units="non-dimensional",
    )

    import xarray as xr
    with xr.open_zarr(str(out_path)) as src:
        ds = src.load()
    ds = ds.assign(
        so3_axis_xyz=(("xyz",), np.asarray(axis, dtype=np.float64)),
    )
    ds.attrs["so3_angle_rad"] = angle
    ds.to_zarr(str(out_path), mode="w", consolidated=True)

    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    rc = main()
    # The Fortran extensions (classic2_sw_sphere, sw_sphere_problem) trigger a
    # double-free in their compiled destructors at process teardown, turning a
    # clean exit into SIGABRT (rc=134).  os._exit bypasses all atexit/destructor
    # chains so the true return code reaches SLURM.
    logging.shutdown()
    os._exit(rc)
