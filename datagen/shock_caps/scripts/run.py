"""Single-run orchestrator: solver subprocess + zarr write.

PyClaw runs in a single process (OMP-only), so this driver does not
need ``mpirun``::

    uv run --project datagen python -m datagen.shock_caps.scripts.run \\
        --config datagen/shock_caps/configs/run_0000.json \\
        --out    $DATASET_ROOT/processed/run_0000.zarr

This module **does not import Clawpack**. The simulation runs in a
subprocess (``_solve.py``) that dumps raw fields to a ``.npz`` next to
the zarr output; this orchestrator then reads the ``.npz`` and writes
the zarr in a fresh, clean heap. The split keeps the Clawpack Fortran
teardown corruption (chunk-header double-write left by the
``classic2_sw_sphere`` / ``sw_sphere_problem`` destructors) from
reaching xarray/zarr's allocation pattern, which previously turned
roughly half of all runs into silent-failure ``corrupted size vs.
prev_size`` aborts.

On solver failure, writes ``<out>.FAILED`` with the exception and
params so the SLURM array can keep going and ``consolidate.py`` can
produce a failure report.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

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
    log = logging.getLogger("run_shock_caps")

    with open(args.config) as f:
        config = json.load(f)
    params = config["params"]
    log.info("Loaded config %s (run_id=%s)", args.config, config.get("run_id"))

    out_path: Path = args.out
    failed_marker = out_path.with_suffix(".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    raw_path = out_path.with_name(out_path.stem + ".raw.npz")
    raw_failed = raw_path.with_suffix(".FAILED")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raw_path.unlink()
    if raw_failed.exists():
        raw_failed.unlink()

    cmd = [
        sys.executable, "-m", "datagen.shock_caps.scripts._solve",
        "--config", str(args.config),
        "--raw-out", str(raw_path),
        "--nlat", str(args.nlat),
        "--nlon", str(args.nlon),
        "--nx", str(args.nx),
        "--ny", str(args.ny),
        "--snapshot-dt", str(args.snapshot_dt),
        "--stop-sim-time", str(args.stop_sim_time),
        "--cfl-desired", str(args.cfl_desired),
        "--cfl-max", str(args.cfl_max),
        "--sub-samples", str(args.sub_samples),
    ]
    log.info("Spawning solver subprocess.")
    rc = subprocess.run(cmd).returncode
    log.info("Solver subprocess returned rc=%d.", rc)

    if rc != 0 or not raw_path.exists():
        if raw_failed.exists():
            failed_marker.write_text(raw_failed.read_text())
            raw_failed.unlink()
        else:
            with open(failed_marker, "w") as f:
                json.dump({
                    "run_id": config.get("run_id"),
                    "config_path": str(args.config),
                    "params": params,
                    "solver_subprocess_rc": rc,
                    "raw_present": raw_path.exists(),
                }, f, indent=2)
                f.write("\n")
        return 2

    with np.load(raw_path) as data:
        fields = data["fields"]                    # (Nt, 3, Nlat, Nlon) float32
        time_arr = data["time"]
        field_names = [str(n) for n in data["field_names"]]

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
            "Shallow-water equations on the unit sphere, random spherical-cap "
            f"Riemann IC (K={params['K']} caps, delta={params['delta']} "
            "velocity scaling, painter's algorithm, plus background), "
            "Calhoun-Helzel mapped sphere via PyClaw shallow_sphere_2D Riemann "
            "solver."
        ),
        run_id=config.get("run_id"),
        params=params,
        time_units="non-dimensional",
    )
    raw_path.unlink()

    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
