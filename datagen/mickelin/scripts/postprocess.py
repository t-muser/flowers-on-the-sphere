"""Resample one Mickelin run's Dedalus HDF5 output onto a regular (lat, lon) Zarr.

Mirrors ``galewsky/scripts/postprocess.py`` but uses the Mickelin field set
(``vorticity`` only) and sim-time units of ``τ`` (with ``τ = 1``).

Invocation (from the repo root so ``datagen.*`` imports resolve)::

    uv run --project datagen python -m datagen.mickelin.scripts.postprocess \\
        --raw $DATASET_ROOT/raw/run_0000/ \\
        --out $DATASET_ROOT/processed/run_0000.zarr \\
        --config datagen/mickelin/configs/run_0000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datagen.resample import MICKELIN_FIELD_SPECS, resample_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True,
                    help="Directory of Dedalus HDF5 snapshots (one run).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output Zarr store path.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional config JSON, attached as Zarr attrs.")
    ap.add_argument("--nlat", type=int, default=128)
    ap.add_argument("--nlon", type=int, default=256)
    ap.add_argument("--skip-tau", type=float, default=30.0,
                    help="Discard leading snapshots below this sim time (in τ) "
                         "and rebase the kept window to t=0 (default: 30 τ).")
    ap.add_argument("--skip-tol", type=float, default=1.0e-3,
                    help="Tolerance below the skip-tau cutoff that still counts "
                         "as in-window, accommodating CFL-induced timestep jitter.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    log = logging.getLogger("postprocess_mickelin")

    failed_marker = args.raw.with_suffix(".FAILED")
    if failed_marker.exists():
        log.warning("Run is marked FAILED (%s); skipping postprocess.", failed_marker)
        return 0

    params = None
    run_id = None
    if args.config is not None:
        with open(args.config) as f:
            entry = json.load(f)
        params = entry.get("params")
        run_id = entry.get("run_id")

    log.info("Resampling %s -> %s (Nlat=%d, Nlon=%d, skip_tau=%g)",
             args.raw, args.out, args.nlat, args.nlon, args.skip_tau)
    resample_run(
        args.raw, args.out,
        Nlat=args.nlat, Nlon=args.nlon,
        time_offset_s=args.skip_tau,
        time_offset_tol=args.skip_tol,
        params=params, run_id=run_id,
        field_specs=MICKELIN_FIELD_SPECS,
        description="Mickelin GNS vorticity snapshot, resampled to regular (lat, lon).",
    )
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
