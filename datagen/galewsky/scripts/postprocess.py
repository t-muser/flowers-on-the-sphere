"""Resample one run's Dedalus HDF5 output onto a regular (lat, lon) Zarr store.

Invocation (from the repo root so ``datagen.*`` imports resolve)::

    uv run --project datagen python -m datagen.galewsky.scripts.postprocess \\
        --raw $DATA_ROOT/raw/run_0000/ \\
        --out $DATA_ROOT/processed/run_0000.zarr \\
        --config datagen/galewsky/configs/run_0000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datagen.resample import resample_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True,
                    help="Directory of Dedalus HDF5 snapshots (one run).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output Zarr store path.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional config JSON, attached as Zarr attrs.")
    ap.add_argument("--nlat", type=int, default=256)
    ap.add_argument("--nlon", type=int, default=512)
    ap.add_argument("--skip-seconds", type=float, default=4 * 86400.0,
                    help="Drop leading snapshots below this sim time and rebase "
                         "the remaining time axis to 0 (default: 4 days).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    log = logging.getLogger("postprocess_latlon")

    # Skip silently if the raw run is missing (treated as a failed solve upstream).
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

    log.info("Resampling %s -> %s (Nlat=%d, Nlon=%d, skip_seconds=%g)",
             args.raw, args.out, args.nlat, args.nlon, args.skip_seconds)
    resample_run(
        args.raw, args.out,
        Nlat=args.nlat, Nlon=args.nlon,
        time_offset_s=args.skip_seconds,
        params=params, run_id=run_id,
    )
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
