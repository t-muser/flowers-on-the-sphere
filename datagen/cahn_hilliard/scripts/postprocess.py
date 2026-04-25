"""Resample one Cahn-Hilliard run's Dedalus HDF5 output to a regular (lat, lon) Zarr.

Mirrors ``mickelin/scripts/postprocess.py`` (structured Gauss-Legendre →
equispaced colatitude path) with the CH field set ``{psi, forcing}``::

    uv run --project datagen python -m datagen.cahn_hilliard.scripts.postprocess \\
        --raw $DATA_ROOT/raw/run_0000/ \\
        --out $DATA_ROOT/processed/run_0000.zarr \\
        --config datagen/cahn_hilliard/configs/run_0000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datagen.resample import resample_run


CH_FIELD_SPECS = {
    "psi":     ("psi", None),
    "forcing": ("forcing", None),
}


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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    log = logging.getLogger("postprocess_cahn_hilliard")

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

    log.info("Resampling %s -> %s (Nlat=%d, Nlon=%d)",
             args.raw, args.out, args.nlat, args.nlon)
    resample_run(
        args.raw, args.out,
        Nlat=args.nlat, Nlon=args.nlon,
        time_offset_s=0.0,  # solver already burned in; sim_time starts at 0.
        params=params, run_id=run_id,
        field_specs=CH_FIELD_SPECS,
        description="Cahn-Hilliard with rotating Y_ell^m forcing, "
                    "resampled to regular (lat, lon).",
    )
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
