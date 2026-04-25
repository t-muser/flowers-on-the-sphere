"""Resample one Cahn-Hilliard run's HDF5 output onto a regular (lat, lon) Zarr.

Mirrors ``galewsky/scripts/postprocess.py`` but uses the FiPy unstructured-mesh
resampling path (kd-tree IDW on the unit sphere) and a single-file raw input
rather than a directory of Dedalus snapshots::

    uv run --project datagen python -m datagen.cahn_hilliard.scripts.postprocess \\
        --raw $DATA_ROOT/raw/run_0000/cahn_s1.h5 \\
        --out $DATA_ROOT/processed/run_0000.zarr \\
        --config datagen/cahn_hilliard/configs/run_0000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datagen.resample import resample_unstructured_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True,
                    help="Path to the FiPy HDF5 snapshot file (one run).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output Zarr store path.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional config JSON, attached as Zarr attrs.")
    ap.add_argument("--nlat", type=int, default=256)
    ap.add_argument("--nlon", type=int, default=512)
    ap.add_argument("--k-neighbors", type=int, default=4,
                    help="Nearest neighbours used for inverse-distance weighting.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    log = logging.getLogger("postprocess_cahn_hilliard")

    # Skip silently if the raw run is marked FAILED upstream. The marker lives
    # next to the run directory (``<out_dir>.FAILED``), i.e. one level up.
    failed_marker = args.raw.parent.with_suffix(".FAILED")
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

    log.info("Resampling %s -> %s (Nlat=%d, Nlon=%d, k=%d)",
             args.raw, args.out, args.nlat, args.nlon, args.k_neighbors)
    resample_unstructured_run(
        args.raw, args.out,
        Nlat=args.nlat, Nlon=args.nlon,
        field_specs={"phi": "phi"},
        params=params, run_id=run_id,
        k_neighbors=args.k_neighbors,
        description="FiPy Cahn-Hilliard on sphere, resampled to regular (lat, lon).",
    )
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
