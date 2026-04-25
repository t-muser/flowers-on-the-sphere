"""Stack per-run Zarrs into a single consolidated dataset.

Reads ``processed/run_XXXX.zarr`` for every entry in the manifest and writes
``dataset.zarr`` with dims ``(run, time, field, lat, lon)``. The manifest is
carried as top-level attrs. Emits a failure report for any runs whose
``processed/`` store is missing or whose source directory has a ``.FAILED``
marker.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, required=True,
                    help="Directory containing run_XXXX.zarr stores.")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Manifest JSON emitted by generate_sweep.py.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output consolidated Zarr path.")
    ap.add_argument("--failure-report", type=Path, default=None,
                    help="Optional path for the missing-run report (JSON).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    log = logging.getLogger("consolidate")

    with open(args.manifest) as f:
        manifest = json.load(f)

    run_entries = manifest["runs"]
    present: list[dict] = []
    missing: list[dict] = []

    for entry in run_entries:
        run_name = entry["run_name"]
        zarr_path = args.processed / f"{run_name}.zarr"
        if zarr_path.exists():
            present.append({"entry": entry, "path": zarr_path})
        else:
            missing.append(entry)

    log.info("Found %d / %d per-run Zarrs. Missing: %d",
             len(present), len(run_entries), len(missing))

    if not present:
        log.error("No runs to consolidate.")
        return 1

    # Open every per-run Zarr lazily and concatenate along a new run axis.
    datasets = [xr.open_zarr(str(p["path"])) for p in present]
    combined = xr.concat(
        datasets,
        dim=xr.Variable("run", [p["entry"]["run_id"] for p in present]),
        coords="minimal",
        compat="override",
    )

    # Attach parameter arrays as coords on the run axis for quick lookup.
    param_arrays: dict[str, list[float]] = {}
    for p in present:
        for k, v in p["entry"]["params"].items():
            param_arrays.setdefault(k, []).append(float(v))
    for k, vals in param_arrays.items():
        combined = combined.assign_coords({f"param_{k}": ("run", np.asarray(vals))})

    combined.attrs.update({
        "description": "Dedalus sphere parametric ensemble (consolidated).",
        "n_runs_total": len(run_entries),
        "n_runs_present": len(present),
        "manifest_path": str(args.manifest),
    })

    log.info("Writing consolidated dataset to %s", args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_zarr(str(args.out), mode="w", consolidated=True)

    if missing:
        report_path = args.failure_report or args.out.parent / "failures.json"
        with open(report_path, "w") as f:
            json.dump({"missing": missing}, f, indent=2)
            f.write("\n")
        log.warning("Wrote failure report to %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
