"""Crop Mickelin per-run Zarr trajectories to a time window.

Reads each ``run_XXXX.zarr`` from ``--src``, slices the ``time`` axis to the
requested window, rebases ``time`` so the first kept snapshot is ``t = 0``,
and writes the result to ``--dst``. The original zarrs are left untouched.

The ``time`` axis stored in the zarrs is in the resample step's "physical
seconds" convention inherited from Galewsky (``τ ≈ 3600`` in stored units;
see ``datagen/resample.py``). Window bounds are specified in τ for clarity
and converted internally.

Default window is ``0 ≤ t ≤ 40 τ`` (≈ 200 snapshots, 10 s of animation
playback at 20 fps cadence) — the v1 publishable window.

Usage::

    uv run --project datagen python -m datagen.mickelin.scripts.crop_zarrs \\
        --src $DATASET_ROOT/processed \\
        --dst $DATASET_ROOT/processed_cropped \\
        --splits datagen/mickelin/configs/splits.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import xarray as xr


TAU_IN_STORED_UNITS = 3600.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="Source processed/ directory containing run_XXXX.zarr.")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Destination directory for cropped zarrs.")
    ap.add_argument("--splits", type=Path, default=None,
                    help="splits.json — if given, crop only the run_ids it lists.")
    ap.add_argument("--t-start-tau", type=float, default=0.0,
                    help="Window start in τ (default: 0).")
    ap.add_argument("--t-stop-tau", type=float, default=40.0,
                    help="Window end in τ (default: 40 τ ≈ 10 s anim @ 20 fps).")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    if args.splits is not None:
        sp = json.loads(args.splits.read_text())
        run_ids = sorted(set(sp["train"] + sp["val"] + sp["test"]))
    else:
        run_ids = sorted(int(p.name[4:8]) for p in args.src.glob("run_*.zarr"))

    t_lo = args.t_start_tau * TAU_IN_STORED_UNITS
    t_hi = args.t_stop_tau * TAU_IN_STORED_UNITS

    n = len(run_ids)
    print(f"Cropping {n} runs to t ∈ [{args.t_start_tau}, {args.t_stop_tau}] τ "
          f"(stored window [{t_lo:.0f}, {t_hi:.0f}])")
    print(f"  src: {args.src}")
    print(f"  dst: {args.dst}", flush=True)

    t_start = time.time()
    for i, rid in enumerate(run_ids):
        src = args.src / f"run_{rid:04d}.zarr"
        dst = args.dst / f"run_{rid:04d}.zarr"
        if dst.exists():
            shutil.rmtree(dst)

        ds = xr.open_zarr(src, decode_times=False)
        mask = (ds.time >= t_lo) & (ds.time <= t_hi)
        ds_c = ds.isel(time=mask)
        ds_c = ds_c.assign_coords(time=ds_c.time - t_lo)
        ds_c.to_zarr(dst, mode="w", consolidated=True)

        if (i + 1) % 50 == 0 or i + 1 == n:
            dt = time.time() - t_start
            print(f"  {i + 1}/{n}  ({dt:.1f}s elapsed, {dt / (i + 1):.2f} s/run)",
                  flush=True)

    print(f"Done. Cropped {n} runs in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
