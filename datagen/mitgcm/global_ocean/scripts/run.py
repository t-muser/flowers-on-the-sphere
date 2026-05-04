"""Run one MITgcm global ocean tutorial simulation.

Example from the repository root::

    uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.run \\
        --out-dir /tmp/global-ocean-smoke \\
        --n-timesteps 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import fields
from pathlib import Path

from datagen.mitgcm.global_ocean import GlobalOceanRunConfig, run_simulation


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--executable", type=Path, default=None)
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--n-timesteps", type=int, default=None)
    ap.add_argument("--snapshot-interval-days", type=float, default=None)
    ap.add_argument("--delta-t-mom", type=float, default=None)
    ap.add_argument("--delta-t-tracer", type=float, default=None)
    ap.add_argument("--delta-t-clock", type=float, default=None)
    ap.add_argument("--delta-t-freesurf", type=float, default=None)
    ap.add_argument("--tracer-level", type=int, default=None)
    ap.add_argument("--velocity-level", type=int, default=None)
    ap.add_argument("--gm-background-k", type=float, default=None)
    ap.add_argument("--visc-ah", type=float, default=None)
    ap.add_argument("--diff-kr", type=float, default=None)
    ap.add_argument("--tau-theta-relax-days", type=float, default=None)
    ap.add_argument("--tau-salt-relax-days", type=float, default=None)
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument(
        "--serial",
        action="store_true",
        help="Run ./mitgcmuv directly instead of `mpirun -n 1 ./mitgcmuv`.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging()
    log = logging.getLogger(__name__)

    cfg_fields = {f.name for f in fields(GlobalOceanRunConfig)}
    overrides = {}
    for key in (
        "executable",
        "input_dir",
        "n_timesteps",
        "snapshot_interval_days",
        "delta_t_mom",
        "delta_t_tracer",
        "delta_t_clock",
        "delta_t_freesurf",
        "tracer_level",
        "velocity_level",
        "gm_background_k",
        "visc_ah",
        "diff_kr",
        "tau_theta_relax_days",
        "tau_salt_relax_days",
        "timeout_s",
    ):
        assert key in cfg_fields, f"Unknown GlobalOceanRunConfig field: {key}"
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.serial:
        overrides["mpirun_cmd"] = ()

    failed_marker = args.out_dir.parent / (args.out_dir.name + ".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    try:
        run_simulation(None, args.out_dir, **overrides)
    except Exception as exc:
        log.exception("Global ocean simulation failed")
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        failed_marker.write_text(
            "exception: " + repr(exc) + "\n\n" + traceback.format_exc()
        )
        return 1

    log.info("Global ocean run completed OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
