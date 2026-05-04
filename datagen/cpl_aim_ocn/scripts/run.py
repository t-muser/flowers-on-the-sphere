"""Single-run driver for the coupled AIM+ocean wrapper.

Designed to be launched from the repo root, typically as one element
of a SLURM array job::

    uv run --project datagen python -m datagen.cpl_aim_ocn.scripts.run \\
        --config datagen/cpl_aim_ocn/configs/run_0000.json \\
        --out-dir $DATA_ROOT/cpl_aim_ocn/runs/run_0000/

The driver loads the per-run JSON emitted by ``generate_sweep.py``,
applies any CLI overrides to :class:`~datagen.cpl_aim_ocn.solver.RunConfig`
fields, and invokes :func:`~datagen.cpl_aim_ocn.solver.run_simulation`.

On any failure (binary crash, namelist error, post-processing error)
the driver writes ``<out_dir>.FAILED`` with the exception text + the
parameter dict, so that the parent SLURM array can keep going and a
downstream consolidation step can produce a failure report.

Each run honours the ``spinup_days`` / ``data_days`` /
``snapshot_interval_days`` keys from FIXED_PARAMS in the JSON; CLI
flags override those when set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Sequence

from datagen.cpl_aim_ocn.solver import RunConfig, run_simulation


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, required=True,
        help="Path to the per-run JSON config emitted by generate_sweep.py.",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Root output directory for this run.",
    )
    # ── RunConfig overrides ──
    ap.add_argument(
        "--spinup-days", type=float, default=None, dest="spinup_days",
        help="Phase-1 spin-up duration [days] (overrides config default).",
    )
    ap.add_argument(
        "--data-days", type=float, default=None, dest="data_days",
        help="Phase-2 data-collection duration [days].",
    )
    ap.add_argument(
        "--snapshot-interval-days", type=float, default=None,
        dest="snapshot_interval_days",
        help="Diagnostic-output cadence in the data phase [days].",
    )
    ap.add_argument(
        "--delta-t-atm", type=float, default=None, dest="delta_t_atm",
        help="Atmospheric timestep [s].",
    )
    ap.add_argument(
        "--delta-t-ocn", type=float, default=None, dest="delta_t_ocn",
        help="Oceanic timestep [s].",
    )
    ap.add_argument(
        "--cpl-send-freq", type=float, default=None, dest="cpl_atm_send_freq_s",
        help="Coupler exchange period [s] (must divide both deltas).",
    )
    ap.add_argument(
        "--mpirun", type=str, default=None,
        help="Override MPI launcher name (default 'mpirun').",
    )
    ap.add_argument(
        "--inputs-root", type=Path, default=None, dest="inputs_root",
        help="Override path to inputs/ tree (default: package-local).",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging()
    log = logging.getLogger("cpl_aim_ocn.run")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    params = config["params"]
    log.info("Loaded config %s (run_id=%s)", args.config, config.get("run_id"))

    out_dir: Path = args.out_dir
    failed_marker = out_dir.parent / (out_dir.name + ".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    # ── Resolve CLI overrides into a dict of RunConfig fields ──
    rc_field_names = {f.name for f in fields(RunConfig)}
    overrides: dict = {}
    cli_to_field = (
        ("spinup_days", "spinup_days"),
        ("data_days", "data_days"),
        ("snapshot_interval_days", "snapshot_interval_days"),
        ("delta_t_atm", "delta_t_atm"),
        ("delta_t_ocn", "delta_t_ocn"),
        ("cpl_atm_send_freq_s", "cpl_atm_send_freq_s"),
        ("mpirun", "mpirun"),
        ("inputs_root", "inputs_root"),
    )
    for cli_dest, rc_name in cli_to_field:
        # Defensive: a typo here would silently drop the override.
        assert rc_name in rc_field_names, f"Unknown RunConfig field: {rc_name!r}"
        val = getattr(args, cli_dest, None)
        if val is not None:
            overrides[rc_name] = val

    # ── Apply per-run defaults from the JSON's FIXED_PARAMS section ──
    # generate_sweep.py stores spinup_days / data_days /
    # snapshot_interval_days in each run's params; we pull them into
    # RunConfig overrides if not already supplied via CLI.
    for k in ("spinup_days", "data_days", "snapshot_interval_days"):
        if k not in overrides and k in params:
            overrides[k] = float(params[k])

    log.info("RunConfig overrides: %s", overrides)

    try:
        run_simulation(params, out_dir, **overrides)
    except Exception as exc:
        log.exception("Simulation failed")
        payload = {
            "run_id":      config.get("run_id"),
            "config_path": str(args.config),
            "params":      params,
            "exception":   repr(exc),
            "traceback":   traceback.format_exc(),
        }
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_marker, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return 1

    log.info("Run completed OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
