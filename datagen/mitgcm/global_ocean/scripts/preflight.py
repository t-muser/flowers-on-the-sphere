"""Corner preflight for the MITgcm global-ocean sweep.

This mirrors :mod:`datagen.mitgcm.held_suarez.scripts.preflight` for
Held-Suarez:
``generate`` emits one config per corner of the parameter box, and ``run``
executes a single corner with reduced duration.

Run from the repository root::

    uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.preflight generate \\
        --out datagen/mitgcm/global_ocean/configs/preflight

    uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.preflight run \\
        --config datagen/mitgcm/global_ocean/configs/preflight/corner_00.json \\
        --out-dir "$DATA_ROOT/mitgcm/global-ocean/preflight/corner_00"
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
import traceback
from dataclasses import fields
from pathlib import Path

from datagen.mitgcm.global_ocean import GlobalOceanRunConfig, run_simulation
from datagen.mitgcm.global_ocean.scripts.generate_sweep import (
    FIXED_PARAMS,
    PARAM_GRID,
)

_PREFLIGHT_N_TIMESTEPS: int = 30
_PREFLIGHT_SNAPSHOT_INTERVAL_DAYS: float = 10.0


def _corner_params() -> list[dict]:
    """All corners: for each sweep axis take its minimum and maximum."""
    axes = list(PARAM_GRID.keys())
    lo_hi = [(min(vals), max(vals)) for vals in PARAM_GRID.values()]
    runs: list[dict] = []
    for combo in itertools.product(*lo_hi):
        params = {axis: float(value) for axis, value in zip(axes, combo)}
        params.update(FIXED_PARAMS)
        runs.append(params)
    return runs


def _hash_params(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def cmd_generate(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    corners = _corner_params()
    manifest_entries: list[dict] = []
    for i, params in enumerate(corners):
        entry = {
            "run_id": i,
            "run_name": f"corner_{i:02d}",
            "params": params,
            "param_hash": _hash_params(params),
        }
        manifest_entries.append(entry)
        _write_json(out_dir / f"corner_{i:02d}.json", entry)

    _write_json(
        out_dir / "manifest.json",
        {"n_runs": len(corners), "corners": manifest_entries},
    )
    print(f"Wrote {len(corners)} global-ocean corner configs to {out_dir}.")
    return 0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    _setup_logging()
    log = logging.getLogger(__name__)

    with open(args.config, encoding="utf-8") as f:
        entry = json.load(f)
    params = entry["params"]
    log.info("Loaded global-ocean corner config %s (run_id=%s)",
             args.config, entry.get("run_id"))

    out_dir: Path = args.out_dir
    failed_marker = out_dir.parent / (out_dir.name + ".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    cfg_field_names = {f.name for f in fields(GlobalOceanRunConfig)}
    overrides: dict = {}
    for dest in (
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
        "timeout_s",
    ):
        assert dest in cfg_field_names, f"Unknown GlobalOceanRunConfig field: {dest!r}"
        val = getattr(args, dest, None)
        if val is not None:
            overrides[dest] = val
    if args.serial:
        overrides["mpirun_cmd"] = ()

    try:
        run_simulation(params, out_dir, **overrides)
    except Exception as exc:
        log.exception("Global-ocean corner run failed")
        payload = {
            "run_id": entry.get("run_id"),
            "config_path": str(args.config),
            "params": params,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        _write_json(failed_marker, payload)
        return 1

    log.info("Global-ocean corner run completed OK.")
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser(
        "generate",
        help="Emit corner-only JSON configs + manifest.",
    )
    ap_gen.add_argument(
        "--out",
        type=Path,
        default=Path("datagen/mitgcm/global_ocean/configs/preflight"),
        help="Output directory for corner configs.",
    )
    ap_gen.set_defaults(func=cmd_generate)

    ap_run = sub.add_parser("run", help="Run one global-ocean corner config.")
    ap_run.add_argument("--config", type=Path, required=True)
    ap_run.add_argument("--out-dir", type=Path, required=True)
    ap_run.add_argument("--executable", type=Path, default=None)
    ap_run.add_argument("--input-dir", type=Path, default=None)
    ap_run.add_argument("--n-timesteps", type=int, default=_PREFLIGHT_N_TIMESTEPS)
    ap_run.add_argument(
        "--snapshot-interval-days",
        type=float,
        default=_PREFLIGHT_SNAPSHOT_INTERVAL_DAYS,
    )
    ap_run.add_argument("--delta-t-mom", type=float, default=None)
    ap_run.add_argument("--delta-t-tracer", type=float, default=None)
    ap_run.add_argument("--delta-t-clock", type=float, default=None)
    ap_run.add_argument("--delta-t-freesurf", type=float, default=None)
    ap_run.add_argument("--tracer-level", type=int, default=None)
    ap_run.add_argument("--velocity-level", type=int, default=None)
    ap_run.add_argument("--timeout-s", type=float, default=None)
    ap_run.add_argument(
        "--serial",
        action="store_true",
        help="Run ./mitgcmuv directly instead of `mpirun -n 1 ./mitgcmuv`.",
    )
    ap_run.set_defaults(func=cmd_run)

    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
