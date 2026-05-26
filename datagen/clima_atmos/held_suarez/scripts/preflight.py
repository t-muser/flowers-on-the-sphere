"""Low/high-corner stability sweep at reduced duration.

Generates 2^N axis-extreme corner configs from the production sweep
grid and runs each for a short ``_PREFLIGHT_T_END`` to surface
integrator instabilities (CFL blowup, NaN propagation, jet sharpness)
before committing to the full 565-day production array.

Sub-commands::

    uv run --project datagen python -m datagen.clima_atmos.held_suarez.scripts.preflight generate \\
        --out datagen/clima_atmos/held_suarez/configs/preflight/

    uv run --project datagen python -m datagen.clima_atmos.held_suarez.scripts.preflight run \\
        --config datagen/clima_atmos/held_suarez/configs/preflight/corner_00.json \\
        --out-dir "$DATA_ROOT/clima/preflight/corner_00/"
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
import traceback
from pathlib import Path

from datagen.clima_atmos.held_suarez.scripts.generate_sweep import (
    FIXED_PARAMS, PARAM_GRID,
)
from datagen.clima_atmos.held_suarez.scripts.run import (
    DEFAULT_CLIMA_DIR, DEFAULT_DRIVER_JL, DEFAULT_ENV_DIR,
    _run_julia, _collect_nc, _setup_logging,
)
from datagen.clima_atmos.held_suarez.postprocess import postprocess_clima_dir


# Preflight defaults — short enough to verify solver health without
# committing to a full 565-day production run.
_PREFLIGHT_T_END: str = "60days"  # 30 d spin-up + 30 d data collection equivalent


def _corner_params() -> list[dict]:
    """One corner per axis-extreme combination, excluding the seed axis.

    The seed axis is a statistical-realization axis, not a physical
    one, so we don't expand it in the preflight grid — preflight only
    needs to stress-test the integrator across physical extremes. We
    fix seed = 0 for every corner.

    For the current 3-physical-axis grid this gives 2^3 = 8 corners.
    """
    axes = [a for a in PARAM_GRID if a != "seed"]
    lo_hi = [(min(PARAM_GRID[a]), max(PARAM_GRID[a])) for a in axes]
    runs: list[dict] = []
    for combo in itertools.product(*lo_hi):
        params: dict = {}
        for axis, value in zip(axes, combo):
            params[axis] = float(value)
        params["seed"] = 0
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
            "run_id":     i,
            "run_name":   f"corner_{i:02d}",
            "params":     params,
            "param_hash": _hash_params(params),
        }
        manifest_entries.append(entry)
        _write_json(out_dir / f"corner_{i:02d}.json", entry)
    _write_json(
        out_dir / "manifest.json",
        {"n_runs": len(corners), "corners": manifest_entries},
    )
    print(f"Wrote {len(corners)} corner configs to {out_dir}.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _setup_logging()
    log = logging.getLogger(__name__)

    with open(args.config, encoding="utf-8") as f:
        entry = json.load(f)
    params = entry["params"]
    run_id = int(entry.get("run_id", -1))
    job_id = entry.get("run_name", f"corner_{run_id:02d}")
    log.info("Loaded corner config %s (run_id=%s)", args.config, run_id)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_marker = out_dir.parent / (out_dir.name + ".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    clima_dir = out_dir / "clima"
    clima_dir.mkdir(exist_ok=True)
    zarr_path = out_dir / "run.zarr"

    julia_depot = args.julia_depot or (args.env_dir / ".julia_depot")

    try:
        _run_julia(
            corner_json=args.config,
            out_dir=clima_dir,
            job_id=job_id,
            base_config=args.base_config,
            resolution_config=args.resolution_config,
            t_end=args.t_end,
            env_dir=args.env_dir,
            driver_jl=args.driver_jl,
            julia_depot=julia_depot,
        )
        postprocess_clima_dir(
            clima_dir=clima_dir,
            out_path=zarr_path,
            run_id=run_id,
            params=params,
        )
    except Exception as exc:
        log.exception("Corner run failed")
        payload = {
            "run_id":      run_id,
            "config_path": str(args.config),
            "params":      params,
            "exception":   repr(exc),
            "traceback":   traceback.format_exc(),
        }
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        _write_json(failed_marker, payload)
        return 1

    log.info("Corner run completed OK.")
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser("generate", help="Emit corners-only configs + manifest.")
    ap_gen.add_argument(
        "--out", type=Path,
        default=Path("datagen/clima_atmos/held_suarez/configs/preflight"),
        help="Output directory for corner configs.",
    )
    ap_gen.set_defaults(func=cmd_generate)

    ap_run = sub.add_parser("run", help="Run one corner config.")
    ap_run.add_argument(
        "--config", type=Path, required=True,
        help="Path to a corner JSON config emitted by the generate sub-command.",
    )
    ap_run.add_argument(
        "--out-dir", type=Path, required=True,
        help="Root output directory for this corner run.",
    )
    ap_run.add_argument(
        "--base-config", type=Path,
        default=DEFAULT_CLIMA_DIR / "config" / "model_configs" / "held_suarez.yml",
    )
    ap_run.add_argument(
        "--resolution-config", type=Path, default=None,
        help="Optional resolution-override YAML.",
    )
    ap_run.add_argument(
        "--t-end", type=str, default=_PREFLIGHT_T_END,
        help=f"t_end for the preflight run (default {_PREFLIGHT_T_END}).",
    )
    ap_run.add_argument("--env-dir",   type=Path, default=DEFAULT_ENV_DIR)
    ap_run.add_argument("--driver-jl", type=Path, default=DEFAULT_DRIVER_JL)
    ap_run.add_argument("--julia-depot", type=Path, default=None)
    ap_run.set_defaults(func=cmd_run)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
