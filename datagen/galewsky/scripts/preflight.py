"""Low-resolution stability sweep over the 32 parameter-box corners.

Generates a corners-only manifest (one config per extreme of the 5-axis box,
2**5 = 32 runs) and runs each at half resolution (``Nphi=256, Ntheta=128``)
for 1 day of simulated time. Mirrors the CLI shape of
``generate_sweep.py`` + ``run.py`` so SLURM can treat it the same way.

Two sub-commands (invoke from the repo root so ``datagen.*`` imports resolve)::

  uv run --project datagen python -m datagen.galewsky.scripts.preflight generate \\
      --out datagen/galewsky/configs/preflight/

  mpirun -n 8 uv run --project datagen python -m datagen.galewsky.scripts.preflight run \\
      --config datagen/galewsky/configs/preflight/corner_00.json \\
      --out-dir "$DATA_ROOT/preflight/raw/corner_00/"

On solver failure, writes ``<out_dir>.FAILED`` (rank 0 only) with the exception
and params so the SLURM array can keep going.
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

from mpi4py import MPI

from datagen.galewsky.scripts.generate_sweep import PARAM_GRID
from datagen.galewsky.solver import RunConfig, run_simulation


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def _corner_params() -> list[dict]:
    """All 32 corners: for each axis, pick min or max."""
    axes = list(PARAM_GRID.keys())
    lo_hi = [(min(vals), max(vals)) for vals in PARAM_GRID.values()]
    return [
        {a: float(v) for a, v in zip(axes, combo)}
        for combo in itertools.product(*lo_hi)
    ]


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON with trailing newline and UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def cmd_generate(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    corners = _corner_params()
    manifest_entries: list[dict] = []
    for i, params in enumerate(corners):
        payload = json.dumps(params, sort_keys=True).encode()
        phash = hashlib.sha1(payload).hexdigest()[:12]
        entry = {
            "run_id": i,
            "run_name": f"corner_{i:02d}",
            "params": params,
            "param_hash": phash,
        }
        manifest_entries.append(entry)
        _write_json(out_dir / f"corner_{i:02d}.json", entry)

    _write_json(
        out_dir / "manifest.json",
        {"n_runs": len(corners), "runs": manifest_entries},
    )
    print(f"Wrote {len(corners)} corner configs to {out_dir}.")
    return 0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
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

    out_dir: Path = args.out_dir
    failed_marker = out_dir.parent / (out_dir.name + ".FAILED")

    if MPI.COMM_WORLD.rank == 0 and failed_marker.exists():
        failed_marker.unlink()
    MPI.COMM_WORLD.Barrier()

    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(RunConfig)
        if hasattr(args, f.name)
    }

    try:
        run_simulation(params, out_dir, **overrides)
    except Exception as exc:
        log.exception("Corner run failed")
        if MPI.COMM_WORLD.rank == 0:
            _write_json(failed_marker, {
                "params": params,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
            })
        return 1

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser("generate", help="Emit corners-only configs.")
    ap_gen.add_argument(
        "--out", type=Path,
        default=Path("datagen/galewsky/configs/preflight"),
    )
    ap_gen.set_defaults(func=cmd_generate)

    ap_run = sub.add_parser("run", help="Run one corner config.")
    ap_run.add_argument("--config", type=Path, required=True)
    ap_run.add_argument("--out-dir", type=Path, required=True)
    ap_run.add_argument("--nphi", type=int, default=256, dest="Nphi")
    ap_run.add_argument("--ntheta", type=int, default=128, dest="Ntheta")
    ap_run.add_argument("--stop-sim-time", type=float, default=86400.0,
                        dest="stop_sim_time")
    ap_run.add_argument("--snapshot-dt", type=float, default=3600.0,
                        dest="snapshot_dt")
    ap_run.set_defaults(func=cmd_run)

    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
