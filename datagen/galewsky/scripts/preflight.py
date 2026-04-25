"""Low-resolution stability sweep over the 32 parameter-box corners.

Generates a corners-only manifest (one config per extreme of the 5-axis box,
2**5 = 32 runs) and runs each at half resolution (``Nphi=256, Ntheta=128``)
for 1 day of simulated time. Mirrors the CLI shape of
``generate_sweep.py`` + ``run_galewsky.py`` so SLURM can treat it the same way.

Two sub-commands (invoke from the repo root so ``datagen.*`` imports resolve)::

  uv run --project datagen python -m datagen.galewsky.scripts.preflight generate \\
      --out datagen/galewsky/configs/preflight/

  mpirun -n 8 uv run --project datagen python -m datagen.galewsky.scripts.preflight run \\
      --config datagen/galewsky/configs/preflight/corner_00.json \\
      --out-dir "$DATA_ROOT/preflight/raw/corner_00/"
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

from datagen.galewsky.scripts.generate_sweep import PARAM_GRID
from datagen.galewsky.solver import run_simulation


def _corner_params() -> list[dict]:
    """All 32 corners: for each axis, pick min or max."""
    axes = list(PARAM_GRID.keys())
    lo_hi = [(min(vals), max(vals)) for vals in PARAM_GRID.values()]
    corners: list[dict] = []
    for combo in itertools.product(*lo_hi):
        corners.append({a: float(v) for a, v in zip(axes, combo)})
    return corners


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
        with open(out_dir / f"corner_{i:02d}.json", "w") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")
    manifest = {"n_runs": len(corners), "runs": manifest_entries}
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(corners)} corner configs to {out_dir}.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("preflight")

    with open(args.config) as f:
        entry = json.load(f)
    params = entry["params"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed_marker = args.out_dir.with_suffix(".FAILED")
    if failed_marker.exists():
        failed_marker.unlink()

    try:
        run_simulation(
            params,
            args.out_dir,
            snapshot_dt=args.snapshot_dt,
            stop_sim_time=args.stop_sim_time,
            Nphi=args.nphi,
            Ntheta=args.ntheta,
        )
    except Exception as exc:
        log.exception("Corner run failed")
        payload = {
            "params": params,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
        with open(failed_marker, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return 2

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser("generate", help="Emit corners-only configs.")
    ap_gen.add_argument("--out", type=Path, default=Path("datagen/galewsky/configs/preflight"))
    ap_gen.set_defaults(func=cmd_generate)

    ap_run = sub.add_parser("run", help="Run one corner config.")
    ap_run.add_argument("--config", type=Path, required=True)
    ap_run.add_argument("--out-dir", type=Path, required=True)
    ap_run.add_argument("--nphi", type=int, default=256)
    ap_run.add_argument("--ntheta", type=int, default=128)
    ap_run.add_argument("--stop-sim-time", type=float, default=86400.0)
    ap_run.add_argument("--snapshot-dt", type=float, default=3600.0)
    ap_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
