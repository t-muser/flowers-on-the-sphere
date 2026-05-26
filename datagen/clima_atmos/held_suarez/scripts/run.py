"""Single-corner driver: run ClimaAtmos via Julia, then post-process to Zarr.

Mirrors the CLI shape of ``datagen.mitgcm.held_suarez.scripts.run`` so the
SLURM array driver is unchanged when swapping producers.

What this script does:

  1. Loads the per-run JSON config emitted by ``generate_sweep.py``.
  2. Invokes ``julia --project=env run_hs.jl --corner ... --out-dir CLIMA/``
     to run the spectral-element HS simulation and emit ClimaAtmos NetCDF.
  3. Invokes ``postprocess.py`` to vertically interpolate onto the 8 ERA5
     pressure levels and write ``<out_dir>/run.zarr``.
  4. On any failure, drops ``<out_dir>.FAILED`` with the exception and
     params so the SLURM array keeps running.

Run from the repo root so ``datagen.*`` imports resolve::

    uv run --project datagen python -m datagen.clima_atmos.held_suarez.scripts.run \\
        --config datagen/clima_atmos/held_suarez/configs/run_0000.json \\
        --out-dir $DATA_ROOT/clima/runs/run_0000/ \\
        --resolution-config /path/to/he12ze31.yml \\
        --t-end 565days
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from datagen.clima_atmos.held_suarez.postprocess import postprocess_clima_dir

REPO_ROOT = Path(__file__).resolve().parents[4]
PRODUCER_DIR = REPO_ROOT / "datagen" / "clima_atmos" / "held_suarez"
DEFAULT_DRIVER_JL = PRODUCER_DIR / "run_hs.jl"
DEFAULT_ENV_DIR = PRODUCER_DIR / "env"
DEFAULT_CLIMA_DIR = REPO_ROOT / "external" / "ClimaAtmos.jl"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_julia(
    *,
    corner_json: Path,
    out_dir: Path,
    job_id: str,
    base_config: Path,
    resolution_config: Path | None,
    t_end: str | None,
    env_dir: Path,
    driver_jl: Path,
    julia_depot: Path | None,
) -> None:
    cmd = [
        "julia", f"--project={env_dir}", str(driver_jl),
        "--corner", str(corner_json),
        "--base-config", str(base_config),
        "--out-dir", str(out_dir),
        "--job-id", job_id,
    ]
    if resolution_config is not None:
        cmd += ["--resolution-config", str(resolution_config)]
    if t_end is not None:
        cmd += ["--t-end", t_end]

    env = dict(os.environ)
    if julia_depot is not None:
        env["JULIA_DEPOT_PATH"] = str(julia_depot)

    logging.getLogger(__name__).info("julia cmd: %s", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def _collect_nc(clima_dir: Path) -> list[Path]:
    """Return all NetCDF outputs written by ClimaAtmos under ``clima_dir``."""
    return sorted(clima_dir.rglob("*.nc"))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, required=True,
        help="Per-run JSON config emitted by generate_sweep.py.",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Root output directory for this run; contains clima/ and run.zarr.",
    )
    ap.add_argument(
        "--base-config", type=Path,
        default=DEFAULT_CLIMA_DIR / "config" / "model_configs" / "held_suarez.yml",
        help="ClimaAtmos base HS YAML config.",
    )
    ap.add_argument(
        "--resolution-config", type=Path, default=None,
        help="Optional YAML overriding h_elem/z_elem/dt/t_end.",
    )
    ap.add_argument(
        "--t-end", type=str, default=None,
        help="Override t_end for this run (e.g. '565days').",
    )
    ap.add_argument(
        "--env-dir", type=Path, default=DEFAULT_ENV_DIR,
        help="Julia project directory.",
    )
    ap.add_argument(
        "--driver-jl", type=Path, default=DEFAULT_DRIVER_JL,
        help="Path to run_hs.jl driver.",
    )
    ap.add_argument(
        "--julia-depot", type=Path, default=None,
        help="JULIA_DEPOT_PATH override (defaults to env-dir/.julia_depot).",
    )
    ap.add_argument(
        "--cleanup-clima", action="store_true",
        help="Delete the ClimaAtmos working dir after a successful Zarr write. "
             "Recommended at production resolution where the raw NetCDFs "
             "would otherwise dominate cluster disk.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging()
    log = logging.getLogger(__name__)

    with open(args.config, encoding="utf-8") as f:
        entry = json.load(f)
    params = entry["params"]
    run_id = int(entry.get("run_id", -1))
    job_id = entry.get("run_name", f"run_{run_id:04d}")
    log.info("Loaded config %s (run_id=%s)", args.config, run_id)

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

        log.info("Post-processing ClimaAtmos output → %s", zarr_path)
        postprocess_clima_dir(
            clima_dir=clima_dir,
            out_path=zarr_path,
            run_id=run_id,
            params=params,
        )
        if args.cleanup_clima:
            log.info("Cleaning up ClimaAtmos working dir %s", clima_dir)
            shutil.rmtree(clima_dir)

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
        with open(failed_marker, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return 1

    log.info("Run completed OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
