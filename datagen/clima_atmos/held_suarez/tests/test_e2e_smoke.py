"""End-to-end smoke: one tiny corner run → Julia → postprocess → DataModule.

This is the "yes it actually works" test for the full producer chain.
Requires Julia + the env/ project + external/ClimaAtmos.jl/ checked out,
so it's behind ``RUN_CLIMA_INTEGRATION=1`` and intended to run on a
compute node (not the login node).

The corner uses a 1-day t_end and the smallest reasonable resolution to
keep this under ~10 minutes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
PRODUCER_DIR = REPO_ROOT / "datagen" / "clima_atmos" / "held_suarez"
ENV_DIR = PRODUCER_DIR / "env"
CLIMA_DIR = REPO_ROOT / "external" / "ClimaAtmos.jl"


def _julia_available() -> bool:
    return shutil.which("julia") is not None


_INTEGRATION_ENABLED = os.environ.get("RUN_CLIMA_INTEGRATION") == "1"


@pytest.mark.skipif(not _INTEGRATION_ENABLED,
                    reason="RUN_CLIMA_INTEGRATION not set")
@pytest.mark.skipif(not _julia_available(), reason="julia not on PATH")
@pytest.mark.skipif(not (ENV_DIR / "Manifest.toml").is_file(),
                    reason="env/Manifest.toml missing; run install_env.sbatch")
@pytest.mark.skipif(
    not (CLIMA_DIR / "config" / "model_configs" / "held_suarez.yml").is_file(),
    reason="external/ClimaAtmos.jl/ not checked out",
)
def test_one_corner_end_to_end(tmp_path: Path):
    # 1. Write a corner JSON.
    corner = {
        "run_id": 0, "run_name": "smoke",
        "params": {
            "tau_drag_days": 1.0, "tau_surf_days": 4.0,
            "tau_atm_days": 40.0, "delta_T_y": 60.0,
            "delta_theta_z": 10.0, "seed": 0,
        },
    }
    corner_path = tmp_path / "corner.json"
    corner_path.write_text(json.dumps(corner, indent=2))

    # 2. Write a cheap resolution override.
    res_yml = tmp_path / "res.yml"
    res_yml.write_text(
        "h_elem: 4\n"
        "z_max: 30000.0\n"
        "z_elem: 10\n"
        "dt: \"300secs\"\n"
        "diagnostics:\n"
        "  - short_name: [ua, va, ta]\n"
        "    period: \"6hours\"\n"
        "    reduction_time: \"inst\"\n"
        "    interpolate_to_pressure_levels: true\n"
        "  - short_name: [ps]\n"
        "    period: \"6hours\"\n"
        "    reduction_time: \"inst\"\n"
    )

    out_dir = tmp_path / "run_out"

    # 3. Driver: julia → postprocess → run.zarr.
    cmd = [
        "uv", "run", "--no-sync", "--project", "datagen", "python",
        "-m", "datagen.clima_atmos.held_suarez.scripts.run",
        "--config", str(corner_path),
        "--out-dir", str(out_dir),
        "--resolution-config", str(res_yml),
        "--t-end", "1days",
    ]
    env = {**os.environ, "JULIA_DEPOT_PATH": str(ENV_DIR / ".julia_depot")}
    subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=True)

    # 4. Zarr exists + the consumer can read it.
    zarr_path = out_dir / "run.zarr"
    assert zarr_path.is_dir(), f"missing {zarr_path}"

    import xarray as xr
    ds = xr.open_zarr(str(zarr_path), consolidated=True)
    assert set(ds.data_vars) >= {"u", "v", "T", "ps"}
    assert ds["u"].dims == ("time", "level", "lat", "lon")
    # Pressure axis is locked to ERA5 8 levels.
    assert tuple(int(round(v)) for v in ds["level"].values) == (
        50, 100, 250, 500, 700, 850, 925, 1000,
    )
