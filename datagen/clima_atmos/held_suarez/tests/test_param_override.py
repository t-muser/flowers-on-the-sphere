"""End-to-end test: corner JSON → ``run_hs.jl`` → NetCDF, with the
requested ΔT_y actually showing up in the steady-state gradient.

This is the only place we verify that our TOML override of the
ClimaParams keys really lands on the forcing tendency. If a future
ClimaAtmos release renames a key, this test catches the silent miss.

Requires:
    - julia on PATH
    - the env/ project instantiated (env/Manifest.toml present)
    - external/ClimaAtmos.jl/ checked out at the pinned version

Tagged ``integration_julia`` so the default ``pytest`` run skips it.
Invoke with ``RUN_CLIMA_INTEGRATION=1 uv run pytest ...`` to enable.

A pure-Python sanity test (no Julia) is also included: it verifies the
corner JSONs emitted by ``generate_sweep.py`` carry the 5 axes that
``run_hs.jl`` expects.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
PRODUCER_DIR = REPO_ROOT / "datagen" / "clima_atmos" / "held_suarez"
DRIVER_JL = PRODUCER_DIR / "run_hs.jl"
ENV_DIR = PRODUCER_DIR / "env"
CLIMA_DIR = REPO_ROOT / "external" / "ClimaAtmos.jl"


def _julia_available() -> bool:
    return shutil.which("julia") is not None


def _env_installed() -> bool:
    return (ENV_DIR / "Manifest.toml").is_file()


def _clima_checked_out() -> bool:
    return (CLIMA_DIR / "config" / "model_configs" / "held_suarez.yml").is_file()


_INTEGRATION_ENABLED = os.environ.get("RUN_CLIMA_INTEGRATION") == "1"


# ─── Pure-Python sanity tests ────────────────────────────────────────────────

class TestCornerSchema:
    """Verify the corner-JSON schema and the count emitted by the
    ClimaAtmos sweep generator. ``run_hs.jl`` reads exactly the 4 axes
    asserted here — adding or removing one without updating the driver
    is the failure mode this test catches.
    """

    def test_run_hs_jl_present(self):
        assert DRIVER_JL.is_file(), f"Missing driver: {DRIVER_JL}"

    def test_clima_corner_json_carries_expected_keys(self):
        """In-memory generation: avoid depending on prior CLI runs."""
        from datagen.clima_atmos.held_suarez.scripts.generate_sweep import iter_grid

        runs = iter_grid()
        assert len(runs) == 162
        for params in runs:
            for key in ("omega_factor", "delta_T_y", "delta_theta_z", "seed"):
                assert key in params, f"missing {key} in {params}"
            assert isinstance(params["seed"], int)
            assert params["omega_factor"] > 0
            # Damping-timescale axes from the old MITgcm grid must not
            # leak back in — they would silently be ignored by the
            # driver and produce hard-to-spot regressions.
            for stale in ("tau_drag_days", "tau_surf_days", "tau_atm_days"):
                assert stale not in params, f"stale axis {stale} in {params}"


# ─── Julia integration tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not _INTEGRATION_ENABLED,
                    reason="RUN_CLIMA_INTEGRATION not set; integration test skipped")
@pytest.mark.skipif(not _julia_available(),
                    reason="julia not on PATH")
@pytest.mark.skipif(not _env_installed(),
                    reason="env/Manifest.toml missing; run install_env.sbatch first")
@pytest.mark.skipif(not _clima_checked_out(),
                    reason="external/ClimaAtmos.jl/ not checked out")
class TestParamOverrideLands:
    """Run two short corners, then check the ΔT_y override actually
    changes the equator-to-pole gradient in ``ta``.

    These runs use the smallest possible config (he4ze10, 1 day) so the
    test fits in a few minutes on a compute node. Run from inside a
    SLURM job, not on the login node.
    """

    def _run_corner(self, tmp_path: Path, name: str, delta_T_y: float) -> Path:
        corner = {
            "run_id": 0, "run_name": name,
            "params": {
                "omega_factor":  1.0,
                "delta_T_y":     float(delta_T_y),
                "delta_theta_z": 10.0,
                "seed":          0,
            },
        }
        corner_path = tmp_path / f"{name}.json"
        corner_path.write_text(json.dumps(corner, indent=2))

        out_dir = tmp_path / name
        out_dir.mkdir()

        # Inline resolution-override YAML for a cheap test run.
        res_override = tmp_path / f"{name}_res.yml"
        res_override.write_text(
            "h_elem: 4\n"
            "z_max: 30000.0\n"
            "z_elem: 10\n"
            "t_end: \"1days\"\n"
            "dt: \"300secs\"\n"
        )

        env = {
            **os.environ,
            "JULIA_DEPOT_PATH": str(ENV_DIR / ".julia_depot"),
        }
        cmd = [
            "julia", f"--project={ENV_DIR}", str(DRIVER_JL),
            "--corner", str(corner_path),
            "--base-config", str(CLIMA_DIR / "config" / "model_configs"
                                 / "held_suarez.yml"),
            "--resolution-config", str(res_override),
            "--out-dir", str(out_dir),
            "--job-id", name,
        ]
        subprocess.run(cmd, env=env, check=True)
        return out_dir

    def test_higher_delta_T_y_produces_steeper_gradient(self, tmp_path: Path):
        import xarray as xr

        low = self._run_corner(tmp_path, "low",  delta_T_y=40.0)
        high = self._run_corner(tmp_path, "high", delta_T_y=80.0)

        def _grad(out_dir: Path) -> float:
            nc = next(out_dir.rglob("*.nc"))
            ds = xr.open_dataset(str(nc), decode_times=False)
            # Take the last time step, vertically average ta in the
            # mid-troposphere, and measure the equator-pole difference.
            ta = ds["ta"].isel(time=-1)
            # Latitude-mean over a 30°-wide tropical band vs a polar band.
            lat = ds["lat"]
            tropic = ta.where(np.abs(lat) <= 15.0, drop=True).mean()
            polar  = ta.where(lat >= 75.0, drop=True).mean()
            return float(tropic - polar)

        g_low  = _grad(low)
        g_high = _grad(high)
        assert g_high > g_low, (
            f"ΔT_y=80 K should give a larger eq-pole gradient than 40 K; "
            f"got high={g_high:.2f} K vs low={g_low:.2f} K"
        )
