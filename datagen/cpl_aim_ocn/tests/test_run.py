"""Tests for ``datagen/cpl_aim_ocn/scripts/run.py``.

The single-run driver is the SLURM-array entry point — its core
responsibilities are:

* parse a per-run JSON config + CLI overrides into a RunConfig;
* call ``run_simulation``;
* on **any** exception, write ``<out_dir>.FAILED`` so the SLURM
  array can move on.

We test these behaviours by monkeypatching ``run_simulation`` so the
driver's plumbing runs in-process without spawning binaries.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_run.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagen.cpl_aim_ocn.scripts import run as run_mod


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def good_config_file(tmp_path: Path) -> Path:
    """Write a minimal, valid run_XXXX.json file."""
    config = {
        "run_id":   42,
        "run_name": "run_0042",
        "params": {
            "co2_ppm":     560.0,
            "solar_scale": 1.03,
            "gm_kappa":    2000.0,
            "seed":        2,
            "spinup_days": 30.0,
            "data_days":   365.0,
            "snapshot_interval_days": 1.0,
        },
        "param_hash": "abcdef012345",
    }
    p = tmp_path / "run_0042.json"
    p.write_text(json.dumps(config) + "\n")
    return p


@pytest.fixture
def captured_run_simulation(monkeypatch):
    """Replace solver.run_simulation with a stub that records its
    inputs and pretends to succeed. Returns the call-record list."""
    calls: list[tuple[dict, Path, dict]] = []

    def fake_run(params, out_dir, **overrides):
        calls.append((dict(params), Path(out_dir), dict(overrides)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return Path(out_dir) / "run.zarr"

    monkeypatch.setattr(run_mod, "run_simulation", fake_run)
    return calls


# ─── Happy-path CLI ──────────────────────────────────────────────────────────

class TestSuccessfulRun:
    def test_loads_params_from_config(
        self, good_config_file: Path, captured_run_simulation, tmp_path: Path
    ):
        out_dir = tmp_path / "out"
        rc = run_mod.main([
            "--config", str(good_config_file),
            "--out-dir", str(out_dir),
        ])
        assert rc == 0
        assert len(captured_run_simulation) == 1
        params, called_out, overrides = captured_run_simulation[0]
        assert params["co2_ppm"] == 560.0
        assert params["seed"] == 2
        assert called_out == out_dir

    def test_fixed_params_become_overrides(
        self, good_config_file: Path, captured_run_simulation, tmp_path: Path
    ):
        # ``spinup_days``, ``data_days``, ``snapshot_interval_days`` are
        # part of FIXED_PARAMS and therefore live in ``params``. The
        # driver must lift them onto RunConfig via overrides so the
        # solver actually picks them up (run_simulation forwards
        # **overrides into RunConfig.replace).
        run_mod.main([
            "--config", str(good_config_file),
            "--out-dir", str(tmp_path / "out"),
        ])
        _, _, overrides = captured_run_simulation[0]
        assert overrides["spinup_days"] == 30.0
        assert overrides["data_days"] == 365.0
        assert overrides["snapshot_interval_days"] == 1.0

    def test_cli_override_wins_over_json(
        self, good_config_file: Path, captured_run_simulation, tmp_path: Path
    ):
        # CLI --spinup-days 0.5 must clobber the JSON's 30.0.
        run_mod.main([
            "--config", str(good_config_file),
            "--out-dir", str(tmp_path / "out"),
            "--spinup-days", "0.5",
        ])
        _, _, overrides = captured_run_simulation[0]
        assert overrides["spinup_days"] == 0.5
        # And the un-overridden value still tracks the JSON.
        assert overrides["data_days"] == 365.0

    def test_extra_runconfig_overrides_pass_through(
        self, good_config_file: Path, captured_run_simulation, tmp_path: Path
    ):
        run_mod.main([
            "--config", str(good_config_file),
            "--out-dir", str(tmp_path / "out"),
            "--delta-t-atm", "300.0",
            "--mpirun", "srun",
        ])
        _, _, overrides = captured_run_simulation[0]
        assert overrides["delta_t_atm"] == 300.0
        assert overrides["mpirun"] == "srun"


# ─── Failure-path: .FAILED marker ────────────────────────────────────────────

class TestFailureMarker:
    def test_failed_marker_written_on_exception(
        self, good_config_file: Path, monkeypatch, tmp_path: Path
    ):
        def explode(*args, **kw):
            raise RuntimeError("boom")
        monkeypatch.setattr(run_mod, "run_simulation", explode)

        out_dir = tmp_path / "runs" / "run_0042"
        rc = run_mod.main([
            "--config", str(good_config_file),
            "--out-dir", str(out_dir),
        ])
        assert rc == 1
        marker = out_dir.parent / "run_0042.FAILED"
        assert marker.is_file()

        payload = json.loads(marker.read_text())
        assert payload["run_id"] == 42
        assert payload["params"]["seed"] == 2
        assert "boom" in payload["exception"]
        assert "Traceback" in payload["traceback"]

    def test_existing_failed_marker_cleared_on_retry(
        self, good_config_file: Path, captured_run_simulation, tmp_path: Path
    ):
        out_dir = tmp_path / "runs" / "run_0042"
        out_dir.parent.mkdir(parents=True)
        marker = out_dir.parent / "run_0042.FAILED"
        marker.write_text('{"stale": true}')
        rc = run_mod.main([
            "--config", str(good_config_file),
            "--out-dir", str(out_dir),
        ])
        # Successful re-run wipes the stale marker.
        assert rc == 0
        assert not marker.exists()


# ─── _parse_args plumbing ────────────────────────────────────────────────────

class TestParseArgs:
    def test_required_flags(self, capsys: pytest.CaptureFixture):
        with pytest.raises(SystemExit):
            run_mod._parse_args([])

    def test_config_must_be_supplied(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        with pytest.raises(SystemExit):
            run_mod._parse_args(["--out-dir", str(tmp_path)])

    def test_all_recognised_overrides(self, tmp_path: Path):
        ns = run_mod._parse_args([
            "--config", "x.json", "--out-dir", str(tmp_path),
            "--spinup-days", "2.0",
            "--data-days", "5.0",
            "--snapshot-interval-days", "0.5",
            "--delta-t-atm", "300.0",
            "--delta-t-ocn", "1800.0",
            "--cpl-send-freq", "1800.0",
            "--mpirun", "srun",
            "--inputs-root", "/tmp/inputs",
        ])
        assert ns.spinup_days == 2.0
        assert ns.delta_t_ocn == 1800.0
        assert ns.cpl_atm_send_freq_s == 1800.0
        assert ns.mpirun == "srun"
        assert ns.inputs_root == Path("/tmp/inputs")
