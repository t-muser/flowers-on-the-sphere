"""Tests for ``datagen/cpl_aim_ocn/solver.py``.

The solver is the orchestrator: it builds RunConfig + SimulationParams,
writes the seeded atm IC, calls the namelist + staging modules, runs
the MPMD launch, and writes the Zarr. We test it by:

1. Pure-Python tests on the dataclass derivations (``RunConfig``
   properties, ``SimulationParams.from_dict``).
2. Tests on the phase-1 / phase-2 namelist construction helpers
   (``_phase_1_namelists`` / ``_phase_2_namelists``) — confirms the
   phase distinction (diagnostics on/off, pickup handling).
3. Integration tests on ``run_simulation`` with **monkeypatched**
   launch and zarr-write helpers, so the orchestration sequence is
   exercised end-to-end without actually invoking ``mpirun``.

All tests are pure-Python — no MITgcm binary or MPI required.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_solver.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datagen.cpl_aim_ocn import solver as solver_mod
from datagen.cpl_aim_ocn.solver import (
    RunConfig,
    SimulationParams,
    _phase_1_namelists,
    _phase_2_namelists,
    _write_atm_ic,
    run_simulation,
)


# ─── RunConfig: derived properties ───────────────────────────────────────────

class TestRunConfigDerivations:
    def test_step_counts_for_default_30d_spinup(self):
        cfg = RunConfig(spinup_days=30.0, delta_t_atm=450.0,
                        delta_t_ocn=3600.0)
        # 30 d × 86400 s / 450 s = 5760 atm steps
        assert cfg.n_atm_steps_spinup == 5760
        # 30 d × 86400 s / 3600 s = 720 ocn steps
        assert cfg.n_ocn_steps_spinup == 720

    def test_step_counts_for_default_365d_data(self):
        cfg = RunConfig(data_days=365.0)
        # 365 × 86400 / 450 = 70080 atm steps
        assert cfg.n_atm_steps_data == 70080
        # 365 × 86400 / 3600 = 8760 ocn steps
        assert cfg.n_ocn_steps_data == 8760

    def test_seconds_helpers(self):
        cfg = RunConfig(spinup_days=2.5, data_days=10.0,
                        snapshot_interval_days=0.5)
        assert cfg.spinup_seconds == 2.5 * 86400.0
        assert cfg.data_seconds == 10.0 * 86400.0
        assert cfg.snapshot_interval_s == 0.5 * 86400.0

    def test_timeout_has_60s_floor(self):
        # Even a 1-second run gets ≥60 s wall budget for startup.
        cfg = RunConfig(timeout_factor=3.0)
        assert cfg.timeout_for_phase_s(1.0) == 60.0

    def test_timeout_scales_with_factor(self):
        cfg = RunConfig(timeout_factor=2.0)
        assert cfg.timeout_for_phase_s(1000.0) == 2000.0

    def test_default_baseline_pickup_metadata_consistent(self):
        cfg = RunConfig()
        assert cfg.pickup_suff_baseline_ocn == "0000072000"
        assert cfg.n_iter0_baseline_ocn == 72000
        # Suffix-as-int matches the explicit baseline iter.
        assert int(cfg.pickup_suff_baseline_ocn) == cfg.n_iter0_baseline_ocn


# ─── SimulationParams ────────────────────────────────────────────────────────

class TestSimulationParams:
    def test_from_dict_extracts_core_keys(self):
        sim = SimulationParams.from_dict({
            "co2_ppm":     348.0,
            "solar_scale": 1.00,
            "gm_kappa":    1000.0,
            "seed":        7,
            "spinup_days": 30.0,    # extra key — ignored at this level
        })
        assert sim.co2_ppm == 348.0
        assert sim.solar_scale == 1.00
        assert sim.gm_kappa == 1000.0
        assert sim.seed == 7

    def test_from_dict_seed_is_int_even_if_passed_as_float(self):
        sim = SimulationParams.from_dict({
            "co2_ppm": 348.0, "solar_scale": 1.0, "gm_kappa": 1000.0,
            "seed": 3.0,
        })
        assert sim.seed == 3 and isinstance(sim.seed, int)

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(KeyError):
            SimulationParams.from_dict({"co2_ppm": 348.0})

    def test_as_sweep_round_trips(self):
        sim = SimulationParams(co2_ppm=560.0, solar_scale=1.03,
                               gm_kappa=2000.0, seed=2)
        sweep = sim.as_sweep()
        assert sweep.co2_ppm == sim.co2_ppm
        assert sweep.solar_scale == sim.solar_scale
        assert sweep.gm_kappa == sim.gm_kappa
        assert sweep.seed == sim.seed
        # And the derived AIM area-mean solar constant scales correctly.
        assert abs(sweep.solar_const_w_m2 - 1.03 * 342.0) < 1e-9


# ─── Phase namelist construction ─────────────────────────────────────────────

class TestPhaseNamelists:
    @pytest.fixture
    def sim(self) -> SimulationParams:
        return SimulationParams(co2_ppm=348.0, solar_scale=1.00,
                                gm_kappa=1000.0, seed=0)

    @pytest.fixture
    def cfg(self) -> RunConfig:
        return RunConfig(spinup_days=30.0, data_days=365.0,
                         snapshot_interval_days=1.0)

    def test_phase_1_uses_baseline_pickup_for_ocn(self, cfg, sim):
        nls = _phase_1_namelists(cfg, sim)
        assert "pickupSuff='0000072000'" in nls["ocn"]["data"]
        # nIter0 must match the baseline iteration count.
        assert "nIter0=72000" in nls["ocn"]["data"]

    def test_phase_1_atm_cold_starts(self, cfg, sim):
        nls = _phase_1_namelists(cfg, sim)
        assert "pickupSuff" not in nls["atm"]["data"]
        assert "nIter0=0" in nls["atm"]["data"]

    def test_phase_1_atm_loads_theta_perturbation(self, cfg, sim):
        nls = _phase_1_namelists(cfg, sim)
        assert "hydrogThetaFile='theta_pert.bin'" in nls["atm"]["data"]

    def test_phase_1_no_diagnostics(self, cfg, sim):
        nls = _phase_1_namelists(cfg, sim)
        assert "useDiagnostics=.FALSE." in nls["atm"]["data.pkg"]
        assert "useDiagnostics=.FALSE." in nls["ocn"]["data.pkg"]
        assert "data.diagnostics" not in nls["atm"]
        assert "data.diagnostics" not in nls["ocn"]

    def test_phase_1_writes_pickup_at_end(self, cfg, sim):
        nls = _phase_1_namelists(cfg, sim)
        # pChkptFreq should be ≈ spinup_seconds (write pickup once,
        # at the end). For 30 d that's 2 592 000 s.
        text = nls["atm"]["data"]
        assert "pChkptFreq=2592000" in text or "pChkptFreq=2.592e" in text.lower()

    def test_phase_2_restarts_from_phase_1_pickups(self, cfg, sim):
        nls = _phase_2_namelists(cfg, sim,
                                 pickup_suff_atm="0000005760",
                                 pickup_suff_ocn="0000072720")
        assert "pickupSuff='0000005760'" in nls["atm"]["data"]
        assert "pickupSuff='0000072720'" in nls["ocn"]["data"]

    def test_phase_2_enables_diagnostics(self, cfg, sim):
        nls = _phase_2_namelists(cfg, sim,
                                 pickup_suff_atm="0000005760",
                                 pickup_suff_ocn="0000072720")
        assert "useDiagnostics=.TRUE." in nls["atm"]["data.pkg"]
        assert "useDiagnostics=.TRUE." in nls["ocn"]["data.pkg"]
        # Diagnostics file present in both rank dirs.
        assert "data.diagnostics" in nls["atm"]
        assert "data.diagnostics" in nls["ocn"]

    def test_phase_2_does_not_rewrite_theta_perturb(self, cfg, sim):
        # The phase-1 pickup already contains the spun-up perturbed
        # state; rewriting hydrogThetaFile would override it.
        nls = _phase_2_namelists(cfg, sim,
                                 pickup_suff_atm="0000005760",
                                 pickup_suff_ocn="0000072720")
        assert "hydrogThetaFile" not in nls["atm"]["data"]


# ─── _write_atm_ic ───────────────────────────────────────────────────────────

class TestWriteAtmIc:
    def test_writes_into_rank_2(self, tmp_path: Path):
        sim = SimulationParams(co2_ppm=348.0, solar_scale=1.0,
                               gm_kappa=1000.0, seed=42)
        run_dir = tmp_path / "run"
        fname = _write_atm_ic(run_dir, sim)
        # Returned filename matches what the namelist expects.
        assert fname == "theta_pert.bin"
        assert (run_dir / "rank_2" / fname).is_file()
        assert (run_dir / "rank_2" / "theta_pert.meta").is_file()

    def test_different_seeds_produce_different_files(self, tmp_path: Path):
        run_a = tmp_path / "a"
        run_b = tmp_path / "b"
        _write_atm_ic(
            run_a, SimulationParams(co2_ppm=348., solar_scale=1.,
                                    gm_kappa=1000., seed=0)
        )
        _write_atm_ic(
            run_b, SimulationParams(co2_ppm=348., solar_scale=1.,
                                    gm_kappa=1000., seed=1)
        )
        assert (run_a / "rank_2" / "theta_pert.bin").read_bytes() \
            != (run_b / "rank_2" / "theta_pert.bin").read_bytes()


# ─── run_simulation: orchestration with monkeypatched launches ───────────────

@pytest.fixture
def fake_inputs(tmp_path: Path) -> Path:
    """Stub inputs/ tree (per-component subdirs with a couple of files
    each) so stage_run's _symlink_dir_contents has something to walk."""
    root = tmp_path / "inputs"
    for sub, files in (
        ("atm",  ["albedo_cs32.bin", "topo.cpl_FM.bin"]),
        ("ocn",  ["bathy_Hmin50.bin", "lev_T_cs_15k.bin",
                  "pickup.0000072000", "pickup.0000072000.meta"]),
        ("grid", [f"grid_cs32.face00{i}.bin" for i in range(1, 7)]),
        ("cpl",  ["RA.bin", "runOff_cs32_3644.bin"]),
    ):
        d = root / sub
        d.mkdir(parents=True)
        for n in files:
            (d / n).write_bytes(n.encode())
    return root


@pytest.fixture
def fake_build_dirs(tmp_path: Path) -> dict[str, Path]:
    """Stub build_<X>/mitgcmuv files so the pre-flight check passes."""
    out = {}
    for c in ("cpl", "ocn", "atm"):
        d = tmp_path / f"build_{c}"
        d.mkdir()
        ex = d / "mitgcmuv"
        ex.write_bytes(b"#!/bin/sh\n")
        ex.chmod(0o755)
        out[c] = d
    return out


@pytest.fixture
def run_simulation_env(monkeypatch, fake_inputs, fake_build_dirs):
    """Replace the two side-effecty subroutines (mpirun launch + Zarr
    write) with stubs so the orchestrator runs in-process.

    Yields a list of recorded launches so individual tests can assert
    on the call sequence.
    """
    launches: list[tuple[str, float]] = []

    def fake_launch(run_dir: Path, cfg: RunConfig, *,
                    phase_name: str, sim_seconds: float) -> None:
        launches.append((phase_name, sim_seconds))
        # Pretend the binaries also wrote successful rank logs so
        # check_run_completed (called from inside _launch_mpmd in real
        # code) wouldn't trigger. Since we replaced the entire
        # _launch_mpmd, we just need to avoid leaving the run_dir in
        # a broken state.

    zarr_calls: list[tuple[Path, Path]] = []

    def fake_zarr(run_dir: Path, out_path: Path, *, attrs=None,
                  **kwargs) -> object:
        zarr_calls.append((Path(run_dir), Path(out_path)))
        Path(out_path).mkdir(parents=True, exist_ok=True)
        return type("R", (), {
            "path": out_path, "n_time": 1,
            "atm_vars": (), "ocn_vars": (),
        })()

    monkeypatch.setattr(solver_mod, "_launch_mpmd", fake_launch)
    monkeypatch.setattr(solver_mod, "write_cs32_zarr", fake_zarr)

    yield launches, zarr_calls


def _make_cfg(fake_inputs, fake_build_dirs, **kw):
    return RunConfig(
        inputs_root=fake_inputs,
        build_dirs=fake_build_dirs,
        spinup_days=kw.pop("spinup_days", 1.0),
        data_days=kw.pop("data_days", 1.0),
        snapshot_interval_days=kw.pop("snapshot_interval_days", 1.0),
        **kw,
    )


class TestRunSimulationOrchestration:
    def test_runs_two_phases_in_order(
        self, run_simulation_env, fake_inputs, fake_build_dirs, tmp_path: Path
    ):
        launches, _ = run_simulation_env
        params = {"co2_ppm": 348.0, "solar_scale": 1.0,
                  "gm_kappa": 1000.0, "seed": 0}
        run_simulation(
            params, tmp_path / "out",
            config=_make_cfg(fake_inputs, fake_build_dirs),
        )
        # Two launches, in canonical order.
        assert [name for name, _ in launches] == ["phase1", "phase2"]

    def test_writes_atm_ic_before_phase_1(
        self, run_simulation_env, fake_inputs, fake_build_dirs, tmp_path: Path
    ):
        params = {"co2_ppm": 348.0, "solar_scale": 1.0,
                  "gm_kappa": 1000.0, "seed": 5}
        out = tmp_path / "out"
        run_simulation(
            params, out,
            config=_make_cfg(fake_inputs, fake_build_dirs),
        )
        # The IC file should exist by the end of the run.
        assert (out / "mitgcm_run" / "rank_2" / "theta_pert.bin").is_file()

    def test_writes_zarr_at_canonical_path(
        self, run_simulation_env, fake_inputs, fake_build_dirs, tmp_path: Path
    ):
        _, zarr_calls = run_simulation_env
        params = {"co2_ppm": 348.0, "solar_scale": 1.0,
                  "gm_kappa": 1000.0, "seed": 0}
        out = tmp_path / "out"
        result = run_simulation(
            params, out,
            config=_make_cfg(fake_inputs, fake_build_dirs),
        )
        assert result == out / "run.zarr"
        # Zarr writer was called with the expected paths.
        assert zarr_calls[0] == (out / "mitgcm_run", out / "run.zarr")

    def test_overrides_apply_to_runconfig(
        self, run_simulation_env, fake_inputs, fake_build_dirs, tmp_path: Path
    ):
        launches, _ = run_simulation_env
        params = {"co2_ppm": 348.0, "solar_scale": 1.0,
                  "gm_kappa": 1000.0, "seed": 0}
        run_simulation(
            params, tmp_path / "out",
            config=_make_cfg(fake_inputs, fake_build_dirs),
            spinup_days=0.5,
        )
        # phase1 sim_seconds == 0.5 days × 86400.
        assert launches[0] == ("phase1", 0.5 * 86400.0)

    def test_phase_2_restart_pickup_suffixes_match_step_counts(
        self, monkeypatch, fake_inputs, fake_build_dirs, tmp_path: Path
    ):
        # Capture the namelists handed to stage_run; verify phase 2
        # uses pickupSuff that equals (n_iter0_baseline + spinup_steps).
        captured: list[dict] = []

        from datagen.cpl_aim_ocn import staging as stg
        original = stg.stage_run
        def capture(run_dir, *, namelists, **kw):
            captured.append(namelists)
            return original(run_dir, namelists=namelists, **kw)

        monkeypatch.setattr(solver_mod, "stage_run", capture)
        monkeypatch.setattr(solver_mod, "_launch_mpmd",
                            lambda *args, **kw: None)
        monkeypatch.setattr(solver_mod, "write_cs32_zarr",
                            lambda *args, **kw:
                                type("R", (), {"path": kw["out_path"]
                                              if "out_path" in kw
                                              else args[1]})())

        params = {"co2_ppm": 348.0, "solar_scale": 1.0,
                  "gm_kappa": 1000.0, "seed": 0}
        cfg = _make_cfg(fake_inputs, fake_build_dirs,
                        spinup_days=30.0, data_days=1.0)
        run_simulation(params, tmp_path / "out", config=cfg)

        # Phase 2 was the second stage_run call.
        phase2_atm = captured[1]["atm"]["data"]
        phase2_ocn = captured[1]["ocn"]["data"]
        # 30 d × 86400 / 450 = 5760 atm steps
        assert "pickupSuff='0000005760'" in phase2_atm
        # baseline 72000 + 30 d × 86400 / 3600 = 72720 ocn steps
        assert "pickupSuff='0000072720'" in phase2_ocn

    def test_missing_executable_raises_before_launch(
        self, run_simulation_env, fake_inputs, fake_build_dirs, tmp_path: Path
    ):
        launches, _ = run_simulation_env
        # Remove one binary.
        (fake_build_dirs["atm"] / "mitgcmuv").unlink()
        params = {"co2_ppm": 348.0, "solar_scale": 1.0,
                  "gm_kappa": 1000.0, "seed": 0}
        with pytest.raises(FileNotFoundError, match="atm"):
            run_simulation(
                params, tmp_path / "out",
                config=_make_cfg(fake_inputs, fake_build_dirs),
            )
        # And no phase actually launched.
        assert launches == []

    def test_attrs_propagated_to_zarr_call(
        self, monkeypatch, fake_inputs, fake_build_dirs, tmp_path: Path
    ):
        recorded_attrs: list[dict] = []
        def fake_zarr(run_dir, out_path, *, attrs=None, **kw):
            recorded_attrs.append(attrs or {})
            Path(out_path).mkdir(parents=True, exist_ok=True)
            return type("R", (), {"path": out_path})()
        monkeypatch.setattr(solver_mod, "_launch_mpmd",
                            lambda *a, **k: None)
        monkeypatch.setattr(solver_mod, "write_cs32_zarr", fake_zarr)

        params = {"co2_ppm": 560.0, "solar_scale": 1.03,
                  "gm_kappa": 2000.0, "seed": 3}
        run_simulation(
            params, tmp_path / "out",
            config=_make_cfg(fake_inputs, fake_build_dirs),
        )
        a = recorded_attrs[0]
        assert a["co2_ppm"] == 560.0
        assert a["solar_scale"] == 1.03
        assert abs(a["solar_const_w_m2"] - 1.03 * 342.0) < 1e-9
        assert a["gm_kappa"] == 2000.0
        assert a["seed"] == 3
        # `params` is a JSON-encoded copy of the input dict.
        assert json.loads(a["params"]) == params
