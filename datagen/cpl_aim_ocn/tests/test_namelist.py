"""Tests for ``datagen/cpl_aim_ocn/namelist.py``.

Three tiers:

1. Low-level unit tests on the ``_fmt`` Fortran value formatter and
   the ``_set`` block-aware key substituter.
2. Per-file ``render_*`` tests: confirm each writer emits valid Fortran
   text that contains the expected keys with correct values, and that
   sweep parameters round-trip through float formatting.
3. ``render_phase_namelists`` integration tests: full dict shape, the
   spin-up vs data-phase diff (``useDiagnostics`` toggle, ``dumpFreq``,
   ``pickupSuff``).

All tests are pure-Python — no MITgcm binary or MPI required.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_namelist.py -v
"""

from __future__ import annotations

import re

import pytest

from datagen.cpl_aim_ocn import namelist as nm
from datagen.cpl_aim_ocn.namelist import (
    PhaseTimeConfig,
    SweepParams,
    _fmt,
    _set,
    render_atm_aimphys,
    render_atm_cpl,
    render_atm_data,
    render_atm_diagnostics,
    render_atm_eedata,
    render_atm_ice,
    render_atm_land,
    render_atm_pkg,
    render_atm_shap,
    render_cpl_data,
    render_ocn_cpl,
    render_ocn_data,
    render_ocn_diagnostics,
    render_ocn_eedata,
    render_ocn_gmredi,
    render_ocn_pkg,
    render_phase_namelists,
)


# ─── _fmt: Fortran value formatting ──────────────────────────────────────────

class TestFmt:
    def test_bool_true(self):
        assert _fmt(True) == ".TRUE."

    def test_bool_false(self):
        assert _fmt(False) == ".FALSE."

    def test_int(self):
        assert _fmt(0) == "0"
        assert _fmt(72000) == "72000"
        assert _fmt(-3) == "-3"

    def test_float_keeps_decimal_point(self):
        # Critical: Fortran lexer needs ".", otherwise 0 → INT not REAL.
        assert "." in _fmt(0.0)
        assert "." in _fmt(1.0)

    def test_float_scientific(self):
        # Large or small magnitudes use exponent form, which is fine.
        s = _fmt(1.0e20)
        assert "e" in s.lower()

    def test_string_single_quoted(self):
        assert _fmt("foo.bin") == "'foo.bin'"

    def test_string_with_apostrophe_rejected(self):
        with pytest.raises(ValueError, match="apostrophe"):
            _fmt("don't")

    def test_unknown_type_rejected(self):
        with pytest.raises(TypeError, match="value type"):
            _fmt(object())

    def test_round_trip_typical_co2_value(self):
        # 348.0 is a representative CO2 value in our sweep; verify
        # the formatter output parses cleanly back to 348.
        s = _fmt(348.0)
        assert float(s) == 348.0


# ─── _set: block-aware key substitution ──────────────────────────────────────

_SAMPLE_NL = """\
# A toy namelist file
 &PARM01
  alpha = 1,
  beta = 2.5,
# this comment should NOT be touched
 &

 &PARM02
  gamma = 'old.bin',
 &
"""


class TestSet:
    def test_replaces_existing_int_key(self):
        out = _set(_SAMPLE_NL, "PARM01", "alpha", 99)
        assert "alpha=99" in out
        # And the original line is gone.
        assert "alpha = 1," not in out
        # Other keys untouched.
        assert "beta = 2.5," in out

    def test_replaces_existing_float_key(self):
        out = _set(_SAMPLE_NL, "PARM01", "beta", 3.14)
        assert re.search(r"beta=3\.14\b", out), out

    def test_replaces_existing_string_key(self):
        out = _set(_SAMPLE_NL, "PARM02", "gamma", "new.bin")
        assert "gamma='new.bin'" in out
        assert "old.bin" not in out

    def test_inserts_new_key_before_close(self):
        out = _set(_SAMPLE_NL, "PARM01", "delta", 7)
        assert "delta=7" in out
        # Make sure it's inside the &PARM01 block (before its closing &).
        block_start = out.index("&PARM01")
        block_end = out.index(" &", block_start + 1)
        assert "delta=7" in out[block_start:block_end]

    def test_block_not_found_raises(self):
        with pytest.raises(ValueError, match="MISSING"):
            _set(_SAMPLE_NL, "MISSING", "x", 1)

    def test_does_not_touch_comments(self):
        # Even if a comment line happens to contain `alpha = 1`,
        # the function should NOT treat it as the active key.
        text = """\
 &PARM01
# alpha = 1,
  alpha = 5,
 &
"""
        out = _set(text, "PARM01", "alpha", 99)
        assert "# alpha = 1," in out  # comment preserved
        assert "alpha=99" in out
        assert "alpha = 5," not in out

    def test_idempotent_under_repeated_sets(self):
        a = _set(_SAMPLE_NL, "PARM01", "alpha", 42)
        b = _set(a, "PARM01", "alpha", 42)
        assert a == b

    def test_replaces_last_occurrence_when_duplicated(self):
        # Upstream MITgcm files sometimes have the same key written
        # twice (e.g. the atm `monitorFreq` line). Fortran takes the
        # last one as authoritative; our patcher must too.
        text = """\
 &PARM03
  monitorFreq = 86400.,
  monitorFreq = 1.,
 &
"""
        out = _set(text, "PARM03", "monitorFreq", 3600.0)
        # The first line is preserved; the second (active) is replaced.
        assert "monitorFreq = 86400." in out
        assert "monitorFreq=3600" in out
        assert "monitorFreq = 1." not in out


# ─── Atmosphere: render_atm_data ─────────────────────────────────────────────

class TestRenderAtmData:
    def test_sets_n_iter0_and_n_timesteps(self):
        out = render_atm_data(
            n_iter0=72000, n_timesteps=40, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        assert "nIter0=72000" in out
        assert "nTimeSteps=40" in out

    def test_default_delta_t_is_450(self):
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        # 450 → "450" with our :.7g formatter (becomes 450 with appended ".")
        assert re.search(r"deltaT=450\.?\b", out), out

    def test_pchkptfreq_equals_run_length_when_writing_pickup(self):
        # 10 steps × 450 s = 4500 s
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=True,
            snapshot_interval_s=None,
        )
        m = re.search(r"pChkptFreq=([\d\.eE+\-]+)", out)
        assert m is not None
        assert abs(float(m.group(1)) - 4500.0) < 1.0

    def test_pchkptfreq_huge_when_not_writing_pickup(self):
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        m = re.search(r"pChkptFreq=([\d\.eE+\-]+)", out)
        assert m is not None
        # ≥ 1e19: longer than any plausible run.
        assert float(m.group(1)) > 1.0e19

    def test_dumpfreq_zero_when_no_diagnostics(self):
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        assert re.search(r"dumpFreq=0\.?\b", out), out

    def test_dumpfreq_equals_snapshot_interval(self):
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=86400.0,
        )
        assert re.search(r"dumpFreq=86400\.?\b", out), out
        assert re.search(r"monitorFreq=86400\.?\b", out), out

    def test_pickup_suff_inserted_when_provided(self):
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None, pickup_suff="0000007200",
        )
        assert "pickupSuff='0000007200'" in out

    def test_pickup_suff_absent_when_none(self):
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        assert "pickupSuff" not in out

    def test_hydrog_theta_file_inserted(self):
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None, hydrog_theta_file="theta_pert.bin",
        )
        assert "hydrogThetaFile='theta_pert.bin'" in out


# ─── Atmosphere: render_atm_aimphys ──────────────────────────────────────────

class TestRenderAtmAimphys:
    def test_co2_select_set_to_1(self):
        out = render_atm_aimphys(co2_ppm=348.0, solar_const_w_m2=342.0)
        assert "aim_select_pCO2=1" in out

    def test_co2_ppm_converts_to_aim_mole_fraction(self):
        for ppm in (280.0, 348.0, 560.0, 1120.0):
            out = render_atm_aimphys(co2_ppm=ppm, solar_const_w_m2=342.0)
            m = re.search(r"aim_fixed_pCO2=([\d\.eE+\-]+)", out)
            assert m is not None
            assert abs(float(m.group(1)) - ppm * 1e-6) < 1e-12

    def test_solar_value_round_trips(self):
        for s in (331.74, 342.0, 352.26):
            out = render_atm_aimphys(co2_ppm=348.0, solar_const_w_m2=s)
            m = re.search(r"\bSOLC=([\d\.eE+\-]+)", out)
            assert m is not None
            assert abs(float(m.group(1)) - s) < 1e-3

    def test_solc_inserted_into_aim_par_for_block(self):
        # The upstream &AIM_PAR_FOR block is empty; SOLC must be
        # inserted there (not into &AIM_PARAMS).
        out = render_atm_aimphys(co2_ppm=348.0, solar_const_w_m2=342.0)
        # Find the AIM_PAR_FOR block boundaries.
        start = out.index("&AIM_PAR_FOR")
        end = out.index(" &", start + 1)
        assert "SOLC=" in out[start:end]

    def test_reference_like_values_render_in_aim_units(self):
        out = render_atm_aimphys(co2_ppm=320.0, solar_const_w_m2=342.0)
        assert "aim_fixed_pCO2=0.00032" in out
        assert re.search(r"\bSOLC=342\.?\b", out), out


# ─── Atmosphere: render_atm_pkg + render_atm_diagnostics ─────────────────────

class TestRenderAtmPkg:
    def test_diagnostics_toggle_on(self):
        out = render_atm_pkg(use_diagnostics=True)
        assert "useDiagnostics=.TRUE." in out

    def test_diagnostics_toggle_off(self):
        out = render_atm_pkg(use_diagnostics=False)
        assert "useDiagnostics=.FALSE." in out

    def test_useaim_kept_true(self):
        # Sanity: we don't accidentally turn off the AIM physics package.
        out = render_atm_pkg(use_diagnostics=True)
        assert "useAIM=.TRUE." in out


class TestRenderAtmDiagnostics:
    def test_one_stream_per_field(self):
        # We emit ONE stream per field (e.g. fileName(N) = 'atm_TS').
        # Multi-field streams break xmitgcm's CS reader.
        out = render_atm_diagnostics(snapshot_interval_s=86400.0)
        for v in ("TS", "QS", "PRECON", "PRECLS", "WINDS",
                  "UFLUX", "VFLUX", "SI_Fract", "SI_Thick",
                  "UVEL", "VVEL", "THETA", "SALT"):
            assert f"'atm_{v}'" in out, f"stream atm_{v} missing"

    def test_no_legacy_combined_streams(self):
        # The early version of this file used combined `atm_2d` +
        # `atm_3d` streams — that's the multi-field shape xmitgcm
        # rejects. Guard against accidental regression.
        out = render_atm_diagnostics(snapshot_interval_s=86400.0)
        assert "'atm_2d'" not in out
        assert "'atm_3d'" not in out

    def test_frequency_is_negative_for_instantaneous_snapshots(self):
        for freq in (3600.0, 86400.0):
            out = render_atm_diagnostics(snapshot_interval_s=freq)
            # Every active frequency line should carry the negative value.
            assert f"frequency(1) = {nm._fmt(-freq)}" in out

    def test_no_invalid_aim_names(self):
        # Past bug: namelist.py used `aim_T2m`, `aim_TPCP`, etc., which
        # are NOT registered as MITgcm diagnostics — they crash the
        # binary with `DIAGNOSTICS_SET_POINTERS`. Guard against it.
        out = render_atm_diagnostics(snapshot_interval_s=86400.0)
        for bad in ("aim_T2m", "aim_TPCP", "aim_USTR", "aim_VSTR",
                    "SIarea", "SIheff"):
            assert bad not in out, f"invalid diagnostic name {bad!r} in atm_diags"


# ─── Atmosphere: verbatim renderers ──────────────────────────────────────────

class TestVerbatimAtmRenderers:
    def test_data_land_returns_template(self):
        out = render_atm_land()
        assert "&LAND_PARAM" in out or "LAND" in out

    def test_data_ice_returns_template(self):
        out = render_atm_ice()
        assert "&THSICE" in out

    def test_data_shap_returns_template(self):
        out = render_atm_shap()
        assert "&SHAP_PARM01" in out

    def test_atm_eedata_has_coupler_flags(self):
        out = render_atm_eedata()
        assert "useCoupler=.TRUE." in out
        assert "useCubedSphereExchange=.TRUE." in out

    def test_atm_data_enables_single_cpu_io(self):
        # ``useSingleCpuIO`` lives in PARM01 of ``data`` — not in
        # ``eedata`` (a common gotcha). Required so xmitgcm's
        # geometry='cs' reader can consume the output.
        out = render_atm_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        assert "useSingleCpuIO=.TRUE." in out

    def test_ocn_data_enables_single_cpu_io(self):
        out = render_ocn_data(
            n_iter0=0, n_timesteps=10, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        assert "useSingleCpuIO=.TRUE." in out

    def test_atm_cpl_sets_send_frequency(self):
        out = render_atm_cpl(cpl_atm_send_freq_s=1800.0)
        m = re.search(r"cpl_atmSendFrq=([\d\.eE+\-]+)", out)
        assert m is not None
        assert abs(float(m.group(1)) - 1800.0) < 1e-6


# ─── Ocean: render_ocn_data ──────────────────────────────────────────────────

class TestRenderOcnData:
    def test_three_timestep_keys_set_together(self):
        out = render_ocn_data(
            n_iter0=72000, n_timesteps=720, write_pickup_at_end=False,
            snapshot_interval_s=86400.0,
        )
        for k in ("deltaTmom", "deltaTtracer", "deltaTClock"):
            assert re.search(rf"\b{k}=3600\.?\b", out), f"{k} missing/wrong"

    def test_pickup_suff_targets_global_ocean_baseline(self):
        out = render_ocn_data(
            n_iter0=72000, n_timesteps=720, write_pickup_at_end=False,
            snapshot_interval_s=None, pickup_suff="0000072000",
        )
        assert "pickupSuff='0000072000'" in out

    def test_n_iter0_set_correctly(self):
        out = render_ocn_data(
            n_iter0=72000, n_timesteps=720, write_pickup_at_end=False,
            snapshot_interval_s=None,
        )
        assert "nIter0=72000" in out


# ─── Ocean: GM-Redi / pkg / diagnostics / cpl ─────────────────────────────

class TestRenderOcnGmredi:
    def test_kappa_round_trip(self):
        for k in (500.0, 1000.0, 2000.0):
            out = render_ocn_gmredi(gm_kappa=k)
            for key in ("GM_background_K", "GM_isopycK"):
                m = re.search(rf"{key}=([\d\.eE+\-]+)", out)
                assert m is not None, f"{key} not set in {out}"
                assert abs(float(m.group(1)) - k) < 1e-3


class TestRenderOcnPkg:
    def test_diagnostics_toggle(self):
        on = render_ocn_pkg(use_diagnostics=True)
        off = render_ocn_pkg(use_diagnostics=False)
        assert "useDiagnostics=.TRUE." in on
        assert "useDiagnostics=.FALSE." in off

    def test_gmredi_left_on(self):
        out = render_ocn_pkg(use_diagnostics=True)
        assert "useGMRedi=.TRUE." in out


class TestRenderOcnDiagnostics:
    def test_one_stream_per_surface_field(self):
        out = render_ocn_diagnostics(snapshot_interval_s=86400.0)
        for v in ("THETA", "SALT", "UVEL", "VVEL", "ETAN", "MXLDEPTH"):
            assert f"'ocn_{v}'" in out, f"stream ocn_{v} missing"

    def test_drops_upstream_dyndiag(self):
        # We deliberately drop the upstream's full-3D dynDiag stream
        # (15 levels × 15 vars at cs32 ≈ 5 MB / snapshot, dominates
        # storage). Spec calls for surface ocean only.
        out = render_ocn_diagnostics(snapshot_interval_s=86400.0)
        assert "dynDiag" not in out
        assert "surfDiag" not in out  # also dropped — we generate fresh

    def test_frequency_is_negative_for_instantaneous_snapshots(self):
        for freq in (3600.0, 86400.0):
            out = render_ocn_diagnostics(snapshot_interval_s=freq)
            assert f"frequency(1) = {nm._fmt(-freq)}" in out

    def test_levels_subset_to_surface_for_3d_fields(self):
        # The 3-D fields THETA/SALT/UVEL/VVEL are sliced to level 1 via
        # ``levels(1,N) = 1.`` so we only write the surface cell.
        out = render_ocn_diagnostics(snapshot_interval_s=86400.0)
        for n in (1, 2, 3, 4):  # 4 3-D fields
            assert f"levels(1,{n}) = 1." in out


class TestRenderOcnCpl:
    def test_returns_verbatim_template(self):
        out = render_ocn_cpl()
        assert "&CPL_OCN_PARAM" in out

    def test_use_import_slp_kept_false(self):
        # The only explicit setting upstream — must persist.
        out = render_ocn_cpl()
        assert "useImportSLP" in out


# ─── Coupler: render_cpl_data ─────────────────────────────────────────────

class TestRenderCplData:
    def test_returns_verbatim(self):
        out = render_cpl_data()
        assert "&COUPLER_PARAMS" in out

    def test_runoff_map_preserved(self):
        out = render_cpl_data()
        # The runOff routing map filename must be preserved (we ship
        # it in inputs/cpl/runOff_cs32_3644.bin).
        assert "runOff_cs32_3644.bin" in out


# ─── render_phase_namelists: full dict integration ──────────────────────────

class TestRenderPhaseNamelists:
    @pytest.fixture
    def sweep(self) -> SweepParams:
        return SweepParams(co2_ppm=560.0, solar_scale=1.03,
                           gm_kappa=2000.0, seed=0)

    def test_returns_three_components_with_expected_files(self, sweep):
        cfg = PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                              snapshot_interval_s=None)
        nls = render_phase_namelists(time_cfg=cfg, sweep=sweep)
        assert set(nls) == {"cpl", "ocn", "atm"}

        # Coupler always has data.cpl + eedata.
        assert set(nls["cpl"]) >= {"data.cpl", "eedata"}
        # Ocean spin-up: data, data.pkg, data.gmredi, data.cpl, eedata.
        assert {"data", "data.pkg", "data.gmredi",
                "data.cpl", "eedata"} <= set(nls["ocn"])
        # Atm spin-up: data, data.aimphys, data.pkg, data.land,
        # data.ice, data.shap, data.cpl, eedata.
        assert {"data", "data.aimphys", "data.pkg", "data.land",
                "data.ice", "data.shap", "data.cpl",
                "eedata"} <= set(nls["atm"])

    def test_spinup_omits_diagnostics_files(self, sweep):
        cfg = PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                              snapshot_interval_s=None)
        nls = render_phase_namelists(time_cfg=cfg, sweep=sweep)
        assert "data.diagnostics" not in nls["atm"]
        assert "data.diagnostics" not in nls["ocn"]

    def test_data_phase_includes_diagnostics_files(self, sweep):
        cfg = PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                              snapshot_interval_s=86400.0)
        nls = render_phase_namelists(time_cfg=cfg, sweep=sweep)
        assert "data.diagnostics" in nls["atm"]
        assert "data.diagnostics" in nls["ocn"]

    def test_diagnostics_toggle_consistent_with_phase(self, sweep):
        spinup = render_phase_namelists(
            time_cfg=PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                                     snapshot_interval_s=None),
            sweep=sweep,
        )
        data = render_phase_namelists(
            time_cfg=PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                                     snapshot_interval_s=86400.0),
            sweep=sweep,
        )
        assert "useDiagnostics=.FALSE." in spinup["atm"]["data.pkg"]
        assert "useDiagnostics=.TRUE."  in data["atm"]["data.pkg"]
        assert "useDiagnostics=.FALSE." in spinup["ocn"]["data.pkg"]
        assert "useDiagnostics=.TRUE."  in data["ocn"]["data.pkg"]

    def test_sweep_values_appear_in_correct_files(self, sweep):
        cfg = PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                              snapshot_interval_s=86400.0)
        nls = render_phase_namelists(time_cfg=cfg, sweep=sweep)
        # CO2 + solar in atm/data.aimphys
        assert "aim_fixed_pCO2=0.00056" in nls["atm"]["data.aimphys"]
        # 1.03 * 342 = 352.26
        m = re.search(r"\bSOLC=([\d\.eE+\-]+)",
                      nls["atm"]["data.aimphys"])
        assert m is not None and abs(float(m.group(1)) - 352.26) < 1e-3
        # GM kappa in ocn/data.gmredi
        assert "GM_background_K=2000" in nls["ocn"]["data.gmredi"]

    def test_run0000_like_values_render_in_aim_units(self):
        cfg = PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                              snapshot_interval_s=86400.0)
        sweep = SweepParams(co2_ppm=280.0, solar_scale=0.97,
                            gm_kappa=500.0, seed=0)
        nls = render_phase_namelists(time_cfg=cfg, sweep=sweep)
        assert "aim_fixed_pCO2=0.00028" in nls["atm"]["data.aimphys"]
        m = re.search(r"\bSOLC=([\d\.eE+\-]+)",
                      nls["atm"]["data.aimphys"])
        assert m is not None and abs(float(m.group(1)) - 331.74) < 1e-3

    def test_reference_like_sweep_values_match_upstream_units(self):
        cfg = PhaseTimeConfig(n_atm_steps=40, n_ocn_steps=5,
                              snapshot_interval_s=86400.0)
        sweep = SweepParams(co2_ppm=320.0, solar_scale=1.0,
                            gm_kappa=800.0, seed=0)
        nls = render_phase_namelists(time_cfg=cfg, sweep=sweep)
        assert "aim_fixed_pCO2=0.00032" in nls["atm"]["data.aimphys"]
        assert re.search(r"\bSOLC=342\.?\b", nls["atm"]["data.aimphys"])
        assert "GM_background_K=800." in nls["ocn"]["data.gmredi"]
        assert "GM_isopycK=800." in nls["ocn"]["data.gmredi"]

    def test_pickup_handling_phase1_ocn_uses_baseline(self, sweep):
        # Phase 1 spin-up: ocean restarts from the global_ocean.cs32x15
        # baseline pickup at iter 72000.
        cfg = PhaseTimeConfig(
            n_atm_steps=40, n_ocn_steps=5, snapshot_interval_s=None,
            pickup_suff_ocn="0000072000",
        )
        nls = render_phase_namelists(time_cfg=cfg, sweep=sweep)
        assert "pickupSuff='0000072000'" in nls["ocn"]["data"]
        # And nIter0 matches the suffix.
        assert "nIter0=72000" in nls["ocn"]["data"]


# ─── SweepParams ─────────────────────────────────────────────────────────────

class TestSweepParams:
    def test_solar_const_derived_correctly(self):
        s = SweepParams(co2_ppm=348.0, solar_scale=1.00,
                        gm_kappa=1000.0, seed=0)
        assert abs(s.solar_const_w_m2 - 342.0) < 1e-9
        s2 = SweepParams(co2_ppm=348.0, solar_scale=0.97,
                         gm_kappa=1000.0, seed=0)
        assert abs(s2.solar_const_w_m2 - 0.97 * 342.0) < 1e-9
