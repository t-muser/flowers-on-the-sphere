"""Tests for Fortran namelist file generation.

Verifies that the generated files contain the correct parameter values and
follow the Fortran namelist syntax that MITgcm expects. No MITgcm binary
is required.

Run::

    uv run --project datagen pytest datagen/mitgcm/tests/test_namelist.py -v
"""

from __future__ import annotations

import re

import pytest

from datagen.mitgcm._constants import (
    CP,
    P0,
    R_DRY,
    R_EARTH,
    HS_SIGMAB,
    HS_T0,
)
from datagen.mitgcm.namelist import (
    write_data,
    write_data_diagnostics,
    write_data_hs_forc,
    write_data_pkg,
    write_data_shap,
    write_eedata,
    write_all_namelists,
)


# ─── write_data ──────────────────────────────────────────────────────────────

class TestWriteData:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=True,
                   pchkpt_freq=600000.0, has_ic_file=True,
                   T0=HS_T0, delta_theta_z=10.0)
        assert p.exists()

    def test_contains_sphericalpolar_flag(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        assert "usingSphericalPolarGrid = .TRUE." in content

    def test_contains_pressure_coordinate_buoyancy(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        assert "buoyancyRelation = 'ATMOSPHERIC'" in content

    def test_contains_atmospheric_buoyancy(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        assert "ATMOSPHERIC" in content

    def test_contains_ideal_gas_eos(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        assert "eosType      = 'IDEALG'" in p.read_text()

    def test_held_suarez_stability_settings(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        assert "selectCoriScheme = 2" in content
        assert "hFacMin        = 1.0" in content

    def test_grid_spacing_correct(self, tmp_path):
        """delX should be 360/Nlon degrees, delY should be 180/Nlat degrees."""
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        assert "128*2.8125" in content   # 360/128
        assert "64*2.8125"  in content   # 180/64

    def test_timestep_in_file(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=450.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        assert "450" in p.read_text()

    def test_n_timesteps_in_file(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=28800, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        assert "28800" in p.read_text()

    def test_n_iter0_in_file(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=28800, n_timesteps=52560, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        assert "nIter0        = 28800" in content

    def test_write_pickup_true(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=True,
                   pchkpt_freq=600000.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        assert "writePickupAtEnd = .TRUE." in p.read_text()

    def test_write_pickup_false(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        assert "writePickupAtEnd = .FALSE." in p.read_text()

    def test_ic_file_referenced_when_requested(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=True,
                   T0=HS_T0, delta_theta_z=10.0)
        assert "hydrogThetaFile" in p.read_text()

    def test_ic_file_absent_when_not_requested(self, tmp_path):
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        assert "hydrogThetaFile" not in p.read_text()

    def test_tref_has_nr_values(self, tmp_path):
        """tRef must have exactly Nr comma-separated values."""
        Nr = 20
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=Nr, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        # Extract the tRef block.
        match = re.search(r"tRef\s*=\s*(.*?)(?=\n\s*[a-zA-Z])", content, re.DOTALL)
        assert match is not None, "tRef block not found in data file"
        values_str = match.group(1)
        values = [v.strip().rstrip(",") for v in re.split(r"[,\n]+", values_str)
                  if v.strip().rstrip(",")]
        assert len(values) == Nr, f"Expected {Nr} tRef values, got {len(values)}"

    def test_tref_is_surface_to_top(self, tmp_path):
        """MITgCM pressure-coordinate tRef is ordered k=1 surface to top."""
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=20, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        match = re.search(r"tRef\s*=\s*(.*?)(?=\n\s*[a-zA-Z])", content, re.DOTALL)
        assert match is not None
        values = [
            float(v.strip().rstrip(","))
            for v in re.split(r"[,\n]+", match.group(1))
            if v.strip().rstrip(",")
        ]
        assert values[0] > values[-1]

    def test_vertical_layers_sum_to_p0(self, tmp_path):
        """delR should define Nr pressure layers summing to p0."""
        Nr = 20
        p = tmp_path / "data"
        write_data(p, Nlon=128, Nlat=64, Nr=Nr, delta_t=600.0,
                   n_iter0=0, n_timesteps=1000, write_pickup=False,
                   pchkpt_freq=0.0, has_ic_file=False,
                   T0=HS_T0, delta_theta_z=10.0)
        content = p.read_text()
        match = re.search(r"delR\s*=\s*(.*?),\n\s*ygOrigin", content, re.DOTALL)
        assert match is not None
        values = [
            float(v.strip().rstrip(","))
            for v in re.split(r"[,\n]+", match.group(1))
            if v.strip().rstrip(",")
        ]
        assert len(values) == Nr
        assert sum(values) == pytest.approx(1.0e5)


# ─── write_data_hs_forc ─────────────────────────────────────────────────────

class TestWriteDataHsForc:
    def _write(self, tmp_path, kf, ka, ks, dTy, dTz, sigmab=0.7, T0=315.0):
        p = tmp_path / "data.hs_forc"
        write_data_hs_forc(p, kf=kf, ka=ka, ks=ks,
                           delta_T_y=dTy, delta_theta_z=dTz,
                           sigmab=sigmab, T0=T0)
        return p

    def test_namelist_name_present(self, tmp_path):
        p = self._write(tmp_path, 1e-5, 1e-6, 1e-5, 60.0, 10.0)
        assert "HS_FORC_PARM01" in p.read_text()

    def test_kf_value_stored_in_si(self, tmp_path):
        kf = 1.0 / (0.5 * 86400.0)   # τ_drag = 0.5 day
        p = self._write(tmp_path, kf=kf, ka=1e-6, ks=1e-5,
                        dTy=60.0, dTz=10.0)
        content = p.read_text()
        # The value should appear as a float, not as "0.5" days.
        assert "HS_kf" in content
        # Verify the written value parses back correctly.
        match = re.search(r"HS_kf\s*=\s*([\d.eE+\-]+)", content)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(kf, rel=1e-6)

    def test_delta_T_y_stored(self, tmp_path):
        p = self._write(tmp_path, 1e-5, 1e-6, 1e-5, dTy=80.0, dTz=5.0)
        content = p.read_text()
        assert "80" in content

    def test_sigmab_stored(self, tmp_path):
        p = self._write(tmp_path, 1e-5, 1e-6, 1e-5, dTy=60.0, dTz=10.0,
                        sigmab=0.7)
        content = p.read_text()
        assert "0.7" in content

    def test_file_closed_properly(self, tmp_path):
        """File should be readable after writing (i.e., properly closed)."""
        p = self._write(tmp_path, 1e-5, 1e-6, 1e-5, 60.0, 10.0)
        _ = p.read_text()   # Should not raise.

    @pytest.mark.parametrize("tau_drag_days", [0.5, 1.0, 2.0])
    def test_roundtrip_kf(self, tmp_path, tau_drag_days):
        """kf = 1/(tau_drag_days * 86400) should roundtrip through the file."""
        kf = 1.0 / (tau_drag_days * 86400.0)
        p = tmp_path / "data.hs_forc"
        write_data_hs_forc(p, kf=kf, ka=HS_SIGMAB, ks=1e-5,
                           delta_T_y=60.0, delta_theta_z=10.0,
                           sigmab=HS_SIGMAB, T0=HS_T0)
        content = p.read_text()
        match = re.search(r"HS_kf\s*=\s*([\d.eE+\-]+)", content)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(kf, rel=1e-5)


# ─── write_data_pkg ─────────────────────────────────────────────────────────

class TestWriteDataPkg:
    def test_no_nonstandard_hs_forc_package_flag(self, tmp_path):
        p = tmp_path / "data.pkg"
        write_data_pkg(p, use_diagnostics=True)
        assert "useHS_FORC" not in p.read_text()

    def test_diagnostics_enabled(self, tmp_path):
        p = tmp_path / "data.pkg"
        write_data_pkg(p, use_diagnostics=True)
        assert "useDiagnostics = .TRUE." in p.read_text()

    def test_diagnostics_disabled(self, tmp_path):
        p = tmp_path / "data.pkg"
        write_data_pkg(p, use_diagnostics=False)
        assert "useDiagnostics = .FALSE." in p.read_text()

    def test_shap_filt_enabled(self, tmp_path):
        p = tmp_path / "data.pkg"
        write_data_pkg(p)
        assert "useShap_Filt   = .TRUE." in p.read_text()


# ─── write_eedata ───────────────────────────────────────────────────────────

class TestWriteEedata:
    def test_threads_set_to_one(self, tmp_path):
        p = tmp_path / "eedata"
        write_eedata(p)
        content = p.read_text()
        assert "nTx = 1" in content
        assert "nTy = 1" in content


# ─── write_data_diagnostics ─────────────────────────────────────────────────

class TestWriteDataDiagnostics:
    def test_all_four_fields_present(self, tmp_path):
        p = tmp_path / "data.diagnostics"
        write_data_diagnostics(p, snapshot_interval_s=86400.0)
        content = p.read_text()
        for field in ("UVEL", "VVEL", "THETA", "ETAN"):
            assert field in content

    def test_filename_is_atm_state(self, tmp_path):
        p = tmp_path / "data.diagnostics"
        write_data_diagnostics(p, snapshot_interval_s=86400.0)
        assert "atm_state" in p.read_text()

    def test_surface_pressure_uses_separate_stream(self, tmp_path):
        p = tmp_path / "data.diagnostics"
        write_data_diagnostics(p, snapshot_interval_s=86400.0)
        content = p.read_text()
        assert "fields(1:1,2) = 'ETAN    '" in content
        assert "fileName(2)   = 'atm_surf'" in content

    def test_all_vertical_levels_requested(self, tmp_path):
        p = tmp_path / "data.diagnostics"
        write_data_diagnostics(p, snapshot_interval_s=86400.0, Nr=4)
        assert "levels(1:4,1) = 1, 2, 3, 4" in p.read_text()

    @pytest.mark.parametrize("interval_s", [3600.0, 86400.0, 43200.0])
    def test_frequency_matches_interval(self, tmp_path, interval_s):
        p = tmp_path / "data.diagnostics"
        write_data_diagnostics(p, snapshot_interval_s=interval_s)
        content = p.read_text()
        match = re.search(r"frequency(?:\(\d+\))?\s*=\s*([\d.eE+\-]+)", content)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(interval_s, rel=1e-6)


# ─── write_data_shap ────────────────────────────────────────────────────────

class TestWriteDataShap:
    def test_shap_filter_enabled(self, tmp_path):
        p = tmp_path / "data.shap"
        write_data_shap(p)
        assert "SHAP_PARM01" in p.read_text()

    def test_fourth_order_filter(self, tmp_path):
        p = tmp_path / "data.shap"
        write_data_shap(p)
        content = p.read_text()
        assert "nShapT        = 4" in content
        assert "nShapUV       = 4" in content


# ─── write_all_namelists ─────────────────────────────────────────────────────

class TestWriteAllNamelists:
    def _default_kwargs(self):
        return dict(
            Nlon=128, Nlat=64, Nr=20,
            delta_t=600.0,
            n_iter0=0, n_timesteps=1000,
            write_pickup=True,
            pchkpt_freq=600000.0,
            has_ic_file=True,
            has_diagnostics=False,
            snapshot_interval_s=86400.0,
            kf=1.0 / 86400.0,
            ka=1.0 / (40 * 86400.0),
            ks=1.0 / (4 * 86400.0),
            delta_T_y=60.0,
            delta_theta_z=10.0,
            sigmab=HS_SIGMAB,
            T0=HS_T0,
        )

    def test_all_files_created_without_diagnostics(self, tmp_path):
        write_all_namelists(tmp_path, **self._default_kwargs())
        for name in ("data", "data.pkg", "eedata", "data.hs_forc",
                     "data.shap"):
            assert (tmp_path / name).exists(), f"{name} missing"
        assert not (tmp_path / "data.diagnostics").exists()

    def test_diagnostics_file_created_when_enabled(self, tmp_path):
        kwargs = self._default_kwargs()
        kwargs["has_diagnostics"] = True
        kwargs["has_ic_file"] = False
        write_all_namelists(tmp_path, **kwargs)
        assert (tmp_path / "data.diagnostics").exists()
