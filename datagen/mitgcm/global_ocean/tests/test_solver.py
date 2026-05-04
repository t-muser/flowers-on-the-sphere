"""Tests for the MITgcm global ocean tutorial wrapper."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from datagen.mitgcm.global_ocean import (
    GLOBAL_OCEAN_DEL_R,
    GlobalOceanRunConfig,
    extract_global_ocean_fields,
    global_ocean_lat_grid,
    global_ocean_lon_grid,
    read_global_ocean_output,
    render_data,
    render_data_gmredi,
    stage_global_ocean_run,
)


def _minimal_input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "data").write_text(
        """\
 &PARM01
 viscAh=5.E5,
 diffKrT=3.E-5,
 diffKrS=3.E-5,
 &
 &PARM03
 nTimeSteps = 20,
 deltaTmom = 1800.,
 deltaTtracer= 86400.,
 deltaTClock = 86400.,
 deltaTfreesurf= 86400.,
 pChkptFreq= 1728000.,
 dumpFreq= 311040000.,
 dumpFreq= 864000.,
 monitorFreq=1.,
 &
 &PARM05
 bathyFile=      'bathymetry.bin',
 hydrogThetaFile='lev_t.bin',
 hydrogSaltFile= 'lev_s.bin',
 zonalWindFile=  'trenberth_taux.bin',
 meridWindFile=  'trenberth_tauy.bin',
 thetaClimFile=  'lev_sst.bin',
 saltClimFile=   'lev_sss.bin',
 surfQnetFile=   'ncep_qnet.bin',
 the_run_name=   'global_oce_latlon',
 EmPmRFile=      'ncep_emp.bin',
 &
"""
    )
    (input_dir / "data.gmredi").write_text(
        """\
 &GM_PARM01
  GM_background_K    = 1.e+3,
  GM_taper_scheme    = 'gkw91',
 &
"""
    )
    (input_dir / "data.pkg").write_text(
        """\
 &PACKAGES
 useGMRedi=.TRUE.,
 usePTRACERS=.TRUE.,
 &
"""
    )
    (input_dir / "data.ptracers").write_text(
        """\
 &PTRACERS_PARM01
 PTRACERS_numInUse=1,
 &
"""
    )
    (input_dir / "eedata").write_text(
        """\
 &EEPARMS
 nTx=1,
 nTy=1,
 &
"""
    )
    for name in (
        "bathymetry.bin",
        "lev_t.bin",
        "lev_s.bin",
        "trenberth_taux.bin",
        "trenberth_tauy.bin",
        "lev_sst.bin",
        "lev_sss.bin",
        "ncep_qnet.bin",
        "ncep_emp.bin",
    ):
        (input_dir / name).write_bytes(b"input")
    return input_dir


def _write_mds(path: Path, arr: np.ndarray, *, iter_num: int) -> None:
    """Write a global one-record MDS file for test data."""
    shape = arr.shape
    fortran_dims = list(reversed(shape))
    dim_str = ",\n         ".join(
        f"{d:6d}, {1:6d}, {d:6d}" for d in fortran_dims
    )
    Path(str(path) + ".meta").write_text(
        f"""\
 nDims = [{len(shape):5d} ];
 dimList = [
         {dim_str}
 ];
 dataprec = [ 'float32' ];
 nrecords = [{1:5d} ];
 timeStepNumber = [{iter_num:10d} ];
"""
    )
    arr.astype(">f4").tofile(str(path) + ".data")


def test_render_data_follows_tutorial_grid_and_forcing_files():
    cfg = GlobalOceanRunConfig(
        n_timesteps=7,
        snapshot_interval_days=3.0,
        visc_ah=2.0e5,
        diff_kr=1.0e-4,
        tau_theta_relax_days=30.0,
        tau_salt_relax_days=90.0,
    )
    text = render_data(cfg)

    assert "usingSphericalPolarGrid=.TRUE." in text
    assert "dySpacing=4." in text
    assert "dxSpacing=4." in text
    assert "ygOrigin=-80." in text
    assert "bathyFile=      'bathymetry.bin'" in text
    assert "zonalWindFile=  'trenberth_taux.bin'" in text
    assert "surfQnetFile=   'ncep_qnet.bin'" in text
    assert "EmPmRFile=      'ncep_emp.bin'" in text
    assert "eosType = 'JMD95Z'" in text
    assert "nTimeSteps= 7" in text
    assert "dumpFreq= 259200." in text
    assert "viscAh= 200000." in text
    assert "diffKrT= 0.0001" in text
    assert "diffKrS= 0.0001" in text
    assert "tauThetaClimRelax= 2592000." in text
    assert "tauSaltClimRelax= 7776000." in text

    match = re.search(r"delR=\s*(.*?),\n\s*ygOrigin", text, re.DOTALL)
    assert match is not None
    values = [
        float(v.strip().rstrip(","))
        for v in re.split(r"[,\n]+", match.group(1))
        if v.strip().rstrip(",")
    ]
    assert tuple(values) == GLOBAL_OCEAN_DEL_R


def test_render_data_gmredi_patches_background_diffusivity(tmp_path):
    input_dir = _minimal_input_dir(tmp_path)
    cfg = GlobalOceanRunConfig(input_dir=input_dir, gm_background_k=750.0)
    text = render_data_gmredi(cfg)
    assert "GM_background_K= 750." in text
    assert "GM_taper_scheme" in text


def test_stage_global_ocean_run_writes_namelists_and_symlinks_inputs(tmp_path):
    input_dir = _minimal_input_dir(tmp_path)
    exe = tmp_path / "mitgcmuv"
    exe.write_text("#!/bin/sh\n")

    cfg = GlobalOceanRunConfig(
        executable=exe,
        input_dir=input_dir,
        n_timesteps=5,
        snapshot_interval_days=2.0,
    )
    run_dir = stage_global_ocean_run(tmp_path / "run", cfg)

    assert (run_dir / "mitgcmuv").is_symlink()
    assert (run_dir / "bathymetry.bin").is_symlink()
    assert not (run_dir / "data").is_symlink()
    assert "nTimeSteps= 5" in (run_dir / "data").read_text()
    assert "useGMRedi=.TRUE." in (run_dir / "data.pkg").read_text()
    assert "PTRACERS_numInUse=1" in (run_dir / "data.ptracers").read_text()


def test_global_ocean_grids_are_tutorial_cell_centers():
    lat_deg = np.rad2deg(global_ocean_lat_grid(40))
    lon_deg = np.rad2deg(global_ocean_lon_grid(90))

    assert lat_deg[0] == pytest.approx(-78.0)
    assert lat_deg[-1] == pytest.approx(78.0)
    assert np.diff(lat_deg).mean() == pytest.approx(4.0)
    assert lon_deg[0] == pytest.approx(2.0)
    assert lon_deg[-1] == pytest.approx(358.0)
    assert np.diff(lon_deg).mean() == pytest.approx(4.0)


def test_read_and_extract_global_ocean_output(tmp_path):
    cfg = GlobalOceanRunConfig(Nlon=3, Nlat=2, Nr=2, delta_t_clock=10.0)
    run_dir = tmp_path
    iters = [10, 20]

    for iter_num in iters:
        offset = float(iter_num)
        shape_3d = (2, 2, 3)
        shape_2d = (2, 3)
        _write_mds(run_dir / f"T.{iter_num:010d}", np.full(shape_3d, offset + 1),
                   iter_num=iter_num)
        _write_mds(run_dir / f"S.{iter_num:010d}", np.full(shape_3d, offset + 2),
                   iter_num=iter_num)
        u = np.stack([
            np.full(shape_2d, offset + 3),
            np.full(shape_2d, offset + 4),
        ])
        v = np.stack([
            np.full(shape_2d, offset + 5),
            np.full(shape_2d, offset + 6),
        ])
        _write_mds(run_dir / f"U.{iter_num:010d}", u, iter_num=iter_num)
        _write_mds(run_dir / f"V.{iter_num:010d}", v, iter_num=iter_num)
        _write_mds(run_dir / f"Eta.{iter_num:010d}", np.full(shape_2d, offset + 7),
                   iter_num=iter_num)

    data = read_global_ocean_output(run_dir, cfg)
    fields, names, time = extract_global_ocean_fields(data, cfg)

    assert names == ["theta_k1", "salt_k1", "u_k2", "v_k2", "eta"]
    np.testing.assert_array_equal(time, np.array([0.0, 100.0]))
    assert fields[0].shape == (2, 2, 3)
    assert fields[0][0, 0, 0] == pytest.approx(11.0)
    assert fields[2][0, 0, 0] == pytest.approx(14.0)
    assert fields[4][1, 0, 0] == pytest.approx(27.0)
