"""Tests for the MITgcm global ocean cubed-sphere wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datagen.mitgcm.global_ocean import (
    GlobalOceanRunConfig,
    extract_global_ocean_fields,
    read_global_ocean_output,
    render_data,
    render_data_gmredi,
    render_data_pkg,
    stage_global_ocean_run,
)
from datagen.mitgcm.global_ocean.solver import _to_face_layout


def _minimal_input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "data").write_text(
        """\
 &PARM01
 viscAh =3.E5,
 diffKrT=3.E-5,
 diffKrS=3.E-5,
 useRealFreshWaterFlux=.TRUE.,
 readBinaryPrec=64,
 &
 &PARM03
 nIter0=72000,
 nTimeSteps=20,
 deltaTmom   =1200.,
 deltaTtracer=86400.,
 deltaTfreesurf=86400.,
 deltaTClock =86400.,
 pChkptFreq  =1728000.,
 dumpFreq    =622080000.,
 monitorFreq =31104000.,
 tauThetaClimRelax = 5184000.,
 tauSaltClimRelax = 62208000.,
 monitorFreq =1.,
 &
 &PARM04
 usingCurvilinearGrid=.TRUE.,
 horizGridFile='grid_cs32',
 radius_fromHorizGrid=6370.E3,
 &
 &PARM05
 bathyFile      ='bathy_Hmin50.bin',
 hydrogThetaFile='lev_T_cs_15k.bin',
 hydrogSaltFile ='lev_S_cs_15k.bin',
 zonalWindFile  ='trenberth_taux.bin',
 meridWindFile  ='trenberth_tauy.bin',
 thetaClimFile  ='lev_surfT_cs_12m.bin',
 saltClimFile   ='lev_surfS_cs_12m.bin',
 surfQnetFile   ='shiQnet_cs32.bin',
 EmPmRFile      ='shiEmPR_cs32.bin',
 &
"""
    )
    (input_dir / "data.gmredi").write_text(
        """\
 &GM_PARM01
  GM_background_K    = 800.,
  GM_taper_scheme    = 'gkw91',
 &
"""
    )
    (input_dir / "data.pkg").write_text(
        """\
 &PACKAGES
 useGMRedi=.TRUE.,
 useDiagnostics=.TRUE.,
 useMNC=.TRUE.,
 &
"""
    )
    (input_dir / "eedata").write_text(
        """\
 &EEPARMS
 useCubedSphereExchange=.TRUE.,
 nTx=1,
 nTy=1,
 &
"""
    )
    for name in (
        "bathy_Hmin50.bin",
        "lev_T_cs_15k.bin",
        "lev_S_cs_15k.bin",
        "trenberth_taux.bin",
        "trenberth_tauy.bin",
        "lev_surfT_cs_12m.bin",
        "lev_surfS_cs_12m.bin",
        "shiQnet_cs32.bin",
        "shiEmPR_cs32.bin",
        "pickup.0000072000.data",
        "pickup.0000072000.meta",
    ):
        (input_dir / name).write_bytes(b"input")
    return input_dir


def _minimal_grid_dir(tmp_path: Path) -> Path:
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    for i in range(6):
        (grid_dir / f"grid_cs32.face{i + 1:03d}.bin").write_bytes(b"grid")
    return grid_dir


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


def test_render_data_patches_runtime_overrides(tmp_path):
    input_dir = _minimal_input_dir(tmp_path)
    cfg = GlobalOceanRunConfig(
        input_dir=input_dir,
        n_iter0=72000,
        n_timesteps=7,
        snapshot_interval_days=3.0,
        visc_ah=2.0e5,
        diff_kr=1.0e-4,
        tau_theta_relax_days=30.0,
        tau_salt_relax_days=90.0,
    )
    text = render_data(cfg)

    assert "usingCurvilinearGrid=.TRUE." in text
    assert "horizGridFile='grid_cs32'" in text
    assert "bathyFile      ='bathy_Hmin50.bin'" in text
    assert "hydrogThetaFile='lev_T_cs_15k.bin'" in text
    assert "shiQnet_cs32.bin" in text
    assert "shiEmPR_cs32.bin" in text
    assert "useSingleCpuIO= .TRUE." in text
    assert "nIter0= 72000" in text
    assert "nTimeSteps= 7" in text
    assert "dumpFreq= 259200." in text
    assert "viscAh= 200000." in text
    assert "diffKrT= 0.0001" in text
    assert "diffKrS= 0.0001" in text
    assert "tauThetaClimRelax= 2592000." in text
    assert "tauSaltClimRelax= 7776000." in text


def test_render_data_gmredi_patches_background_diffusivity(tmp_path):
    input_dir = _minimal_input_dir(tmp_path)
    cfg = GlobalOceanRunConfig(input_dir=input_dir, gm_background_k=750.0)
    text = render_data_gmredi(cfg)
    assert "GM_background_K= 750." in text
    assert "GM_taper_scheme" in text


def test_render_data_pkg_disables_diagnostics_and_mnc(tmp_path):
    input_dir = _minimal_input_dir(tmp_path)
    cfg = GlobalOceanRunConfig(input_dir=input_dir)
    text = render_data_pkg(cfg)
    assert "useGMRedi=.TRUE." in text
    assert "useDiagnostics= .FALSE." in text
    assert "useMNC= .FALSE." in text


def test_stage_global_ocean_run_writes_namelists_and_symlinks_inputs(tmp_path):
    input_dir = _minimal_input_dir(tmp_path)
    grid_dir = _minimal_grid_dir(tmp_path)
    exe = tmp_path / "mitgcmuv"
    exe.write_text("#!/bin/sh\n")

    cfg = GlobalOceanRunConfig(
        executable=exe,
        input_dir=input_dir,
        grid_dir=grid_dir,
        n_timesteps=5,
        snapshot_interval_days=2.0,
    )
    run_dir = stage_global_ocean_run(tmp_path / "run", cfg)

    assert (run_dir / "mitgcmuv").is_symlink()
    assert (run_dir / "bathy_Hmin50.bin").is_symlink()
    assert (run_dir / "pickup.0000072000.data").is_symlink()
    assert (run_dir / "pickup.0000072000.meta").is_symlink()
    for i in range(6):
        assert (run_dir / f"grid_cs32.face{i + 1:03d}.bin").is_symlink()
    assert not (run_dir / "data").is_symlink()
    assert not (run_dir / "data.pkg").is_symlink()
    assert "nTimeSteps= 5" in (run_dir / "data").read_text()
    assert "useGMRedi=.TRUE." in (run_dir / "data.pkg").read_text()
    assert "useDiagnostics= .FALSE." in (run_dir / "data.pkg").read_text()
    assert "useCubedSphereExchange=.TRUE." in (run_dir / "eedata").read_text()


def test_to_face_layout_assigns_tiles_in_x_order():
    # Build an input where columns 0..31 are face 1, 32..63 face 2, etc.
    arr = np.empty((2, 32, 6 * 32), dtype=np.float32)
    for face in range(6):
        arr[..., face * 32:(face + 1) * 32] = float(face)

    out = _to_face_layout(arr)
    assert out.shape == (2, 6, 32, 32)
    for face in range(6):
        assert np.all(out[:, face] == float(face))


def test_read_and_extract_global_ocean_output(tmp_path):
    cfg = GlobalOceanRunConfig(
        n_face=2, face_size=3, Nr=2, delta_t_clock=10.0,
    )
    run_dir = tmp_path
    iters = [10, 20]

    n_face = 2
    fs = 3
    for iter_num in iters:
        offset = float(iter_num)
        # Global tile layout: (Nr, fs, n_face*fs) for 3-D and (fs, n_face*fs) for 2-D.
        shape_3d = (cfg.Nr, fs, n_face * fs)
        shape_2d = (fs, n_face * fs)
        _write_mds(run_dir / f"T.{iter_num:010d}", np.full(shape_3d, offset + 1),
                   iter_num=iter_num)
        _write_mds(run_dir / f"S.{iter_num:010d}", np.full(shape_3d, offset + 2),
                   iter_num=iter_num)
        _write_mds(run_dir / f"U.{iter_num:010d}", np.full(shape_3d, offset + 3),
                   iter_num=iter_num)
        _write_mds(run_dir / f"V.{iter_num:010d}", np.full(shape_3d, offset + 5),
                   iter_num=iter_num)
        _write_mds(run_dir / f"Eta.{iter_num:010d}", np.full(shape_2d, offset + 7),
                   iter_num=iter_num)

    data = read_global_ocean_output(run_dir, cfg)
    assert data["THETA"].shape == (2, cfg.Nr, n_face, fs, fs)
    assert data["ETAN"].shape == (2, n_face, fs, fs)

    fields, names, time = extract_global_ocean_fields(data, cfg)

    assert names == ["theta_k1", "salt_k1", "u_k2", "v_k2", "eta"]
    np.testing.assert_array_equal(time, np.array([0.0, 100.0]))
    assert fields[0].shape == (2, n_face, fs, fs)
    assert fields[0][0, 0, 0, 0] == pytest.approx(11.0)
    assert fields[2][0, 0, 0, 0] == pytest.approx(13.0)
    assert fields[4][1, 0, 0, 0] == pytest.approx(27.0)
