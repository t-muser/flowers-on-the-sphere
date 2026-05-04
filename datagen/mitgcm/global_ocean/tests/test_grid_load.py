"""Tests for cs32 grid file parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datagen.mitgcm.global_ocean import load_grid_cs32


_FACE_SIZE = 32
_REC_N = _FACE_SIZE + 1
_REC_BYTES = _REC_N * _REC_N * 8
_N_RECORDS = 18  # MITgcm grid_cs32 face files contain 18 records of 33x33 doubles


def _write_fake_grid(grid_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Generate 6 fake face files; return the expected interior xC/yC arrays."""
    rng = np.random.default_rng(0)
    xc_full = rng.uniform(0.0, 360.0, size=(6, _REC_N, _REC_N))
    yc_full = rng.uniform(-90.0, 90.0, size=(6, _REC_N, _REC_N))
    for i in range(6):
        path = grid_dir / f"grid_cs32.face{i + 1:03d}.bin"
        with open(path, "wb") as f:
            f.write(xc_full[i].astype(">f8").tobytes())
            f.write(yc_full[i].astype(">f8").tobytes())
            # Pad with junk records so the file size matches the real format.
            for _ in range(_N_RECORDS - 2):
                f.write(np.zeros(_REC_N * _REC_N, dtype=">f8").tobytes())
    return xc_full[:, :_FACE_SIZE, :_FACE_SIZE], yc_full[:, :_FACE_SIZE, :_FACE_SIZE]


def test_load_grid_cs32_returns_face_major_arrays(tmp_path):
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    xc_expected, yc_expected = _write_fake_grid(grid_dir)

    grid = load_grid_cs32(grid_dir)
    assert grid["xc"].shape == (6, _FACE_SIZE, _FACE_SIZE)
    assert grid["yc"].shape == (6, _FACE_SIZE, _FACE_SIZE)
    np.testing.assert_allclose(grid["xc"], xc_expected)
    np.testing.assert_allclose(grid["yc"], yc_expected)


def test_load_grid_cs32_real_files_have_sensible_lat_lon():
    repo_root = Path(__file__).resolve().parents[4]
    grid_dir = repo_root / "mitgcm" / "verification" / "tutorial_held_suarez_cs" / "input"
    if not (grid_dir / "grid_cs32.face001.bin").is_file():
        pytest.skip("MITgcm submodule not initialized")
    grid = load_grid_cs32(grid_dir)
    assert grid["xc"].shape == (6, _FACE_SIZE, _FACE_SIZE)
    assert grid["yc"].shape == (6, _FACE_SIZE, _FACE_SIZE)
    assert -180.0 <= grid["xc"].min() and grid["xc"].max() <= 360.0
    assert -90.0 <= grid["yc"].min() and grid["yc"].max() <= 90.0
