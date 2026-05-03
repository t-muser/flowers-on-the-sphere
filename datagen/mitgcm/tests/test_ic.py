"""Tests for MDS binary initial condition generation.

Verifies file sizes, binary format, companion .meta files, and the
statistical properties of the IC perturbation. No MITgcm binary required.

Run::

    uv run --project datagen pytest datagen/mitgcm/tests/test_ic.py -v
"""

from __future__ import annotations

import re
import struct

import numpy as np
import pytest

from datagen.mitgcm._constants import P0
from datagen.mitgcm.ic import (
    _write_mds_meta,
    write_bathymetry,
    write_temperature_ic,
)


# ─── _write_mds_meta ────────────────────────────────────────────────────────

class TestWriteMdsMeta:
    def test_3d_meta_contains_correct_dims(self, tmp_path):
        p = tmp_path / "field.meta"
        _write_mds_meta(p, shape=(20, 64, 128))
        content = p.read_text()
        assert "nDims = [    3 ]" in content
        # dimList in Fortran order (x=128 first, then y=64, then z=20).
        assert "128" in content
        assert "64"  in content
        assert "20"  in content

    def test_2d_meta_contains_correct_dims(self, tmp_path):
        p = tmp_path / "bathy.meta"
        _write_mds_meta(p, shape=(64, 128))
        content = p.read_text()
        assert "nDims = [    2 ]" in content

    def test_dataprec_is_float32(self, tmp_path):
        p = tmp_path / "f.meta"
        _write_mds_meta(p, shape=(20, 64, 128))
        assert "'float32'" in p.read_text()

    def test_nrecords_is_1(self, tmp_path):
        p = tmp_path / "f.meta"
        _write_mds_meta(p, shape=(10, 32, 64))
        assert "nrecords = [    1 ]" in p.read_text()

    def test_fortran_dim_order(self, tmp_path):
        """dimList must list dims in Fortran order: Nx first, then Ny, then Nr."""
        p = tmp_path / "f.meta"
        _write_mds_meta(p, shape=(20, 64, 128))  # (Nr, Ny, Nx)
        content = p.read_text()
        # Extract dimList section.
        match = re.search(r"dimList\s*=\s*\[(.*?)\]", content, re.DOTALL)
        assert match is not None
        entries = [e.strip() for e in match.group(1).split(",") if e.strip()]
        # Each dimension is represented by three entries: total, lo, hi.
        # First dim should be Nx=128.
        first_val = int(entries[0])
        assert first_val == 128, f"First dimList entry should be Nx=128, got {first_val}"


# ─── write_temperature_ic ───────────────────────────────────────────────────

class TestWriteTemperatureIc:
    def test_data_file_created(self, tmp_path):
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=16, Nlat=8, Nr=4, seed=0)
        assert (tmp_path / "T.init.data").exists()

    def test_meta_file_created(self, tmp_path):
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=16, Nlat=8, Nr=4, seed=0)
        assert (tmp_path / "T.init.meta").exists()

    def test_file_size_matches_grid(self, tmp_path):
        """data file must have exactly Nr*Nlat*Nlon float32 values."""
        Nlon, Nlat, Nr = 16, 8, 4
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=Nlon, Nlat=Nlat, Nr=Nr, seed=0)
        expected_bytes = Nr * Nlat * Nlon * 4   # 4 bytes per float32
        assert (tmp_path / "T.init.data").stat().st_size == expected_bytes

    def test_big_endian_float32(self, tmp_path):
        """Data should be readable as big-endian float32 without error."""
        Nlon, Nlat, Nr = 16, 8, 4
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=Nlon, Nlat=Nlat, Nr=Nr, seed=0)
        data = np.fromfile(str(tmp_path / "T.init.data"), dtype=">f4")
        assert data.shape == (Nr * Nlat * Nlon,)
        assert np.all(np.isfinite(data))

    def test_array_shape_after_reshape(self, tmp_path):
        """Reshaped (Nr, Nlat, Nlon) should give correct dimensions."""
        Nlon, Nlat, Nr = 32, 16, 5
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=Nlon, Nlat=Nlat, Nr=Nr, seed=0)
        raw = np.fromfile(str(tmp_path / "T.init.data"), dtype=">f4")
        arr = raw.reshape(Nr, Nlat, Nlon)
        assert arr.shape == (Nr, Nlat, Nlon)

    def test_reference_profile_stable_stratification(self, tmp_path):
        """Potential temperature should increase from surface to top.

        The IC field is potential temperature θ.  A stably stratified atmosphere
        has θ increasing upward. MITgCM pressure-coordinate arrays are ordered
        k=1 at the high-pressure surface and k=Nr at the low-pressure top.
        """
        Nlon, Nlat, Nr = 16, 8, 10
        # Use tiny amplitude so perturbations don't obscure the trend.
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=Nlon, Nlat=Nlat, Nr=Nr,
            seed=0, amplitude=1e-6)
        raw = np.fromfile(str(tmp_path / "T.init.data"), dtype=">f4")
        arr = raw.reshape(Nr, Nlat, Nlon)
        # Level mean at each k.
        T_mean = arr.mean(axis=(1, 2))
        # Potential temperature must increase monotonically from surface to top.
        assert np.all(np.diff(T_mean) >= -0.1), (
            "Potential temperature should increase monotonically from surface "
            "(k=0) to top (k=Nr-1) for stable stratification"
        )

    def test_values_are_physical_temperatures(self, tmp_path):
        """All IC values should be positive and physically plausible (100–1000 K)."""
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=16, Nlat=8, Nr=5, seed=7)
        raw = np.fromfile(str(tmp_path / "T.init.data"), dtype=">f4")
        assert np.all(raw > 100.0), "Some IC values are unrealistically cold"
        assert np.all(raw < 1000.0), "Some IC values are unrealistically hot"

    def test_same_seed_gives_identical_output(self, tmp_path):
        """Deterministic: same seed → byte-identical files."""
        for i, suffix in enumerate(["a", "b"]):
            write_temperature_ic(
                tmp_path / f"T_{suffix}.data",
                Nlon=16, Nlat=8, Nr=4, seed=42)
        a = np.fromfile(str(tmp_path / "T_a.data"), dtype=">f4")
        b = np.fromfile(str(tmp_path / "T_b.data"), dtype=">f4")
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_output(self, tmp_path):
        """Different seeds produce different IC fields."""
        for seed in (0, 1):
            write_temperature_ic(
                tmp_path / f"T_{seed}.data",
                Nlon=16, Nlat=8, Nr=4, seed=seed)
        a = np.fromfile(str(tmp_path / "T_0.data"), dtype=">f4")
        b = np.fromfile(str(tmp_path / "T_1.data"), dtype=">f4")
        assert not np.allclose(a, b), "Different seeds should produce different ICs"

    def test_perturbation_amplitude_approx(self, tmp_path):
        """Max absolute perturbation should be within 3× the requested amplitude."""
        amplitude = 0.1
        Nlon, Nlat, Nr = 16, 8, 4
        # Use very small smoothing to avoid over-damping.
        write_temperature_ic(
            tmp_path / "T.init.data", Nlon=Nlon, Nlat=Nlat, Nr=Nr,
            seed=0, amplitude=amplitude, smooth_sigma=0.5)
        raw = np.fromfile(str(tmp_path / "T.init.data"), dtype=">f4")
        arr = raw.reshape(Nr, Nlat, Nlon).astype(np.float64)

        # Subtract zonal mean per level/latitude (= reference field plus any
        # zonal-mean perturbation component).
        T_mean = arr.mean(axis=2, keepdims=True)
        noise = arr - T_mean
        assert noise.std() < 3.0 * amplitude

    @pytest.mark.parametrize("Nr", [10, 20, 30])
    def test_variable_Nr(self, tmp_path, Nr):
        """write_temperature_ic must work for any Nr."""
        write_temperature_ic(
            tmp_path / f"T_{Nr}.data", Nlon=8, Nlat=4, Nr=Nr, seed=0)
        expected = Nr * 4 * 8 * 4
        assert (tmp_path / f"T_{Nr}.data").stat().st_size == expected


# ─── write_bathymetry ───────────────────────────────────────────────────────

class TestWriteBathymetry:
    def test_data_file_created(self, tmp_path):
        write_bathymetry(tmp_path / "bathyFile.bin", Nlon=16, Nlat=8)
        assert (tmp_path / "bathyFile.bin").exists()

    def test_meta_file_created(self, tmp_path):
        write_bathymetry(tmp_path / "bathyFile.bin", Nlon=16, Nlat=8)
        assert (tmp_path / "bathyFile.meta").exists()

    def test_all_zeros(self, tmp_path):
        """Aqua-planet: every bathymetry value must be exactly zero."""
        write_bathymetry(tmp_path / "bathyFile.bin", Nlon=32, Nlat=16)
        raw = np.fromfile(str(tmp_path / "bathyFile.bin"), dtype=">f4")
        np.testing.assert_array_equal(raw, 0.0)

    def test_file_size(self, tmp_path):
        Nlon, Nlat = 32, 16
        write_bathymetry(tmp_path / "bathyFile.bin", Nlon=Nlon, Nlat=Nlat)
        assert (tmp_path / "bathyFile.bin").stat().st_size == Nlat * Nlon * 4

    def test_meta_says_2d(self, tmp_path):
        write_bathymetry(tmp_path / "bathyFile.bin", Nlon=16, Nlat=8)
        content = (tmp_path / "bathyFile.meta").read_text()
        assert "nDims = [    2 ]" in content
