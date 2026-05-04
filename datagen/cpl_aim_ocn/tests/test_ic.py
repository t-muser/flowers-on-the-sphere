"""Tests for ``datagen/cpl_aim_ocn/ic.py``.

Validates the cs32 atm theta IC binary writer:

* Output file shape and byte order (big-endian **float64** —
  ``readBinaryPrec=64`` in upstream's atm namelist), ``Nr × Ny × Nx``.
* Companion ``.meta`` file dim list (Fortran order: Nx, Ny, Nr) and
  ``dataprec='float64'``.
* Determinism per seed and divergence across seeds.
* Reference-profile recovery (mean per level matches ``THETA_REF_K``).
* Per-face smoothness (no high-frequency grid-scale noise).
* Cube-edge sanity: noise is independent across face boundaries
  (which is what we want — face-by-face smoothing on the cs grid).

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_ic.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datagen.cpl_aim_ocn.ic import (
    FACE_NX,
    FACE_NY,
    NR_ATM,
    NX_GLOBAL,
    NY_GLOBAL,
    N_FACE,
    THETA_REF_K,
    _faces_to_unfolded,
    _gen_per_face_noise,
    _write_mds_meta,
    write_atm_theta_ic,
)


# ─── Module-level constants ──────────────────────────────────────────────────

class TestConstants:
    def test_grid_dimensions_match_cs32(self):
        assert N_FACE == 6
        assert FACE_NX == 32 and FACE_NY == 32
        assert NX_GLOBAL == 192 and NY_GLOBAL == 32
        assert NR_ATM == 5

    def test_theta_ref_strictly_increasing(self):
        # Stable atm: θ increases upward, k=1 at bottom → k=5 at top.
        for a, b in zip(THETA_REF_K, THETA_REF_K[1:]):
            assert b > a, f"theta_ref must strictly increase: {THETA_REF_K}"

    def test_theta_ref_values_match_upstream_data(self):
        # Hard-coded check guards against an accidental edit drifting
        # away from the upstream verification's tRef line.
        assert THETA_REF_K == (289.6, 298.1, 314.5, 335.8, 437.4)


# ─── Internal helpers ────────────────────────────────────────────────────────

class TestGenPerFaceNoise:
    def test_shape(self):
        n = _gen_per_face_noise(seed=0)
        assert n.shape == (NR_ATM, N_FACE, FACE_NY, FACE_NX)

    def test_deterministic_for_same_seed(self):
        a = _gen_per_face_noise(seed=42)
        b = _gen_per_face_noise(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = _gen_per_face_noise(seed=0)
        b = _gen_per_face_noise(seed=1)
        # Allow some statistical collision but require substantial
        # difference (at least one element off by > 1e-6).
        assert np.max(np.abs(a - b)) > 1e-6

    def test_amplitude_rescaling(self):
        # Per-(level, face) RMS should match `amplitude` to ≲ 5%.
        n = _gen_per_face_noise(seed=7, amplitude=0.5)
        rms = np.sqrt(np.mean(n ** 2, axis=(2, 3)))
        assert np.allclose(rms, 0.5, rtol=1e-9, atol=1e-12)

    def test_zero_mean_per_panel_after_smoothing(self):
        # Within a small numerical tolerance, smoothing preserves mean ≈ 0.
        n = _gen_per_face_noise(seed=0, smooth_sigma=3.0)
        means = np.mean(n, axis=(2, 3))
        # Per-panel rescaling preserves the sign of the mean but
        # divides by the per-panel RMS, so the mean ÷ RMS is the
        # quantity to check (should remain near zero).
        assert np.all(np.abs(means) < 0.5)

    def test_smoothing_reduces_high_frequency(self):
        # Compare neighbouring-cell variance vs domain variance.
        # Smoothed noise should be much "smoother" than IID noise.
        n = _gen_per_face_noise(seed=11, smooth_sigma=3.0)
        # Take one panel, compute first differences along i.
        panel = n[2, 3]  # arbitrary level/face
        diff_var = np.var(np.diff(panel, axis=1))
        total_var = np.var(panel)
        # For IID Gaussian, var(diff) = 2 × var. After 2-D smoothing
        # with σ=3, var(diff) ≪ 2 var. Require ≤ var (i.e. far smoother).
        assert diff_var < total_var, (
            f"diff variance {diff_var:.3g} not below total variance {total_var:.3g}"
        )


class TestFacesToUnfolded:
    def test_shape(self):
        panels = np.zeros((NR_ATM, N_FACE, FACE_NY, FACE_NX))
        out = _faces_to_unfolded(panels)
        assert out.shape == (NR_ATM, NY_GLOBAL, NX_GLOBAL)

    def test_face_layout_in_X(self):
        # Tag each face with a unique constant; check it ends up at the
        # right column slice in the unfolded array.
        panels = np.zeros((NR_ATM, N_FACE, FACE_NY, FACE_NX))
        for f in range(N_FACE):
            panels[:, f, :, :] = float(f + 1)  # avoid collision with default 0
        out = _faces_to_unfolded(panels)
        for f in range(N_FACE):
            sl = slice(f * FACE_NX, (f + 1) * FACE_NX)
            assert np.all(out[:, :, sl] == float(f + 1)), \
                f"face {f} not at columns {sl.start}–{sl.stop}"

    def test_face_layout_preserves_within_face_indexing(self):
        # Tag each cell with a unique value. After unfolding, indexing
        # into the panel should match the unfolded coordinates.
        panels = np.zeros((1, N_FACE, FACE_NY, FACE_NX))
        for f in range(N_FACE):
            for j in range(FACE_NY):
                for i in range(FACE_NX):
                    panels[0, f, j, i] = (
                        f * 1_000_000 + j * 1000 + i
                    )
        out = _faces_to_unfolded(panels)
        for f in range(N_FACE):
            for j in range(FACE_NY):
                for i in range(FACE_NX):
                    expected = f * 1_000_000 + j * 1000 + i
                    assert out[0, j, f * FACE_NX + i] == expected

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            _faces_to_unfolded(np.zeros((NR_ATM, 5, FACE_NY, FACE_NX)))


# ─── _write_mds_meta ─────────────────────────────────────────────────────────

class TestWriteMdsMeta:
    def test_dim_list_in_fortran_order(self, tmp_path: Path):
        # C-shape (Nr, Ny, Nx) → Fortran dimList = [Nx, Ny, Nr].
        p = tmp_path / "x.meta"
        _write_mds_meta(p, shape=(5, 32, 192))
        text = p.read_text()
        # First dim listed should be 192 (Nx, fastest-varying).
        lines = [ln.strip() for ln in text.splitlines()]
        first_dim_line = next(ln for ln in lines if ", " in ln and ln.strip()[0].isdigit())
        assert first_dim_line.startswith("192,"), text

    def test_3d_meta_header_well_formed(self, tmp_path: Path):
        p = tmp_path / "x.meta"
        _write_mds_meta(p, shape=(5, 32, 192))
        text = p.read_text()
        assert "nDims = [    3 ];" in text
        assert "float64" in text
        assert "nrecords = [    1 ];" in text


# ─── write_atm_theta_ic: end-to-end ──────────────────────────────────────────

class TestWriteAtmThetaIc:
    def test_creates_data_and_meta_files(self, tmp_path: Path):
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=0)
        assert out.is_file()
        assert (tmp_path / "theta.meta").is_file()

    def test_file_size_matches_layout(self, tmp_path: Path):
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=0)
        # Nr × Ny × Nx × 8 bytes (big-endian float64 — matches
        # MITgcm's readBinaryPrec=64 in the upstream atm namelist).
        expected = NR_ATM * NY_GLOBAL * NX_GLOBAL * 8
        assert out.stat().st_size == expected

    def test_byte_order_is_big_endian(self, tmp_path: Path):
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=0)
        # Big-endian read: every value should be in a sane temperature
        # range (~289 K to ~437 K plus tiny perturbation).
        be = np.fromfile(out, dtype=">f8")
        assert np.all((be > 280.0) & (be < 450.0)), (
            "Big-endian read produced values outside the reference profile range"
        )
        # Little-endian read of the same bytes scrambles them into
        # nonsense: most values will be far outside the temperature
        # range. Use np.errstate to silence overflow warnings from the
        # scrambled byte interpretation.
        le_raw = np.fromfile(out, dtype="<f8")
        with np.errstate(invalid="ignore", over="ignore"):
            plausible = np.isfinite(le_raw) & (le_raw > 100.0) & (le_raw < 1000.0)
        assert not np.all(plausible), (
            "Little-endian read produced all-plausible values — file may "
            "have been written in the wrong byte order"
        )

    def test_meta_dataprec_is_float64(self, tmp_path: Path):
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=0)
        # `.meta` must declare float64 to match the actual on-disk
        # precision; with float32 here MITgcm reads garbage at startup.
        meta_text = (tmp_path / "theta.meta").read_text()
        assert "float64" in meta_text
        assert "float32" not in meta_text

    def test_per_level_mean_matches_reference_profile(self, tmp_path: Path):
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=0)
        arr = np.fromfile(out, dtype=">f8").reshape(
            NR_ATM, NY_GLOBAL, NX_GLOBAL
        )
        # Spatial mean per level should equal tRef[k] to within the
        # standard error of the noise field. With per-panel RMS = 0.1 K
        # and 6 independent panels each containing 32×32 = 1024 cells,
        # the global mean's standard deviation is ≤ 0.1 / sqrt(6144) ≈
        # 0.0013 K. We allow 0.05 K to give plenty of headroom.
        for k in range(NR_ATM):
            assert abs(arr[k].mean() - THETA_REF_K[k]) < 0.05, (
                f"level {k}: mean={arr[k].mean()} ref={THETA_REF_K[k]}"
            )

    def test_perturbation_amplitude_within_factor(self, tmp_path: Path):
        # Each level's stddev around tRef[k] should be ≈ amplitude (0.1)
        # to within a factor of 2.
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=0, amplitude=0.1)
        arr = np.fromfile(out, dtype=">f8").reshape(
            NR_ATM, NY_GLOBAL, NX_GLOBAL
        )
        for k in range(NR_ATM):
            sd = float(np.std(arr[k] - THETA_REF_K[k]))
            assert 0.05 < sd < 0.5, f"level {k} sd={sd}"

    def test_deterministic_for_same_seed(self, tmp_path: Path):
        out_a = tmp_path / "a.bin"
        out_b = tmp_path / "b.bin"
        write_atm_theta_ic(out_a, seed=42)
        write_atm_theta_ic(out_b, seed=42)
        a = out_a.read_bytes()
        b = out_b.read_bytes()
        assert a == b

    def test_different_seeds_produce_different_bytes(self, tmp_path: Path):
        out_a = tmp_path / "a.bin"
        out_b = tmp_path / "b.bin"
        write_atm_theta_ic(out_a, seed=0)
        write_atm_theta_ic(out_b, seed=1)
        assert out_a.read_bytes() != out_b.read_bytes()

    def test_meta_file_lists_x_first(self, tmp_path: Path):
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=0)
        meta_text = (tmp_path / "theta.meta").read_text()
        # Find the first dimension entry; it must be Nx=192.
        for line in meta_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("192,"):
                break
        else:
            pytest.fail(f"Nx=192 not first in dimList:\n{meta_text}")

    def test_per_face_smoothing_no_cube_edge_spikes(self, tmp_path: Path):
        # The on-disk array has the six panels concatenated along Nx.
        # Differences across the panel boundary (col k*32 vs col k*32-1)
        # are uncorrelated noise (independent panels), so the boundary
        # discontinuity is on the scale of the noise amplitude — not a
        # huge spike. We require the boundary discontinuity to be at
        # most a few × the within-panel noise stddev.
        out = tmp_path / "theta.bin"
        write_atm_theta_ic(out, seed=3, amplitude=0.1, smooth_sigma=3.0)
        arr = np.fromfile(out, dtype=">f8").reshape(
            NR_ATM, NY_GLOBAL, NX_GLOBAL
        )
        for boundary in range(1, N_FACE):
            x = boundary * FACE_NX
            jumps = arr[:, :, x] - arr[:, :, x - 1]
            # All on the order of the perturbation, never dominant
            # over the reference profile.
            assert np.max(np.abs(jumps)) < 1.0

    def test_rejects_wrong_theta_ref_length(self, tmp_path: Path):
        out = tmp_path / "x.bin"
        with pytest.raises(ValueError, match="theta_ref"):
            write_atm_theta_ic(out, seed=0, theta_ref=[300.0, 310.0])
