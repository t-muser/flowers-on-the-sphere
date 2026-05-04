"""Tests for ``datagen/cpl_aim_ocn/zarr_writer.py``.

Two tiers:

1. Pure-Python unit tests on the small helpers (constants, vertical-
   dimension renaming) — fast, no external state.
2. Integration tests that monkeypatch ``xmitgcm.open_mdsdataset`` to
   return synthetic ``xarray.Dataset`` objects mirroring what the real
   loader would produce for cs32 atm + ocn streams. This exercises the
   merge, rename, and Zarr-write path without needing actual MITgcm
   output.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_zarr_writer.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from datagen.cpl_aim_ocn import zarr_writer
from datagen.cpl_aim_ocn.zarr_writer import (
    ATM_STREAMS,
    OCN_STREAMS,
    _rename_atm_vertical,
    write_cs32_zarr,
)


# ─── Constants ───────────────────────────────────────────────────────────────

class TestStreamConstants:
    def test_atm_streams_match_namelist_filenames(self):
        # The atm diagnostics file names emitted by namelist.py's
        # render_atm_diagnostics() must be the same set the writer
        # tries to read. Hard-coded check guards against drift.
        from datagen.cpl_aim_ocn.namelist import render_atm_diagnostics
        text = render_atm_diagnostics(snapshot_interval_s=86400.0)
        for stream in ATM_STREAMS:
            assert f"'{stream}'" in text, (
                f"ATM_STREAMS list contains {stream!r} but the namelist "
                "renderer doesn't write it"
            )

    def test_ocn_streams_match_namelist_filenames(self):
        from datagen.cpl_aim_ocn.namelist import render_ocn_diagnostics
        text = render_ocn_diagnostics(snapshot_interval_s=86400.0)
        for stream in OCN_STREAMS:
            assert f"'{stream}'" in text


# ─── _rename_atm_vertical ────────────────────────────────────────────────────

class TestRenameAtmVertical:
    def test_renames_z_to_zsigma(self):
        ds = xr.Dataset(
            data_vars={"u": (("Z",), np.zeros(5))},
            coords={"Z": np.arange(5)},
        )
        out = _rename_atm_vertical(ds)
        assert "Zsigma" in out.dims
        assert "Z" not in out.dims

    def test_renames_zl_zp1_when_present(self):
        ds = xr.Dataset(
            data_vars={
                "u": (("Z",), np.zeros(5)),
                "w": (("Zl",), np.zeros(5)),
            },
            coords={"Z": np.arange(5), "Zl": np.arange(5)},
        )
        out = _rename_atm_vertical(ds)
        assert "Zsigma" in out.dims
        assert "Zlsigma" in out.dims

    def test_no_op_when_no_z_dim(self):
        ds = xr.Dataset(
            data_vars={"x": (("face", "j", "i"), np.zeros((6, 32, 32)))},
        )
        out = _rename_atm_vertical(ds)
        assert set(out.dims) == set(ds.dims)


# ─── Integration via monkeypatched xmitgcm ───────────────────────────────────

def _make_synthetic_atm(nt: int = 3) -> xr.Dataset:
    """Build a synthetic atm-side Dataset shaped like xmitgcm's cs output.

    Mirrors the dims / vars an actual ``open_mdsdataset(..., geometry='cs',
    prefix=['atm_2d','atm_3d'])`` call would produce.
    """
    nr, nf, ny, nx = 5, 6, 32, 32
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            # 2D fields (atm_2d stream)
            "aim_T2m":  (("time", "face", "j", "i"),
                         rng.normal(290, 5, (nt, nf, ny, nx)).astype(np.float32)),
            "aim_TPCP": (("time", "face", "j", "i"),
                         rng.lognormal(0, 1, (nt, nf, ny, nx)).astype(np.float32)),
            # 3D fields (atm_3d stream)
            "UVEL":  (("time", "Z", "face", "j", "i"),
                      rng.normal(0, 5, (nt, nr, nf, ny, nx)).astype(np.float32)),
            "VVEL":  (("time", "Z", "face", "j", "i"),
                      rng.normal(0, 5, (nt, nr, nf, ny, nx)).astype(np.float32)),
            "THETA": (("time", "Z", "face", "j", "i"),
                      rng.normal(290, 30, (nt, nr, nf, ny, nx)).astype(np.float32)),
            "SALT":  (("time", "Z", "face", "j", "i"),
                      rng.normal(1e-3, 1e-3, (nt, nr, nf, ny, nx)).astype(np.float32)),
        },
        coords={
            "time": np.arange(nt) * 86400.0,
            "Z":    np.linspace(50, 1000, nr) * 100.0,  # Pa
            "face": np.arange(nf),
            "XC":   (("face", "j", "i"),
                     np.tile(np.linspace(-180, 180, nx), (nf, ny, 1))
                       .astype(np.float32)),
            "YC":   (("face", "j", "i"),
                     np.tile(np.linspace(-90, 90, ny)[:, None], (nf, 1, nx))
                       .astype(np.float32)),
        },
    )


def _make_synthetic_ocn(nt: int = 3) -> xr.Dataset:
    """Synthetic ocn-side Dataset for the ``ocn_surf`` stream.

    Surface-only: only ``levels(1,1)=1.`` is requested in our
    namelist, so xmitgcm exposes the 2-D fields already collapsed in Z.
    """
    nf, ny, nx = 6, 32, 32
    rng = np.random.default_rng(1)
    return xr.Dataset(
        data_vars={
            "ETAN":     (("time", "face", "j", "i"),
                         rng.normal(0, 0.1, (nt, nf, ny, nx)).astype(np.float32)),
            "THETA":    (("time", "face", "j", "i"),
                         rng.normal(285, 10, (nt, nf, ny, nx)).astype(np.float32)),
            "SALT":     (("time", "face", "j", "i"),
                         rng.normal(35, 1, (nt, nf, ny, nx)).astype(np.float32)),
            "UVEL":     (("time", "face", "j", "i"),
                         rng.normal(0, 0.5, (nt, nf, ny, nx)).astype(np.float32)),
            "VVEL":     (("time", "face", "j", "i"),
                         rng.normal(0, 0.5, (nt, nf, ny, nx)).astype(np.float32)),
            "MXLDEPTH": (("time", "face", "j", "i"),
                         rng.lognormal(3, 1, (nt, nf, ny, nx)).astype(np.float32)),
        },
        coords={
            "time": np.arange(nt) * 86400.0,
            "face": np.arange(nf),
            "XC":   (("face", "j", "i"),
                     np.tile(np.linspace(-180, 180, nx), (nf, ny, 1))
                       .astype(np.float32)),
            "YC":   (("face", "j", "i"),
                     np.tile(np.linspace(-90, 90, ny)[:, None], (nf, 1, nx))
                       .astype(np.float32)),
        },
    )


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    """Create rank_1/ and rank_2/ subdirs with placeholder MDS .meta
    files so the writer's ``has_diagnostics`` gate fires."""
    rd = tmp_path / "run"
    for name, prefixes in (("rank_2", ATM_STREAMS), ("rank_1", OCN_STREAMS)):
        d = rd / name
        d.mkdir(parents=True)
        for p in prefixes:
            (d / f"{p}.0000000010.meta").write_text("# placeholder\n")
            (d / f"{p}.0000000010.data").write_bytes(b"\x00")
    return rd


@pytest.fixture
def fake_xmitgcm(monkeypatch):
    """Replace ``xmitgcm.open_mdsdataset`` with a stub that hands back
    one of the synthetic Datasets above, depending on which rank dir
    it was called for."""
    import xmitgcm

    def fake_open(path: str, **kwargs):
        rd = Path(path).name
        if rd == "rank_2":
            return _make_synthetic_atm()
        if rd == "rank_1":
            return _make_synthetic_ocn()
        raise RuntimeError(f"unexpected rank dir in test: {rd}")

    monkeypatch.setattr(xmitgcm, "open_mdsdataset", fake_open)


class TestWriteCs32Zarr:
    def test_writes_zarr_at_requested_path(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        result = write_cs32_zarr(fake_run_dir, out)
        assert out.is_dir()
        assert (out / ".zgroup").is_file() or (out / "zarr.json").is_file()
        assert result.path == out

    def test_atm_vars_prefixed_in_output(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        result = write_cs32_zarr(fake_run_dir, out)
        # Every atm data variable should be prefixed with "atm_".
        assert all(v.startswith("atm_") for v in result.atm_vars)
        # And UVEL/VVEL/THETA are present (originally collide with ocn).
        assert "atm_UVEL" in result.atm_vars
        assert "atm_THETA" in result.atm_vars

    def test_ocn_vars_prefixed_in_output(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        result = write_cs32_zarr(fake_run_dir, out)
        assert all(v.startswith("ocn_") for v in result.ocn_vars)
        assert "ocn_THETA" in result.ocn_vars      # SST
        assert "ocn_MXLDEPTH" in result.ocn_vars   # mixed-layer depth

    def test_vertical_axis_renamed_for_atm(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        write_cs32_zarr(fake_run_dir, out)
        # Read back and verify the atm UVEL has dim "Zsigma".
        ds = xr.open_zarr(out)
        assert "Zsigma" in ds["atm_UVEL"].dims
        # And there's no plain "Z" lurking on atm fields.
        for v in ds.data_vars:
            if v.startswith("atm_") and "Zsigma" in ds[v].dims:
                assert "Z" not in ds[v].dims

    def test_n_time_in_result(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        result = write_cs32_zarr(fake_run_dir, out)
        # The synthetic datasets both have nt=3.
        assert result.n_time == 3

    def test_attrs_propagated(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        write_cs32_zarr(
            fake_run_dir, out,
            attrs={"run_id": 42, "co2_ppm": 348.0, "gm_kappa": 1000.0},
        )
        ds = xr.open_zarr(out)
        assert ds.attrs.get("run_id") == 42
        assert ds.attrs.get("co2_ppm") == 348.0
        # Default grid attr is also set.
        assert "cs32" in ds.attrs.get("grid", "")

    def test_chunks_along_time(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        write_cs32_zarr(fake_run_dir, out, chunk_time=2)
        ds = xr.open_zarr(out)
        # First time chunk is min(chunk_time, nt) = 2.
        assert ds["atm_aim_T2m"].chunks[0][0] == 2

    def test_overwrite_replaces_existing(
        self, fake_run_dir: Path, fake_xmitgcm, tmp_path: Path
    ):
        out = tmp_path / "run.zarr"
        out.mkdir()
        (out / "stale_file").write_text("leftover")
        write_cs32_zarr(fake_run_dir, out, overwrite=True)
        # Stale leftover gone, fresh contents in place.
        assert not (out / "stale_file").exists()

    def test_missing_rank_dir_is_clear_error(self, tmp_path: Path):
        rd = tmp_path / "incomplete"
        rd.mkdir()
        (rd / "rank_2").mkdir()
        # rank_1 missing.
        with pytest.raises(FileNotFoundError, match="rank_1"):
            write_cs32_zarr(rd, tmp_path / "out.zarr")

    def test_missing_diagnostics_files_is_clear_error(
        self, fake_xmitgcm, tmp_path: Path
    ):
        # rank dirs exist but have no .meta files → writer should
        # surface a helpful error pointing at useDiagnostics.
        rd = tmp_path / "no_diags"
        (rd / "rank_1").mkdir(parents=True)
        (rd / "rank_2").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="useDiagnostics"):
            write_cs32_zarr(rd, tmp_path / "out.zarr")
