"""Tests for ``datagen/cpl_aim_ocn/scripts/finalize_dataset.py``.

We build a small synthetic ``runs/run_XXXX/run.zarr`` tree on disk —
each store has the canonical cs32 schema with XC/YC coords (so the
regridder can build weights), the 35 channel-source streams, and a
short time axis. Then we drive the finalize entry points and assert:

* run discovery skips runs with sibling ``.FAILED`` markers and runs
  whose ``run.zarr`` is missing,
* split assignment is deterministic given a seed,
* materialised splits create symlinks under ``train/val/test``,
* compute_stats writes the right schema/keys.

The synthetic XC/YC use a face-major equiangular tile assignment that
passes the regridder's ``cKDTree`` build without error — the goal is
plumbing, not a faithful cs32 reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datagen.cpl_aim_ocn.channels import (
    ATM_2D_STREAMS,
    ATM_3D_STREAMS,
    ATM_VERTICAL_DIM,
    N_SIGMA,
    OCN_2D_STREAMS,
    channel_names,
)
from datagen.cpl_aim_ocn.scripts.finalize_dataset import (
    compute_stats,
    discover_runs,
    main,
    materialise_splits,
    split_runs,
)


# ─── Synthetic cs32 zarr fixture ────────────────────────────────────────────


def _build_synthetic_xy(nf: int = 6, ny: int = 8, nx: int = 8):
    """Per-face equiangular tiles giving distinct XC/YC sample points.

    Six faces tile (-180,180)x(-90,90) without precise cs32 geometry —
    distinct enough that ``cKDTree.query`` can find unique nearest
    neighbours during regrid weight construction.
    """
    XC = np.zeros((nf, ny, nx), dtype=np.float64)
    YC = np.zeros((nf, ny, nx), dtype=np.float64)
    # Six longitude bands, two latitude bands per face. Just give each
    # tile a plausible spread.
    lon_centers = np.linspace(-150, 150, nf)
    for f in range(nf):
        # Each face gets a small lon/lat patch around its centre.
        local_lon = np.linspace(lon_centers[f] - 25, lon_centers[f] + 25, nx)
        local_lat = np.linspace(-80 + (f % 2) * 80, -10 + (f % 2) * 80, ny)
        lon_grid, lat_grid = np.meshgrid(local_lon, local_lat)
        XC[f] = lon_grid
        YC[f] = lat_grid
    return XC, YC


def _make_run_zarr(out_path: Path, nt: int = 5, seed: int = 0) -> Path:
    """Materialise a small synthetic cs32 run zarr at ``out_path``."""
    nf, ny, nx = 6, 8, 8
    nr = N_SIGMA
    rng = np.random.default_rng(seed)

    XC, YC = _build_synthetic_xy(nf=nf, ny=ny, nx=nx)

    data: dict[str, tuple] = {}
    for v in ATM_2D_STREAMS:
        data[v] = (
            ("time", "face", "j", "i"),
            rng.normal(0, 1, (nt, nf, ny, nx)).astype(np.float32),
        )
    for v in ATM_3D_STREAMS:
        data[v] = (
            ("time", ATM_VERTICAL_DIM, "face", "j", "i"),
            rng.normal(0, 1, (nt, nr, nf, ny, nx)).astype(np.float32),
        )
    for v in OCN_2D_STREAMS:
        data[v] = (
            ("time", "face", "j", "i"),
            rng.normal(0, 1, (nt, nf, ny, nx)).astype(np.float32),
        )

    ds = xr.Dataset(
        data_vars=data,
        coords={
            "time": np.arange(nt) * 86400.0,
            ATM_VERTICAL_DIM: np.linspace(0, 1, nr),
            "XC": (("face", "j", "i"), XC),
            "YC": (("face", "j", "i"), YC),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(out_path), mode="w", consolidated=True)
    return out_path


@pytest.fixture
def runs_tree(tmp_path: Path) -> Path:
    """Six runs: ids 0..4 successful, id 5 marked FAILED.

    Run 0 deliberately omits ``run.zarr`` to test the "missing" branch.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # Run 0: dir exists but no run.zarr → must be skipped.
    (runs_dir / "run_0000").mkdir()
    # Runs 1..4: valid synthetic zarrs.
    for rid in (1, 2, 3, 4):
        _make_run_zarr(runs_dir / f"run_{rid:04d}" / "run.zarr",
                       seed=rid)
    # Run 5: zarr exists but FAILED marker present → must be skipped.
    _make_run_zarr(runs_dir / "run_0005" / "run.zarr", seed=5)
    (runs_dir / "run_0005.FAILED").write_text("{}\n")
    return runs_dir


# ─── discover_runs ──────────────────────────────────────────────────────────


class TestDiscoverRuns:
    def test_skips_failed_marker(self, runs_tree: Path):
        ids = [rid for rid, _ in discover_runs(runs_tree)]
        assert 5 not in ids

    def test_skips_missing_run_zarr(self, runs_tree: Path):
        ids = [rid for rid, _ in discover_runs(runs_tree)]
        assert 0 not in ids

    def test_returns_sorted_successful_runs(self, runs_tree: Path):
        ids = [rid for rid, _ in discover_runs(runs_tree)]
        assert ids == [1, 2, 3, 4]

    def test_paths_point_to_existing_zarrs(self, runs_tree: Path):
        for _, p in discover_runs(runs_tree):
            assert p.is_dir()
            assert p.name == "run.zarr"


# ─── split_runs ─────────────────────────────────────────────────────────────


class TestSplitRuns:
    def test_totals_match_input(self):
        ids = list(range(180))
        s = split_runs(ids, ratios=(0.8, 0.1, 0.1), seed=0)
        assert sum(len(s[k]) for k in ("train", "val", "test")) == 180

    def test_default_ratios_80_10_10(self):
        ids = list(range(180))
        s = split_runs(ids, ratios=(0.8, 0.1, 0.1), seed=0)
        assert len(s["train"]) == 144
        assert len(s["val"]) == 18
        assert len(s["test"]) == 18

    def test_no_overlap(self):
        ids = list(range(50))
        s = split_runs(ids, seed=42)
        assert set(s["train"]).isdisjoint(s["val"])
        assert set(s["train"]).isdisjoint(s["test"])
        assert set(s["val"]).isdisjoint(s["test"])

    def test_is_deterministic_given_seed(self):
        ids = list(range(50))
        a = split_runs(ids, seed=42)
        b = split_runs(ids, seed=42)
        assert a == b

    def test_different_seed_gives_different_split(self):
        ids = list(range(50))
        a = split_runs(ids, seed=1)
        b = split_runs(ids, seed=2)
        assert a != b

    def test_negative_ratio_rejected(self):
        with pytest.raises(ValueError):
            split_runs([0, 1], ratios=(0.5, -0.1, 0.1), seed=0)

    def test_oversized_ratios_rejected(self):
        with pytest.raises(ValueError):
            split_runs([0, 1], ratios=(0.9, 0.5, 0.1), seed=0)


# ─── materialise_splits (symlinks vs copies) ────────────────────────────────


class TestMaterialiseSplits:
    def test_creates_split_symlinks(self, tmp_path: Path):
        # Synthetic per-run zarr-shaped dirs (don't need real zarr content
        # for symlink test).
        runs = {
            i: tmp_path / "runs" / f"run_{i:04d}" / "run.zarr"
            for i in range(3)
        }
        for p in runs.values():
            p.mkdir(parents=True)
            (p / "data").write_text("x")  # so it's a non-empty dir

        out = tmp_path / "out"
        splits = {"train": [0, 1], "val": [2], "test": []}
        materialise_splits(runs, splits, out)

        for split, ids in splits.items():
            for rid in ids:
                link = out / split / f"run_{rid:04d}.zarr"
                assert link.is_symlink()
                assert link.resolve() == runs[rid].resolve()

    def test_idempotent_replaces_existing(self, tmp_path: Path):
        # First run.
        runs1 = {0: tmp_path / "a" / "run_0000" / "run.zarr"}
        runs1[0].mkdir(parents=True)
        out = tmp_path / "out"
        materialise_splits(runs1, {"train": [0], "val": [], "test": []}, out)
        link = out / "train" / "run_0000.zarr"
        assert link.resolve() == runs1[0].resolve()

        # Second run with a different source path; should replace.
        runs2 = {0: tmp_path / "b" / "run_0000" / "run.zarr"}
        runs2[0].mkdir(parents=True)
        materialise_splits(runs2, {"train": [0], "val": [], "test": []}, out)
        assert link.resolve() == runs2[0].resolve()


# ─── End-to-end main() ──────────────────────────────────────────────────────


class TestMainEndToEnd:
    def test_writes_splits_and_stats_json(self, runs_tree: Path, tmp_path: Path):
        out_root = tmp_path / "ds"
        rc = main([
            "--runs-dir", str(runs_tree),
            "--out-root", str(out_root),
            # Tiny target grid to keep the stats-pass cheap.
            "--nlat", "8", "--nlon", "16",
            # 4 successful runs (ids 1..4) → keep them all in train so
            # stats has data even with rounding-down. Use 1.0/0.0/0.0
            # to send them all to train.
            "--ratios", "1.0", "0.0", "0.0",
        ])
        assert rc == 0

        splits_path = out_root / "splits.json"
        stats_path = out_root / "stats.json"
        assert splits_path.is_file()
        assert stats_path.is_file()

        with open(splits_path) as f:
            splits = json.load(f)
        assert sorted(splits["train"]) == [1, 2, 3, 4]
        assert splits["val"] == []
        assert splits["test"] == []

        with open(stats_path) as f:
            stats = json.load(f)
        assert set(stats.keys()) == set(channel_names())
        for name, entry in stats.items():
            assert set(entry.keys()) == {"mean", "std", "mean_delta", "std_delta"}
            for v in entry.values():
                assert isinstance(v, (int, float))

    def test_skip_stats_does_not_write_stats_json(
        self, runs_tree: Path, tmp_path: Path,
    ):
        out_root = tmp_path / "ds"
        rc = main([
            "--runs-dir", str(runs_tree),
            "--out-root", str(out_root),
            "--ratios", "1.0", "0.0", "0.0",
            "--skip-stats",
        ])
        assert rc == 0
        assert (out_root / "splits.json").is_file()
        assert not (out_root / "stats.json").exists()


# ─── compute_stats direct-call ──────────────────────────────────────────────


class TestComputeStats:
    def test_keys_match_channel_names(self, runs_tree: Path):
        zarrs = sorted(p for p in (runs_tree).rglob("run.zarr"))
        # Drop the one with a FAILED sibling.
        zarrs = [
            p for p in zarrs
            if not (runs_tree / f"{p.parent.name}.FAILED").exists()
        ]
        stats = compute_stats(zarrs, nlat=8, nlon=16)
        assert set(stats.keys()) == set(channel_names())

    def test_std_is_nonnegative(self, runs_tree: Path):
        zarrs = sorted(p for p in (runs_tree).rglob("run.zarr"))
        zarrs = [
            p for p in zarrs
            if not (runs_tree / f"{p.parent.name}.FAILED").exists()
        ]
        stats = compute_stats(zarrs, nlat=8, nlon=16)
        for entry in stats.values():
            assert entry["std"] >= 0
            assert entry["std_delta"] >= 0
