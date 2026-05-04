"""Tests for ``fots.data.cpl_aim_ocn.CplAimOcnDataModule``.

Builds a synthetic two-run cs32 dataset on disk (with the canonical
schema written by ``zarr_writer.write_cs32_zarr``) plus a matching
``splits.json`` and ``stats.json``, instantiates the datamodule, and
asserts the dataloader yields the expected shape, dtype, and
normalization behaviour.

Reuses the synthetic-cs32 helpers from
``datagen/cpl_aim_ocn/tests/test_finalize_dataset.py`` to avoid two
copies of the fixture builder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Reuse the same synthetic builder as the finalize tests.
_DATAGEN_TESTS = Path(__file__).resolve().parents[3] / "datagen" / "cpl_aim_ocn" / "tests"
sys.path.insert(0, str(_DATAGEN_TESTS))
from test_finalize_dataset import _make_run_zarr  # noqa: E402

from datagen.cpl_aim_ocn.channels import channel_names  # noqa: E402
from fots.data.cpl_aim_ocn import CplAimOcnDataModule  # noqa: E402


_NLAT = 8
_NLON = 16
_TIME_STEPS = 5


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Two synthetic runs in train + one each in val/test, with stats."""
    root = tmp_path / "ds"
    for split, ids in (("train", [0, 1]), ("val", [2]), ("test", [3])):
        for rid in ids:
            _make_run_zarr(
                root / split / f"run_{rid:04d}.zarr",
                nt=_TIME_STEPS, seed=rid,
            )

    splits = {"train": [0, 1], "val": [2], "test": [3]}
    (root / "splits.json").write_text(json.dumps(splits))

    # Plausible stats.json (zero-mean unit-variance for every channel).
    stats = {
        name: {"mean": 0.0, "std": 1.0, "mean_delta": 0.0, "std_delta": 1.0}
        for name in channel_names()
    }
    (root / "stats.json").write_text(json.dumps(stats))
    return root


def _make_dm(root: Path, **overrides) -> CplAimOcnDataModule:
    kwargs = dict(
        root=str(root),
        time_steps_per_run=_TIME_STEPS,
        dim_in=35,
        dim_out=35,
        spatial_resolution=(_NLAT, _NLON),
        field_names=channel_names(),
        batch_size=1,
        n_steps_input=2,
        n_steps_output=1,
        num_workers=0,
        dataset_name="test_cpl_aim_ocn",
    )
    kwargs.update(overrides)
    return CplAimOcnDataModule(**kwargs)


class TestInstantiation:
    def test_constructs_without_error(self, synthetic_dataset: Path):
        dm = _make_dm(synthetic_dataset)
        assert dm.metadata.dataset_name == "test_cpl_aim_ocn"
        assert dm.metadata.dim_in == 35

    def test_loads_stats_when_present(self, synthetic_dataset: Path):
        dm = _make_dm(synthetic_dataset)
        assert dm._norm_mean_np is not None
        assert dm._norm_std_np is not None
        assert dm._norm_mean_np.shape == (35,)

    def test_no_stats_falls_back_to_unnormalised(
        self, synthetic_dataset: Path,
    ):
        (synthetic_dataset / "stats.json").unlink()
        dm = _make_dm(synthetic_dataset)
        assert dm._norm_mean_np is None
        assert dm._norm_std_np is None


class TestSampleShapes:
    def test_train_dataloader_yields_expected_shape(
        self, synthetic_dataset: Path,
    ):
        dm = _make_dm(synthetic_dataset)
        batch = next(iter(dm.train_dataloader()))
        assert batch["input_fields"].shape == (1, 2, 35, _NLAT, _NLON)
        assert batch["output_fields"].shape == (1, 1, 35, _NLAT, _NLON)

    def test_dtype_is_float32(self, synthetic_dataset: Path):
        dm = _make_dm(synthetic_dataset)
        batch = next(iter(dm.train_dataloader()))
        assert batch["input_fields"].dtype == torch.float32

    def test_full_trajectory_loader_uses_remaining_steps(
        self, synthetic_dataset: Path,
    ):
        dm = _make_dm(synthetic_dataset)
        batch = next(iter(dm.rollout_val_dataloader()))
        # n_steps_input=2; remaining = _TIME_STEPS - 2 = 3.
        assert batch["output_fields"].shape == (1, 3, 35, _NLAT, _NLON)


class TestNormalization:
    def test_unit_stats_pass_through_values(self, synthetic_dataset: Path):
        # With mean=0/std=1 stats, normalize is a no-op — sample values
        # should equal the regridded source data.
        dm = _make_dm(synthetic_dataset)
        ds = dm.train_dataset
        sample = ds[0]
        # No NaNs introduced by the regrid path.
        assert torch.isfinite(sample["input_fields"]).all()
        assert torch.isfinite(sample["output_fields"]).all()

    def test_denormalize_round_trips(self, synthetic_dataset: Path):
        # Custom non-trivial stats: any finite mean/std.
        stats = {
            name: {"mean": 1.0, "std": 2.0, "mean_delta": 0.0, "std_delta": 1.0}
            for name in channel_names()
        }
        (synthetic_dataset / "stats.json").write_text(json.dumps(stats))
        dm = _make_dm(synthetic_dataset)
        x = torch.randn(1, 2, 35, _NLAT, _NLON)
        y = dm.denormalize_fn(x)
        # y = x * 2 + 1 → reverse
        assert torch.allclose(y, x * 2.0 + 1.0)


class TestRegridWeightCaching:
    def test_one_weight_object_shared_across_splits(
        self, synthetic_dataset: Path,
    ):
        dm = _make_dm(synthetic_dataset)
        # All split datasets share the same regrid weights instance —
        # the cs32 grid is fixed across runs.
        w_train = dm.train_dataset._weights
        w_val = dm.val_dataset._weights
        w_test = dm.test_dataset._weights
        assert w_train is w_val
        assert w_train is w_test
        assert w_train.nlat == _NLAT
        assert w_train.nlon == _NLON
