"""Smoke test for the global-ocean DataModule.

Skips if the dataset isn't available locally (CI). On a dev machine with
the dataset at ``$DATA_ROOT/global-ocean`` (or the default sciCORE GROUP
path), exercises the full pipeline: regrid, mask, doy channels, masked
loss.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

_DEFAULT_ROOT = Path(
    "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean"
)


def _dataset_root() -> Path:
    p = Path(os.environ.get("DATA_ROOT", "")) / "global-ocean"
    if p.is_dir():
        return p
    if _DEFAULT_ROOT.is_dir():
        return _DEFAULT_ROOT
    pytest.skip("global-ocean dataset not found locally")


def test_global_ocean_datamodule_smoke():
    from fots.data.global_ocean import (
        N_DYNAMIC,
        N_INPUT_CHANNELS,
        GlobalOceanDataModule,
    )

    root = _dataset_root()
    dm = GlobalOceanDataModule(
        root=str(root),
        time_steps_per_run=1189,
        dim_in=N_INPUT_CHANNELS,
        dim_out=N_DYNAMIC,
        spatial_resolution=(64, 128),
        batch_size=2,
        n_steps_input=4,
        n_steps_output=1,
        num_workers=0,
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert batch["input_fields"].shape == (2, 4, 8, 64, 128)
    assert batch["output_fields"].shape == (2, 1, 5, 64, 128)
    assert batch["valid_mask"].shape == (2, 5, 64, 128)
    assert torch.isfinite(batch["input_fields"]).all()
    assert torch.isfinite(batch["output_fields"]).all()

    # Mask is bool-like: only 0 and 1 values.
    mvals = torch.unique(batch["valid_mask"]).tolist()
    assert set(mvals).issubset({0.0, 1.0})

    # depth channel (index 5) should be constant across time within a sample.
    depth_channel = batch["input_fields"][0, :, 5]
    assert torch.allclose(depth_channel[0], depth_channel[-1])

    # doy channels (6, 7) should vary across time within a sample
    # (sin, cos move from one snapshot to the next).
    doy_sin = batch["input_fields"][0, :, 6, 0, 0]  # spatially-constant
    assert torch.unique(doy_sin).numel() >= 2

    # Loss with mask is finite. Use the actual loss class the trainer
    # uses to verify the kwarg interface.
    from fots.metrics import LatitudeWeightedMSELoss
    loss_fn = LatitudeWeightedMSELoss(nlat=64)
    y_pred = torch.zeros_like(batch["output_fields"][:, 0])
    y = batch["output_fields"][:, 0]
    mask = batch["valid_mask"]
    out = loss_fn(y_pred, y, mask=mask)
    assert torch.isfinite(out)


def test_rotate_uv_to_geographic_shapes():
    """Rotation is a local linear combine; shape and identity-at-zero check."""
    from datagen.mitgcm.global_ocean.regrid import rotate_uv_to_geographic
    rng = np.random.default_rng(0)
    u = rng.standard_normal((3, 6, 32, 32)).astype(np.float32)
    v = rng.standard_normal((3, 6, 32, 32)).astype(np.float32)
    cs = np.ones((6, 32, 32), dtype=np.float32)
    sn = np.zeros((6, 32, 32), dtype=np.float32)
    u_e, v_n = rotate_uv_to_geographic(u, v, cs, sn)
    np.testing.assert_allclose(u_e, u)
    np.testing.assert_allclose(v_n, v)

    # 90° rotation: cs=0, sn=1 → (u_e, v_n) = (-v, u)
    cs = np.zeros((6, 32, 32), dtype=np.float32)
    sn = np.ones((6, 32, 32), dtype=np.float32)
    u_e, v_n = rotate_uv_to_geographic(u, v, cs, sn)
    np.testing.assert_allclose(u_e, -v)
    np.testing.assert_allclose(v_n, u)
