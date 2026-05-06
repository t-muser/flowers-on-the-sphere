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


def test_apply_dynamic_3d_squashes_levels_into_channels():
    """3-D regrid: levels squashed into channels, u/v rotated, eta last."""
    from datagen.cpl_aim_ocn.regrid import build_weights
    from datagen.mitgcm.global_ocean.regrid import (
        FIELDS_2D,
        FIELDS_3D,
        GlobalOceanLatLon,
        apply_dynamic_3d,
        field_names_3d,
    )

    nlat, nlon = 8, 16
    n_face, fs = 6, 4
    Nt, Nlevel = 2, 3
    rng = np.random.default_rng(0)
    # Synthetic monotone grid is enough for the IDW weights to build.
    xc = rng.uniform(-180, 180, size=(n_face, fs, fs))
    yc = rng.uniform(-89, 89, size=(n_face, fs, fs))
    weights = build_weights(xc, yc, nlat=nlat, nlon=nlon, method="idw", k=4)

    def _grid(angle_cs_val: float, angle_sn_val: float) -> GlobalOceanLatLon:
        return GlobalOceanLatLon(
            nlat=nlat, nlon=nlon, weights=weights,
            angle_cs=np.full((n_face, fs, fs), angle_cs_val, dtype=np.float32),
            angle_sn=np.full((n_face, fs, fs), angle_sn_val, dtype=np.float32),
            depth_ll=np.zeros((nlat, nlon), dtype=np.float32),
            mask_k1_ll=np.ones((nlat, nlon), dtype=bool),
            mask_k2_ll=np.ones((nlat, nlon), dtype=bool),
            mask_eta_ll=np.ones((nlat, nlon), dtype=bool),
        )

    fields_3d = {
        name: rng.standard_normal((Nt, Nlevel, n_face, fs, fs)).astype(np.float32)
        for name in FIELDS_3D
    }
    fields_2d = {
        name: rng.standard_normal((Nt, n_face, fs, fs)).astype(np.float32)
        for name in FIELDS_2D
    }

    # Identity rotation: angle_cs=1, angle_sn=0 → u/v pass through unchanged.
    out = apply_dynamic_3d(fields_3d, fields_2d, _grid(1.0, 0.0))

    n_3d_chans = len(FIELDS_3D) * Nlevel
    n_2d_chans = len(FIELDS_2D)
    assert out.shape == (Nt, n_3d_chans + n_2d_chans, nlat, nlon)
    assert out.dtype == np.float32

    names = field_names_3d(np.array([1, 2, 3]))
    assert names == (
        "theta_k01", "theta_k02", "theta_k03",
        "salt_k01",  "salt_k02",  "salt_k03",
        "u_k01",     "u_k02",     "u_k03",
        "v_k01",     "v_k02",     "v_k03",
        "eta",
    )

    # 90° rotation flips u → -v, v → u. Regrid is linear, so the channel-axis
    # offsets for the u and v slabs swap with a sign flip post-regrid.
    out_rot = apply_dynamic_3d(fields_3d, fields_2d, _grid(0.0, 1.0))
    u_off = FIELDS_3D.index("u") * Nlevel
    v_off = FIELDS_3D.index("v") * Nlevel
    np.testing.assert_allclose(
        out_rot[:, u_off:u_off + Nlevel],
        -out[:, v_off:v_off + Nlevel],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        out_rot[:, v_off:v_off + Nlevel],
        out[:, u_off:u_off + Nlevel],
        atol=1e-5,
    )


def test_apply_dynamic_3d_impute_land_kills_zero_bleed():
    """impute_land=True should remove the 0.0 land-fill bleed at deep levels."""
    from datagen.cpl_aim_ocn.regrid import build_weights
    from datagen.mitgcm.global_ocean.regrid import (
        FIELDS_2D,
        FIELDS_3D,
        GlobalOceanLatLon,
        apply_dynamic_3d,
    )

    nlat, nlon = 8, 16
    n_face, fs = 6, 4
    Nlevel = 3
    rng = np.random.default_rng(2)
    xc = rng.uniform(-180, 180, size=(n_face, fs, fs))
    yc = rng.uniform(-89, 89, size=(n_face, fs, fs))
    weights = build_weights(xc, yc, nlat=nlat, nlon=nlon, method="idw", k=4)

    # Per-level mask: ~half wet, half dry, varied per level.
    mask_c = rng.random((Nlevel, n_face, fs, fs)) > 0.5
    mask_w = mask_c.copy()

    grid_ll = GlobalOceanLatLon(
        nlat=nlat, nlon=nlon, weights=weights,
        angle_cs=np.ones((n_face, fs, fs), dtype=np.float32),
        angle_sn=np.zeros((n_face, fs, fs), dtype=np.float32),
        depth_ll=np.zeros((nlat, nlon), dtype=np.float32),
        mask_k1_ll=np.ones((nlat, nlon), dtype=bool),
        mask_k2_ll=np.ones((nlat, nlon), dtype=bool),
        mask_eta_ll=np.ones((nlat, nlon), dtype=bool),
        mask_c_3d_src=mask_c, mask_w_3d_src=mask_w,
    )

    # Build wet values around 34.7 (mimicking salt at depth) with land = 0.
    Nt = 1
    fields_3d = {}
    for var in FIELDS_3D:
        arr = np.full((Nt, Nlevel, n_face, fs, fs), 34.7, dtype=np.float32)
        for k in range(Nlevel):
            arr[:, k][..., ~mask_c[k]] = 0.0  # land-fill
        fields_3d[var] = arr
    fields_2d = {"eta": np.zeros((Nt, n_face, fs, fs), dtype=np.float32)}

    out_off = apply_dynamic_3d(fields_3d, fields_2d, grid_ll, impute_land=False)
    out_on = apply_dynamic_3d(fields_3d, fields_2d, grid_ll, impute_land=True)

    # With impute off: salt slabs in lat/lon contain values mixed between
    # 34.7 (wet) and 0 (land bleed) — std should be large.
    salt_off = out_off[:, FIELDS_3D.index("salt") * Nlevel:
                          (FIELDS_3D.index("salt") + 1) * Nlevel]
    salt_on = out_on[:, FIELDS_3D.index("salt") * Nlevel:
                        (FIELDS_3D.index("salt") + 1) * Nlevel]
    # impute_on values cluster around 34.7 (the wet mean) almost everywhere.
    assert salt_on.std() < salt_off.std() / 5
    assert abs(salt_on.mean() - 34.7) < 0.5
    # impute_off shows much wider spread because of the 0 bleed.
    assert salt_off.std() > 3.0


def test_apply_dynamic_3d_level_subset_via_level_idx():
    """Passing level_idx=[1, 3] should pick the right per-level src masks."""
    from datagen.cpl_aim_ocn.regrid import build_weights
    from datagen.mitgcm.global_ocean.regrid import (
        FIELDS_3D,
        GlobalOceanLatLon,
        apply_dynamic_3d,
    )

    nlat, nlon = 8, 16
    n_face, fs = 6, 4
    Nr = 4
    rng = np.random.default_rng(3)
    xc = rng.uniform(-180, 180, size=(n_face, fs, fs))
    yc = rng.uniform(-89, 89, size=(n_face, fs, fs))
    weights = build_weights(xc, yc, nlat=nlat, nlon=nlon, method="idw", k=4)

    # Distinct per-level masks so picking level=1 vs level=3 changes which
    # cells are imputed.
    mask_c = np.zeros((Nr, n_face, fs, fs), dtype=bool)
    mask_c[0] = True   # level 1: all wet
    mask_c[1] = False  # level 2: all dry
    mask_c[2, :3] = True  # level 3: first 3 faces wet
    mask_c[3] = False
    mask_w = mask_c.copy()

    grid_ll = GlobalOceanLatLon(
        nlat=nlat, nlon=nlon, weights=weights,
        angle_cs=np.ones((n_face, fs, fs), dtype=np.float32),
        angle_sn=np.zeros((n_face, fs, fs), dtype=np.float32),
        depth_ll=np.zeros((nlat, nlon), dtype=np.float32),
        mask_k1_ll=np.ones((nlat, nlon), dtype=bool),
        mask_k2_ll=np.ones((nlat, nlon), dtype=bool),
        mask_eta_ll=np.ones((nlat, nlon), dtype=bool),
        mask_c_3d_src=mask_c, mask_w_3d_src=mask_w,
    )

    Nt, Nsel = 1, 2
    # Two-level subset corresponding to model levels 1 and 3.
    fields_3d = {
        var: np.full((Nt, Nsel, n_face, fs, fs), 5.0, dtype=np.float32)
        for var in FIELDS_3D
    }
    # Level 0 in the input (= model level 1, all wet): no imputation needed.
    # Level 1 in the input (= model level 3): mask faces 3-5 dry; pre-fill
    # faces 3-5 with 0.0 to simulate land-fill, then expect impute to lift
    # those cells to the wet mean (≈5.0).
    for var in FIELDS_3D:
        fields_3d[var][:, 1, 3:] = 0.0  # land cells per level 3 mask

    out = apply_dynamic_3d(
        fields_3d, {"eta": np.zeros((Nt, n_face, fs, fs), dtype=np.float32)},
        grid_ll,
        level_idx=np.array([1, 3], dtype=np.int64),
        impute_land=True,
    )

    # All channels should now be ~5.0 across the whole sphere; bleed gone.
    salt_off = FIELDS_3D.index("salt") * Nsel
    np.testing.assert_allclose(out[:, salt_off:salt_off + Nsel].mean(),
                               5.0, atol=0.5)
