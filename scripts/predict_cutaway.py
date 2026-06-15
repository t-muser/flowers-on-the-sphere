"""Autoregressive rollout of a trained model on the Held-Suarez (ClimaAtmos)
HDF5 dataset, rendered as a prediction-vs-truth cutaway-sphere MP4.

The cutaway-sphere renderer is lifted verbatim from
``datagen/clima_atmos/held_suarez/scripts/render_corner.py`` (itself lifted from
the held_suarez_clima visualization notebook) so the prediction video matches
the ground-truth "corner" renders we ship for this dataset.

Pipeline:
  1. Build the datamodule + model exactly as ``fots.train`` does (the
     ``fold_levels_into_channels`` config makes this a 2-D, 25-channel model:
     channels ``[T_lev0..7, ps, u_lev0..7, v_lev0..7]``).
  2. Load ``best.pt`` (or run with random weights for a pipeline smoke test).
  3. Take one full test trajectory, warm up on ``n_steps_input`` frames, then
     roll the model forward ``--steps`` steps autoregressively (normalized
     space, sliding the history buffer as the Trainer's rollout loop does).
  4. Extract predicted & ground-truth zonal wind ``u`` (channels 9..16 -> the
     8 pressure levels), denormalize to m/s, and render the cutaway MP4.

Needs a GPU for the real (lifting_dim=160) model -> run via
``scripts/predict_cutaway.sbatch``. ``--device cpu --lifting-dim 40 --steps 2``
is a light pipeline smoke test.

Run::

    uv run --no-sync python scripts/predict_cutaway.py \\
        --ckpt checkpoints/held_suarez_clima-held_suarez_hdf5-Flower2D-0.0005/best.pt \\
        --steps 160 --out-dir notebooks/figures/clima_predict_renders
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import trange

from fots.train import build_model, load_modular_config

# Held-Suarez ClimaAtmos pressure levels (hPa), in stored level order lev0..lev7.
LEVELS_HPA = np.array([50.0, 100.0, 250.0, 500.0, 700.0, 850.0, 925.0, 1000.0])
# Folded channel layout (see module docstring): u occupies channels 9..16.
U_CHANNEL_START = 9
N_LEVELS = 8

DEFAULT_DATA_ROOT = "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-clima-hdf5"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------------------------------------------------------
# Cutaway-sphere renderer — lifted verbatim from render_corner.py::cutaway_sphere
# (originally from the held_suarez_clima visualization notebook).
# ----------------------------------------------------------------------------
def cutaway_sphere(
    field_3d, ocean_mask_3d, level_depths, lat, lon,
    *, vmin, vmax, cmap, lon_wedge=(-30.0, 60.0),
    r_outer=1.0, r_inner=0.55,
    elev=20.0, azim=None, title="", ax=None,
):
    """Render one cutaway-sphere frame."""
    NLAT, NLON = len(lat), len(lon)
    Nlevel = field_3d.shape[0]

    LON1, LON2 = float(lon_wedge[0]), float(lon_wedge[1])
    LON1n = ((LON1 + 180) % 360) - 180
    LON2n = ((LON2 + 180) % 360) - 180
    span = (LON2n - LON1n) % 360.0
    if span <= 0 or span >= 360:
        raise ValueError(f"degenerate wedge {lon_wedge}")

    max_d = float(level_depths.max())
    r_per_level = r_outer - (r_outer - r_inner) * (level_depths / max_d)

    in_wedge_col = ((lon - LON1n) % 360.0) < span
    n_wedge = int(in_wedge_col.sum())
    n_out = NLON - n_wedge
    starts = np.where(in_wedge_col & ~np.roll(in_wedge_col, 1))[0]
    if not len(starts):
        raise ValueError("invalid wedge")
    first_nonwedge_idx = (starts[0] + n_wedge) % NLON
    shift = -first_nonwedge_idx
    lon_r = np.roll(lon, shift)
    in_wedge_r = np.roll(in_wedge_col, shift)
    assert not in_wedge_r[:n_out].any()
    orig_idx_r = np.roll(np.arange(NLON), shift)

    M = Nlevel + n_out + Nlevel + n_wedge
    r_col = np.empty(M, dtype=np.float32)
    lon_col = np.empty(M, dtype=np.float32)
    sec_id = np.empty(M, dtype=np.int8)
    level_per = np.full(M, -1, dtype=np.int32)
    sphere_lon_idx = np.full(M, -1, dtype=np.int32)

    for k in range(Nlevel):
        lev = Nlevel - 1 - k
        r_col[k] = r_per_level[lev]; lon_col[k] = LON2n
        sec_id[k] = 0; level_per[k] = lev
    for k in range(n_out):
        c = Nlevel + k
        r_col[c] = r_outer; lon_col[c] = lon_r[k]
        sec_id[c] = 1; sphere_lon_idx[c] = orig_idx_r[k]
    for k in range(Nlevel):
        c = Nlevel + n_out + k
        r_col[c] = r_per_level[k]; lon_col[c] = LON1n
        sec_id[c] = 2; level_per[c] = k
    for k in range(n_wedge):
        c = Nlevel + n_out + Nlevel + k
        r_col[c] = r_inner; lon_col[c] = lon_r[n_out + k]
        sec_id[c] = 3; sphere_lon_idx[c] = orig_idx_r[n_out + k]
        level_per[c] = Nlevel - 1

    PHI_lat = np.deg2rad(90.0 - lat)
    THETA = np.deg2rad(lon_col)[None, :]
    R = r_col[None, :]
    PHI = PHI_lat[:, None]
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    data = np.empty((NLAT, M), dtype=np.float32)
    mask = np.empty((NLAT, M), dtype=bool)
    ix_lon1 = int(np.argmin(np.abs(lon - LON1n)))
    ix_lon2 = int(np.argmin(np.abs(lon - LON2n)))
    for c in range(M):
        if sec_id[c] == 0:
            lev = int(level_per[c])
            data[:, c] = field_3d[lev, :, ix_lon2]
            mask[:, c] = ocean_mask_3d[lev, :, ix_lon2]
        elif sec_id[c] == 1:
            j = int(sphere_lon_idx[c])
            data[:, c] = field_3d[0, :, j]
            mask[:, c] = ocean_mask_3d[0, :, j]
        elif sec_id[c] == 2:
            lev = int(level_per[c])
            data[:, c] = field_3d[lev, :, ix_lon1]
            mask[:, c] = ocean_mask_3d[lev, :, ix_lon1]
        else:
            j = int(sphere_lon_idx[c])
            data[:, c] = field_3d[Nlevel - 1, :, j]
            mask[:, c] = ocean_mask_3d[Nlevel - 1, :, j]

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fc = cmap(norm(np.where(mask, data, vmin)))
    fc[~mask] = (0.8275, 0.8275, 0.8275, 1.)

    if azim is None:
        mid = LON1n + 0.5 * span
        azim = ((mid + 180) % 360) - 180
    cu = np.array([
        np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
        np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),
        np.sin(np.deg2rad(elev)),
    ])
    skin_cols = ((sec_id == 1) | (sec_id == 3))[None, :]
    dot = X * cu[0] + Y * cu[1] + Z * cu[2]
    drop = (dot < 0.0) & skin_cols
    X = np.where(drop, np.nan, X)
    Y = np.where(drop, np.nan, Y)
    Z = np.where(drop, np.nan, Z)

    if ax is None:
        fig = plt.figure(figsize=(8, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                    antialiased=False, linewidth=0, shade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    return ax


# ----------------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------------
def _load_cfg(args) -> "DictConfig":  # noqa: F821
    cfg = load_modular_config(
        "configs/data/held_suarez_hdf5.yaml",
        args.model_config,
        "configs/train_4-to-1.yaml",
    )
    cfg.data.path = args.data_path
    cfg.data.batch_size = 1
    # One window per test trajectory covering warm-up + the full rollout horizon.
    cfg.data.n_steps_output = args.steps
    if args.lifting_dim is not None:
        cfg.model.lifting_dim = args.lifting_dim
        # groups/num_heads must divide lifting_dim at every U-Net level.
        cfg.model.groups = args.lifting_dim
        cfg.model.num_heads = args.lifting_dim
    return cfg


@torch.inference_mode()
def rollout(args):
    log = logging.getLogger(__name__)
    device = torch.device(args.device)
    cfg = _load_cfg(args)

    log.info("Instantiating datamodule (folded 25-channel 2-D view)…")
    dm = instantiate(cfg.data)
    md = dm.metadata
    field_names = list(md.field_names)
    assert field_names[U_CHANNEL_START:U_CHANNEL_START + N_LEVELS] == [
        f"u_lev{i}" for i in range(N_LEVELS)
    ], f"unexpected channel layout: {field_names}"
    n_field_ch = md.dim_out  # 25

    model = build_model(cfg, dm).to(device)
    if args.ckpt:
        log.info("Loading checkpoint %s", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded weights from epoch %s", ckpt.get("epoch", "?"))
    else:
        log.warning("No --ckpt given: running with RANDOM weights (smoke test).")
    model.eval()

    # One full-trajectory window: input_fields (1,Ti,C,H,W), output (1,steps,C,H,W).
    test_ds = dm.test_dataset
    sample = test_ds[args.traj_index]
    x = sample["input_fields"].unsqueeze(0).to(device)   # (1, Ti, C, H, W)
    y = sample["output_fields"].unsqueeze(0).to(device)  # (1, steps, C, H, W)
    Ti = x.shape[1]
    steps = y.shape[1]
    log.info("Trajectory %d: warm-up %d frames, rolling %d steps (%.1f days @ 6h)",
             args.traj_index, Ti, steps, steps * 6 / 24)

    hist = x.reshape(1, Ti * n_field_ch, *x.shape[3:])  # fold time into channels
    preds = []
    for k in trange(steps, desc="rollout"):
        y_pred = model(hist)                       # (1, C, H, W) normalized
        preds.append(y_pred[:, U_CHANNEL_START:U_CHANNEL_START + N_LEVELS].cpu())
        if k < steps - 1:
            hist = torch.cat([hist[:, n_field_ch:], y_pred], dim=1)

    pred_u = torch.cat(preds, dim=0).squeeze(1)  # (steps, 8, H, W) normalized
    true_u = y[0, :, U_CHANNEL_START:U_CHANNEL_START + N_LEVELS].cpu()  # (steps,8,H,W)

    # Denormalize u per level using the folded per-channel stats.
    mean, std = test_ds.folded_denorm_stats
    u_mean = mean[U_CHANNEL_START:U_CHANNEL_START + N_LEVELS].view(1, N_LEVELS, 1, 1)
    u_std = std[U_CHANNEL_START:U_CHANNEL_START + N_LEVELS].view(1, N_LEVELS, 1, 1)
    pred_u = (pred_u * u_std + u_mean).numpy()
    true_u = (true_u * u_std + u_mean).numpy()

    # lat/lon are identical across files; read them from the HDF5 dimensions.
    import h5py
    with h5py.File(test_ds.files_paths[0], "r") as f:
        lat = np.asarray(f["dimensions"]["lat"])
        lon = np.asarray(f["dimensions"]["lon"])
    return pred_u, true_u, lat, lon


def _cutaway_prep(u_seq):
    """Replicate render_corner.render_mp4 pre-processing: drop sponge levels
    (p<=100 hPa), order by descending pressure (surface outer), build the
    pseudo-depth radius profile. ``u_seq`` is (T, 8, lat, lon)."""
    keep = LEVELS_HPA > 100.0
    u = u_seq[:, keep]
    p = LEVELS_HPA[keep]
    order = np.argsort(-p)
    u = u[:, order]
    p = p[order]
    pseudo_depth = (p.max() - p).astype(np.float32)
    return u, pseudo_depth


def render(args, pred_u, true_u, lat, lon):
    log = logging.getLogger(__name__)
    pred, pseudo_depth = _cutaway_prep(pred_u)
    true, _ = _cutaway_prep(true_u)
    mask3d = np.ones(pred.shape[1:], dtype=bool)

    # Shared symmetric color scale from the ground truth.
    mv = float(np.nanmax(np.abs(true))) * 0.9
    vmin, vmax = -mv, mv
    cmap = plt.get_cmap("RdBu_r")

    n = pred.shape[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if not args.ckpt else Path(args.ckpt).parent.name
    mp4 = out_dir / f"predict_cutaway_{tag}_traj{args.traj_index}.mp4"
    log.info("Rendering %d frames (pred|truth) @ %d dpi -> %s", n, args.dpi, mp4)

    frames = []
    for i in trange(n, desc="render"):
        fig = plt.figure(figsize=(15, 8), constrained_layout=True, dpi=args.dpi)
        if args.rotate:
            azim = args.azim + (i / max(n, 1)) * 360.0
            azim = ((azim + 180.0) % 360.0) - 180.0
        else:
            azim = args.azim  # static: keep the camera on the cut face
        lead_d = (i + 1) * 6 / 24.0
        for j, (data, label) in enumerate(((pred, "prediction"), (true, "ground truth"))):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            cutaway_sphere(
                data[i], mask3d, pseudo_depth, lat, lon,
                vmin=vmin, vmax=vmax, cmap=cmap,
                lon_wedge=(-15.0, 165.0), r_inner=0.78,
                elev=25.0, azim=azim, ax=ax,
                title=f"u — {label}",
            )
        fig.suptitle(f"Flower2D rollout  |  lead +{lead_d:.2f} d", fontsize=15)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        # Crop to a multiple of 16 so libx264 doesn't silently resize.
        frame = frame[: h - (h % 16), : w - (w % 16), :3]
        frames.append(frame)
        plt.close(fig)

    iio.imwrite(str(mp4), np.stack(frames, 0), fps=args.fps,
                codec="libx264", pixelformat="yuv420p")
    log.info("Wrote %s (%d frames @ %d fps)", mp4, len(frames), args.fps)
    return mp4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="",
                    help="Path to best.pt; empty -> random weights (smoke test).")
    ap.add_argument("--model-config", default="configs/models/flower.yaml",
                    help="Model config to build (must match the checkpoint's "
                         "architecture, e.g. configs/models/zinnia_v5.yaml).")
    ap.add_argument("--data-path", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--traj-index", type=int, default=0,
                    help="Which test-set trajectory window (0 = first).")
    ap.add_argument("--steps", type=int, default=160,
                    help="Autoregressive rollout steps (6h each; 160 = 40 d).")
    ap.add_argument("--out-dir", default="notebooks/figures/clima_predict_renders")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--azim", type=float, default=75.0,
                    help="Camera azimuth (deg). Default 75 looks into the cut "
                         "wedge; the field's vertical structure stays in view.")
    ap.add_argument("--rotate", action="store_true",
                    help="Rotate the globe over the rollout (default: static).")
    ap.add_argument("--lifting-dim", type=int, default=None,
                    help="Override model width (use a small value for CPU smoke).")
    args = ap.parse_args()

    _setup_logging()
    pred_u, true_u, lat, lon = rollout(args)
    render(args, pred_u, true_u, lat, lon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
