"""Autoregressive rollout of a trained model on a zarr spherical dataset,
rendered as a prediction-vs-truth rotating-globe MP4.

Matches the spherical (Orthographic) animation style of the dataset
visualization notebooks (notebooks/galewsky_visualize.ipynb,
notebooks/shock_caps_visualize.ipynb): one rotating globe per field, RdBu_r,
``add_cyclic_point`` at the dateline, central_latitude=30, azimuth sweeping
360 over the rollout. Here we put the model prediction and the ground-truth
trajectory side by side for the same notebook-chosen run.

Pipeline mirrors scripts/predict_cutaway.py but for the 2-D zarr datasets
(ZarrDataModule): warm up on ``n_steps_input`` frames, roll the model forward
over the full test trajectory in normalized space (sliding the history buffer
as the Trainer's rollout loop does), then denormalize via the datamodule's
per-channel stats and render the chosen field.

Needs a GPU for the real (lifting_dim=160, 256x512) model -> run via
scripts/predict_globe.sbatch. ``--device cpu --lifting-dim 40 --steps 2`` is a
light pipeline smoke test.

Run::

    uv run --no-sync python scripts/predict_globe.py --dataset galewsky \\
        --ckpt checkpoints/galewsky-galewsky-Flower2D-0.0005/best.pt
    uv run --no-sync python scripts/predict_globe.py --dataset shock_caps \\
        --ckpt checkpoints/shock_caps-shock_caps-Flower2D-0.001/best.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from cartopy.util import add_cyclic_point
from hydra.utils import instantiate
from tqdm import trange

from fots.train import build_model, load_modular_config

SCICORE_ROOT = Path("/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs")

# Per-dataset recipe. `field`/`run_id` follow the visualization notebooks;
# `symmetric` picks a ±max color scale (vorticity) vs. a percentile min/max
# scale (height). data_config drives field order + model in/out channels.
DATASETS = {
    "galewsky": dict(
        data_config="configs/data/galewsky.yaml",
        root=SCICORE_ROOT / "galewsky-sw",
        field="vorticity",
        run_id=49,
        symmetric=True,
    ),
    "shock_caps": dict(
        data_config="configs/data/shock_caps.yaml",
        root=SCICORE_ROOT / "shock-caps",
        field="height",
        run_id=37,
        symmetric=False,
    ),
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )


def _load_cfg(args, spec):
    cfg = load_modular_config(
        spec["data_config"], "configs/models/flower.yaml", "configs/train_4-to-1.yaml"
    )
    cfg.data.root = str(spec["root"])
    cfg.data.batch_size = 1
    if args.lifting_dim is not None:
        cfg.model.lifting_dim = args.lifting_dim
        cfg.model.groups = args.lifting_dim
        cfg.model.num_heads = args.lifting_dim
    return cfg


@torch.inference_mode()
def rollout(args, spec):
    log = logging.getLogger(__name__)
    device = torch.device(args.device)
    cfg = _load_cfg(args, spec)

    dm = instantiate(cfg.data)
    field_names = list(dm.metadata.field_names)
    field_idx = field_names.index(spec["field"])
    n_field_ch = dm.metadata.dim_out
    log.info("Fields %s; visualizing '%s' (channel %d)",
             field_names, spec["field"], field_idx)

    model = build_model(cfg, dm).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded %s (epoch %s)", args.ckpt, ckpt.get("epoch", "?"))
    else:
        log.warning("No --ckpt: RANDOM weights (smoke test).")
    model.eval()

    # Full-trajectory window for the notebook-chosen run.
    rds = dm.rollout_test_dataset
    run_idx = list(rds.run_ids).index(spec["run_id"])
    sample = rds[run_idx]
    x = sample["input_fields"].unsqueeze(0).to(device)    # (1, Ti, C, H, W)
    y = sample["output_fields"].unsqueeze(0).to(device)   # (1, To, C, H, W)
    Ti, To = x.shape[1], y.shape[1]
    steps = min(args.steps, To) if args.steps else To
    log.info("run_%04d: warm-up %d frames, rolling %d steps",
             spec["run_id"], Ti, steps)

    hist = x.reshape(1, Ti * n_field_ch, *x.shape[3:])
    preds = []
    for k in trange(steps, desc="rollout"):
        y_pred = model(hist)                          # (1, C, H, W) normalized
        preds.append(dm.denormalize_fn(y_pred)[:, field_idx].cpu())
        if k < steps - 1:
            hist = torch.cat([hist[:, n_field_ch:], y_pred], dim=1)

    pred = torch.cat(preds, 0).numpy()                              # (steps, H, W)
    true = dm.denormalize_fn(y[0, :steps])[:, field_idx].cpu().numpy()

    run_zarr = spec["root"] / "test" / f"run_{spec['run_id']:04d}.zarr"
    ds = xr.open_zarr(run_zarr)
    lat = ds.lat.to_numpy()
    lon = ds.lon.to_numpy()
    return pred, true, lat, lon


def render(args, spec, pred, true, lat, lon):
    log = logging.getLogger(__name__)
    cmap = plt.get_cmap("RdBu_r")
    if spec["symmetric"]:
        m = float(np.nanpercentile(np.abs(true), 99))
        vmin, vmax = -m, m
    else:
        vmin = float(np.nanpercentile(true, 1))
        vmax = float(np.nanpercentile(true, 99))

    n = pred.shape[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if not args.ckpt else Path(args.ckpt).parent.name
    mp4 = out_dir / f"predict_globe_{args.dataset}_{tag}_run{spec['run_id']:04d}.mp4"
    log.info("Rendering %d frames (pred|truth) @ %d dpi -> %s", n, args.dpi, mp4)

    cyc_pred, clon = add_cyclic_point(pred, coord=lon, axis=2)
    cyc_true, _ = add_cyclic_point(true, coord=lon, axis=2)

    frames = []
    for i in trange(n, desc="render"):
        cent_lon = (i / max(n, 1)) * 360.0 - 180.0
        proj = ccrs.Orthographic(central_longitude=cent_lon, central_latitude=30.0)
        fig = plt.figure(figsize=(15, 8), constrained_layout=True, dpi=args.dpi)
        for j, (data, label) in enumerate(((cyc_pred, "prediction"),
                                           (cyc_true, "ground truth"))):
            ax = fig.add_subplot(1, 2, j + 1, projection=proj)
            ax.set_global()
            ax.gridlines(alpha=0.3, linestyle="--")
            ax.pcolormesh(clon, lat, data[i], cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="auto", transform=ccrs.PlateCarree(),
                          rasterized=True)
            ax.set_title(f"{spec['field']} — {label}", fontsize=13)
        fig.suptitle(f"Flower2D rollout — {args.dataset} run_{spec['run_id']:04d}"
                     f"  |  step {i + 1}/{n}", fontsize=15)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = frame[: h - (h % 16), : w - (w % 16), :3]
        frames.append(frame)
        plt.close(fig)

    iio.imwrite(str(mp4), np.stack(frames, 0), fps=args.fps,
                codec="libx264", pixelformat="yuv420p")
    log.info("Wrote %s (%d frames @ %d fps)", mp4, len(frames), args.fps)
    return mp4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--run-id", type=int, default=None,
                    help="Override the notebook-chosen run id.")
    ap.add_argument("--field", default=None, help="Override the visualized field.")
    ap.add_argument("--steps", type=int, default=0, help="0 = full trajectory.")
    ap.add_argument("--out-dir", default="notebooks/figures/clima_predict_renders")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lifting-dim", type=int, default=None)
    args = ap.parse_args()

    spec = dict(DATASETS[args.dataset])
    if args.run_id is not None:
        spec["run_id"] = args.run_id
    if args.field is not None:
        spec["field"] = args.field

    _setup_logging()
    pred, true, lat, lon = rollout(args, spec)
    render(args, spec, pred, true, lat, lon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
