"""Validation and diagnostic metrics for spherical PDE training.

All metrics operate on channels-first tensors ``(B, C, H, W)``. ``H``
indexes latitude; spherical averages weight by ``cos(lat)`` (normalized
to mean 1) so equal-area regions contribute equally regardless of the
grid oversampling near the poles.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch


def latitude_weights(
    nlat: int, grid: str = "equiangular", device: Optional[torch.device] = None
) -> torch.Tensor:
    """``cos(lat)`` weights on the grid's latitude nodes, normalized to mean 1.

    Returns a 1-D tensor of shape ``(nlat,)`` suitable for broadcasting
    against ``(B, C, H, W)`` along ``H``.
    """
    if grid == "legendre-gauss":
        from torch_harmonics.quadrature import precompute_latitudes
        colat, _ = precompute_latitudes(nlat, grid="legendre-gauss")
        colat = torch.as_tensor(colat, dtype=torch.float32)
        lat = math.pi / 2 - colat
    else:  # equiangular
        lat = torch.linspace(math.pi / 2, -math.pi / 2, nlat)
    w = torch.cos(lat).clamp(min=0.0)
    w = w / w.mean()
    if device is not None:
        w = w.to(device)
    return w


def _spatial_mean(
    x: torch.Tensor, lat_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Weighted mean over ``(H, W)`` → ``(B, C)``.

    With ``lat_weights`` this is the cos-latitude-weighted spherical
    average; without, a plain mean over spatial dims.
    """
    if lat_weights is None:
        return x.mean(dim=(-2, -1))
    w = lat_weights.view(1, 1, -1, 1)
    return (x * w).mean(dim=(-2, -1))


def param_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    with torch.no_grad():
        for p in parameters:
            total += p.detach().float().pow(2).sum().item()
    return total ** 0.5


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    with torch.no_grad():
        for p in parameters:
            if p.grad is None:
                continue
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def compute_loss_metrics(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
    field_names: Optional[list[str]] = None,
) -> Dict[str, float]:
    """Full metric suite on a single batch.

    Computes flat and sphere-weighted MSE / RMSE, per-field RMSE, and
    variable-relative L2 (``||ŷ − y||₂ / ||y||₂``, averaged over the
    batch). Returns a flat dict of scalar Python floats keyed with a
    ``<metric>`` or ``<metric>/<field>`` scheme compatible with wandb.
    """
    assert y_pred.shape == y_target.shape, (
        f"shape mismatch: pred {tuple(y_pred.shape)} vs target {tuple(y_target.shape)}"
    )
    B, C = y_pred.shape[:2]
    diff = y_pred - y_target
    sq = diff.pow(2)

    # Flat MSE (no spherical weighting).
    flat_mse = sq.mean().item()

    # Sphere-weighted per-field MSE → (C,)
    per_field_mse = _spatial_mean(sq, lat_weights).mean(dim=0)
    per_field_rmse = per_field_mse.sqrt()
    sphere_mse = per_field_mse.mean().item()
    sphere_rmse = math.sqrt(max(sphere_mse, 0.0))

    # Variable-relative L2 per field, averaged over batch.
    pred_flat = y_pred.reshape(B, C, -1)
    targ_flat = y_target.reshape(B, C, -1)
    num = (pred_flat - targ_flat).norm(dim=-1)
    den = targ_flat.norm(dim=-1).clamp(min=1e-12)
    rel_l2_per_sample = num / den  # (B, C)
    rel_l2_per_field = rel_l2_per_sample.mean(dim=0)
    rel_l2_mean = rel_l2_per_field.mean().item()

    metrics: Dict[str, float] = {
        "mse": flat_mse,
        "mse_sphere": sphere_mse,
        "rmse_sphere": sphere_rmse,
        "rel_l2": rel_l2_mean,
    }
    names = field_names or [f"field_{i}" for i in range(C)]
    for i, name in enumerate(names[:C]):
        metrics[f"rmse_sphere/{name}"] = per_field_rmse[i].item()
        metrics[f"rel_l2/{name}"] = rel_l2_per_field[i].item()
    return metrics
