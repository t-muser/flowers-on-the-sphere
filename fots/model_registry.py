"""Model registry for fots — trimmed from torch-harmonics examples.

Keeps the equiangular-grid baselines (``UNet``, ``Transformer``,
``Segformer``) and adds rows targeting the real Zinnia/Dahlia from
``flowers``. The old torch-harmonics V15 Zinnia is dropped.

Spherical baselines from ``torch_harmonics.examples.models`` are
available when the full torch-harmonics examples wheel is installed;
imports are lazy so a missing ``natten`` / missing examples dir only
fails at instantiation.
"""
from __future__ import annotations

from functools import partial

from fots.models.flowers import Dahlia, Zinnia


def _load_baselines():
    try:
        from fots.models.baselines import Segformer, Transformer, UNet
        return Transformer, UNet, Segformer
    except ImportError as e:  # pragma: no cover - runtime optional
        raise ImportError(f"baselines import failed: {e}")


def _load_th_spherical():
    try:
        from torch_harmonics.examples.models import (
            SphericalFourierNeuralOperator,
            SphericalSegformer,
            SphericalTransformer,
            SphericalUNet,
        )
        return SphericalFourierNeuralOperator, SphericalSegformer, SphericalTransformer, SphericalUNet
    except ImportError:  # pragma: no cover
        return None, None, None, None


def get_baseline_models(
    img_size=(128, 256),
    in_chans=3,
    out_chans=3,
    residual_prediction: bool = False,
    drop_path_rate: float = 0.0,
    grid: str = "equiangular",
):
    Transformer, UNet, Segformer = _load_baselines()
    SFNO, S2Segformer, S2Transformer, S2UNet = _load_th_spherical()

    registry = {
        "unet_e128": partial(
            UNet,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            depths=[2, 2, 2, 2],
            scale_factor=2,
            activation_function="gelu",
            drop_path_rate=drop_path_rate,
        ),
        "transformer_e128": partial(
            Transformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(3, 3),
            drop_path_rate=drop_path_rate,
            attention_mode="global",
            upsampling_method="conv",
            bias=False,
        ),
        "segformer_e128": partial(
            Segformer,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(4, 4),
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False,
        ),
        "zinnia": partial(
            Zinnia,
            inp_shape=img_size,
            inp_chans=in_chans,
            out_chans=out_chans,
        ),
        "dahlia": partial(
            Dahlia,
            inp_shape=img_size,
            inp_chans=in_chans,
            out_chans=out_chans,
        ),
    }

    if SFNO is not None:
        registry["sfno_e128"] = partial(
            SFNO,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            use_mlp=True,
            normalization_layer="instance_norm",
        )
    if S2UNet is not None:
        registry["s2unet_e128"] = partial(
            S2UNet,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            depths=[2, 2, 2, 2],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=0.1,
            transform_skip=False,
            upsampling_mode="conv",
            downsampling_mode="conv",
        )
    if S2Transformer is not None:
        registry["s2transformer_e128"] = partial(
            S2Transformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="global",
            bias=False,
        )
    if S2Segformer is not None:
        registry["s2segformer_e128"] = partial(
            S2Segformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False,
        )

    return registry
