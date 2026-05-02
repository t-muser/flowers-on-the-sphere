"""Dandelion: U-Net with spherical Flower blocks and DISCO conv up/down.

Reuses the ``FlowerUNet`` scaffold and ``FlowerBlock`` from ``zinnia.py``;
swaps the ``LatDeform`` / ``Spectral`` 2x down/up for ``DiscreteContinuousConvS2``
and ``DiscreteContinuousConvTransposeS2`` from ``torch_harmonics``. Inside
each ``FlowerBlock`` the warp defaults to ``SphericalSelfWarp`` (with optional
``TangentSelfWarp`` via ``tangent=True``).
"""

import functools
import math

import torch
import torch.nn as nn

from torch_harmonics import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2

from fots.models.flowers.zinnia import (
    FlowerBlock,
    FlowerUNet,
    make_norm2d,
)


# Heuristic from torch_harmonics.examples.models.lsno._compute_cutoff_radius.
_DISCO_CUTOFF_FACTOR = {
    "piecewise linear": 0.5,
    "morlet": 0.5,
    "zernike": math.sqrt(2.0),
}


def _disco_cutoff(nlat: int, kernel_shape, basis_type: str) -> float:
    kh = kernel_shape[0] if isinstance(kernel_shape, (tuple, list)) else kernel_shape
    return (kh + 1) * _DISCO_CUTOFF_FACTOR[basis_type] * math.pi / float(nlat - 1)


# =============================================================================
# DiscoDownBlock — FlowerBlock + DISCO 2x down (changes channels and resolution)
# =============================================================================

class DiscoDownBlock(nn.Module):
    """Encoder stage: FlowerBlock at high res, then DISCO conv to low res."""

    def __init__(self, in_channels, out_channels, spatial_resolution,
                 num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range,
                 tangent=False, polar_fold=False,
                 low_res=None, grid_type: str = "equiangular",
                 norm_type: str = "group",
                 disco_kernel_shape=(3, 3),
                 disco_basis_type: str = "piecewise linear",
                 disco_groups: int = 1,
                 **kwargs):
        super().__init__()
        assert low_res is not None, "DiscoDownBlock requires low_res"
        self.flower = FlowerBlock(
            in_channels, spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate,
            tangent=tangent, polar_fold=polar_fold,
            norm_type=norm_type,
        )
        in_shape = tuple(spatial_resolution)
        out_shape = tuple(low_res)
        self.disco_down = DiscreteContinuousConvS2(
            in_channels, out_channels,
            in_shape=in_shape, out_shape=out_shape,
            kernel_shape=disco_kernel_shape,
            basis_type=disco_basis_type,
            grid_in=grid_type, grid_out=grid_type,
            groups=disco_groups,
            bias=True,
            theta_cutoff=_disco_cutoff(in_shape[0], disco_kernel_shape, disco_basis_type),
        )
        self.norm = make_norm2d(norm_type, out_channels, groups=groups)
        self.act = nn.GELU()

    @property
    def lat_deform(self):
        # Interface parity with LatDeformDownBlock so the decoder can read
        # encoder_blocks[idx].lat_deform — DiscoUpBlock ignores this.
        return self.disco_down

    def forward(self, x, meta=None):
        x = self.flower(x, meta) + x
        skip = x
        x_down = self.act(self.norm(self.disco_down(x)))
        return skip, x_down


# =============================================================================
# DiscoUpBlock — DISCO transpose 2x up, cat skip, FlowerBlock at high res
# =============================================================================

class DiscoUpBlock(nn.Module):
    """Decoder stage: DISCO transpose to high res, cat skip, FlowerBlock."""

    def __init__(self, in_channels, splat_out_channels, total_channels,
                 down_block,  # ignored; kept for signature parity
                 out_spatial_resolution, num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range,
                 tangent=False, polar_fold=False,
                 low_res=None, grid_type: str = "equiangular",
                 norm_type: str = "group",
                 disco_kernel_shape=(3, 3),
                 disco_basis_type: str = "piecewise linear",
                 disco_groups: int = 1,
                 **kwargs):
        super().__init__()
        assert low_res is not None, "DiscoUpBlock requires low_res"
        in_shape = tuple(low_res)
        out_shape = tuple(out_spatial_resolution)
        self.disco_up = DiscreteContinuousConvTransposeS2(
            in_channels, splat_out_channels,
            in_shape=in_shape, out_shape=out_shape,
            kernel_shape=disco_kernel_shape,
            basis_type=disco_basis_type,
            grid_in=grid_type, grid_out=grid_type,
            groups=disco_groups,
            bias=True,
            theta_cutoff=_disco_cutoff(out_shape[0], disco_kernel_shape, disco_basis_type),
        )
        self.norm = make_norm2d(norm_type, splat_out_channels, groups=groups)
        self.act = nn.GELU()
        self.flower = FlowerBlock(
            total_channels, out_spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate,
            tangent=tangent, polar_fold=polar_fold,
            norm_type=norm_type,
        )

    def forward(self, x, skip, meta=None):
        x = self.act(self.norm(self.disco_up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.flower(x, meta) + x
        return x


# =============================================================================
# Dandelion — Flower architecture with spherical warps + DISCO down/up
# =============================================================================

class Dandelion(nn.Module):
    """U-Net with spherical FlowerBlocks and DISCO conv up/down on S².

    Defaults to ``SphericalSelfWarp`` with ``polar_fold=True``; pass
    ``tangent=True`` for ``TangentSelfWarp`` (Rodrigues exponential map). If
    the grid height is not divisible by 2^(n_levels-1), the south-pole row
    is dropped on entry and reconstructed (zonal mean of row H-1) on egress,
    matching Dahlia.
    """

    def __init__(self, inp_shape=(121, 240), out_shape=None,
                 inp_chans=69, out_chans=69,
                 lifting_dim=150, n_levels=4,
                 channel_multipliers=None,
                 num_heads=50, groups=50, dropout_rate=0.0,
                 tangent=False, polar_fold=True,
                 coord_dim=3,
                 disco_kernel_shape=(3, 3),
                 disco_basis_type: str = "piecewise linear",
                 disco_groups: int = 1,
                 grid_type: str = "equiangular",
                 **kwargs):
        super().__init__()

        H, W = inp_shape
        divisor = 2 ** (n_levels - 1)

        self.orig_H = H
        self.drop_south_pole = False

        if H % divisor != 0:
            assert (H - 1) % divisor == 0, (
                f"H={H}: dropping one row gives H-1={H-1} which is still not "
                f"divisible by {divisor}."
            )
            self.drop_south_pole = True
            H = H - 1

        assert W % divisor == 0, f"Width {W} must be divisible by {divisor}"

        disco_kwargs = dict(
            disco_kernel_shape=tuple(disco_kernel_shape),
            disco_basis_type=disco_basis_type,
            disco_groups=disco_groups,
        )
        down_cls = functools.partial(DiscoDownBlock, **disco_kwargs)
        up_cls = functools.partial(DiscoUpBlock, **disco_kwargs)

        self.model = FlowerUNet(
            dim_in=inp_chans,
            dim_out=out_chans,
            spatial_resolution=(H, W),
            lifting_dim=lifting_dim,
            n_levels=n_levels,
            channel_multipliers=channel_multipliers,
            num_heads=num_heads,
            groups=groups,
            dropout_rate=dropout_rate,
            coord_dim=coord_dim,
            tangent=tangent,
            polar_fold=polar_fold,
            up_block_cls=up_cls,
            down_block_cls=down_cls,
            drop_south_pole=self.drop_south_pole,
            grid_type=grid_type,
        )

    def forward(self, x, meta=None):
        if self.drop_south_pole:
            x = x[:, :, :-1, :]

        x = self.model(x, meta)

        if self.drop_south_pole:
            south_pole = x[:, :, -1:, :].mean(dim=3, keepdim=True)
            south_pole = south_pole.expand(-1, -1, -1, x.shape[3])
            x = torch.cat([x, south_pole], dim=2)

        return x
