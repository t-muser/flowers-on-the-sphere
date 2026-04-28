"""Zinnia and Dahlia: U-Net weather models with spherical warps.

Zinnia
    Legendre–Gauss internal grid with SHT boundary resampling; tangent warp;
    SHT-based 2x down/upsampling; no learnable gamma residual scaling.

Dahlia
    Equiangular internal grid (optional south-pole-row drop); spherical warp
    with polar fold (or tangent via config); fixed sec(lat)-scaled 3×3
    downsampler + splat upsampler; no learnable gamma residual scaling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch_harmonics as th

from fots.models.flowers.warps import (
    custom_grid_sample2d,
    SphericalSelfWarp,
    TangentSelfWarp,
    flow_fold_heads,
    value_fold_heads,
    value_unfold_heads,
)


# =============================================================================
# Normalization
# =============================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm across channels for (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class RMSNorm2d(nn.Module):
    """RMSNorm across channels for (B, C, H, W) — no mean subtraction."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms.to(x.dtype)) * self.weight.view(1, -1, 1, 1)


def make_norm2d(norm_type: str, num_channels: int, groups: int = 32) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "group":
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=True)
    if norm_type == "layer":
        return LayerNorm2d(num_channels)
    if norm_type == "rms":
        return RMSNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=False)
    raise ValueError(
        f"Unknown norm_type={norm_type!r}; "
        f"expected one of: group, layer, rms, instance, none"
    )


# =============================================================================
# Lat/lon grid helpers
# =============================================================================

def _grid_lats(H, grid_type: str = "equiangular", drop_south_pole: bool = False):
    """Latitude nodes (radians, descending from +π/2 toward -π/2)."""
    if grid_type == "legendre-gauss":
        from torch_harmonics.quadrature import precompute_latitudes
        colat, _ = precompute_latitudes(H, grid="legendre-gauss")  # [0, π] ascending
        colat = torch.as_tensor(colat, dtype=torch.float32)
        return math.pi / 2 - colat
    if drop_south_pole:
        true_H = H + 1
        delta = math.pi / (true_H - 1)
        return (math.pi / 2) - torch.arange(H, dtype=torch.float32) * delta
    return torch.linspace(math.pi / 2, -math.pi / 2, H)


def make_lat_lon_grid(H, W, drop_south_pole: bool = False,
                      grid_type: str = "equiangular"):
    """(1, H, W, 2) with [..., 0]=lon, [..., 1]=lat (radians)."""
    lat = _grid_lats(H, grid_type=grid_type, drop_south_pole=drop_south_pole)
    lon = torch.linspace(0, 2 * math.pi * (1 - 1 / W), W)
    lat_g = lat[:, None].expand(H, W)
    lon_g = lon[None, :].expand(H, W)
    return torch.stack([lon_g, lat_g], dim=-1).unsqueeze(0).contiguous()


def sphere_coord_grid(H, W, drop_south_pole: bool = False,
                      grid_type: str = "equiangular"):
    """3D Cartesian unit-sphere embedding: (1, 3, H, W)."""
    lat = _grid_lats(H, grid_type=grid_type, drop_south_pole=drop_south_pole)
    lon = torch.linspace(0, 2 * math.pi * (1 - 1 / W), W)
    cos_lat, sin_lat = torch.cos(lat), torch.sin(lat)
    cos_lon, sin_lon = torch.cos(lon), torch.sin(lon)
    x = cos_lat[:, None] * cos_lon[None, :]
    y = cos_lat[:, None] * sin_lon[None, :]
    z = sin_lat[:, None].expand(-1, W)
    return torch.stack([x, y, z], dim=0).unsqueeze(0).contiguous()


# =============================================================================
# FlowerBlock — pre-norm residual block with a spherical warp
# =============================================================================

class FlowerBlock(nn.Module):
    """Pre-norm SelfWarp block. Returns f(norm(x)); caller adds + x."""

    def __init__(self, channels, spatial_resolution, num_heads,
                 lat_lon_grid, lat_top, lat_range,
                 groups=32, dropout_rate=0.0,
                 tangent=False, polar_fold=False,
                 norm_type: str = "group"):
        super().__init__()
        self.norm = make_norm2d(norm_type, channels, groups=groups)
        if tangent:
            self.warp = TangentSelfWarp(
                channels, channels, spatial_resolution, num_heads,
                lat_lon_grid, lat_top, lat_range,
            )
        else:
            self.warp = SphericalSelfWarp(
                channels, channels, spatial_resolution, num_heads,
                lat_lon_grid, lat_top, lat_range, polar_fold=polar_fold,
            )
        self.w = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, meta=None):
        out = self.norm(x)
        out = self.w(self.warp(out))
        out = self.act(out)
        out = self.drop(out)
        return out


# =============================================================================
# LatDeform 2x down — fixed sec(lat)-scaled 3×3 spherical sampling
# =============================================================================

class LatDeform2xDown(nn.Module):
    """Fixed spherical 3×3 sampling 2x-downsampling (Dahlia path).

    Stencil is a 3×3 neighborhood around each patch center (patch_size=2).
    Longitudinal spacing is scaled by sec(lat) (clamped) so the physical
    footprint of each patch is roughly constant from equator to poles.

    Offsets are a ``register_buffer`` — NOT trainable.
    """

    def __init__(self, dim_in, dim_out, spatial_resolution, groups=32,
                 drop_south_pole: bool = False, sec_max: float | None = None,
                 norm_type: str = "group"):
        super().__init__()
        H, W = spatial_resolution
        patch_size = 2
        assert H % patch_size == 0, f"H={H} must be divisible by {patch_size}"
        assert W % patch_size == 0, f"W={W} must be divisible by {patch_size}"

        self.patch_size = patch_size
        self.H_p = H // patch_size
        self.W_p = W // patch_size
        self.H = H
        self.W = W
        self.n_points = 9
        self.drop_south_pole = drop_south_pole
        if sec_max is None:
            sec_max = W / 4.0
        self.sec_max = float(sec_max)

        pr = torch.arange(self.H_p, dtype=torch.float32) * patch_size + (patch_size - 1) / 2
        pc = torch.arange(self.W_p, dtype=torch.float32) * patch_size + (patch_size - 1) / 2
        cy = pr / (H - 1) * 2 - 1
        cx = pc / (W - 1) * 2 - 1
        centers = torch.stack(
            [cx[None, :].expand(self.H_p, -1),
             cy[:, None].expand(-1, self.W_p)], dim=-1
        )
        self.register_buffer("centers", centers)

        if drop_south_pole:
            true_H = H + 1
            delta = math.pi / (true_H - 1)
        else:
            delta = math.pi / (H - 1)
        lat_per_row = (math.pi / 2) - pr * delta

        sec_lat = 1.0 / torch.cos(lat_per_row)
        sec_lat = sec_lat.clamp(min=-self.sec_max, max=self.sec_max)

        ky_pix, kx_pix = torch.meshgrid(
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.tensor([-1.0, 0.0, 1.0]),
            indexing='ij',
        )
        kx_flat = kx_pix.flatten()
        ky_flat = ky_pix.flatten()

        dx_pix = kx_flat[None, :] * sec_lat[:, None]
        dy_pix = ky_flat[None, :].expand(self.H_p, -1)

        dx = dx_pix / (W - 1) * 2
        dy = dy_pix / (H - 1) * 2
        offsets = torch.stack([dx, dy], dim=-1)
        self.register_buffer("offsets", offsets)

        self.proj = nn.Sequential(
            nn.Conv2d(self.n_points * dim_in, dim_out, kernel_size=1),
            make_norm2d(norm_type, dim_out, groups=groups),
            nn.GELU(),
        )

    def get_sampling_grid(self, batch_size, device):
        sample_locs = (
            self.centers.unsqueeze(2) +
            self.offsets.unsqueeze(1)
        )
        grid = sample_locs.reshape(1, self.H_p, self.W_p * self.n_points, 2)
        return grid.expand(batch_size, -1, -1, -1)

    @torch.compile
    def forward(self, x):
        B, C_in, H, W = x.shape
        grid = self.get_sampling_grid(B, x.device)

        lat_pad = "polar_dropped_south" if self.drop_south_pole else "polar"
        sampled = custom_grid_sample2d(
            x, grid, mode="bilinear",
            padding_modes=(lat_pad, "periodic"),
            align_corners=True,
        )
        sampled = sampled.reshape(B, C_in, self.H_p, self.W_p, self.n_points)
        sampled = sampled.permute(0, 1, 4, 2, 3).reshape(
            B, C_in * self.n_points, self.H_p, self.W_p
        )
        return self.proj(sampled)


# =============================================================================
# LatDeformSplat 2x up — node-centred polar wrap + Wx+b split
# =============================================================================

class LatDeformSplat2xUp(nn.Module):
    """2x-upsampler that splats low-res activations onto the high-res grid
    using the paired down-block's sampling locations.

    - Node-centred polar wrap (dropped-pole semantics): North Pole at row 0,
      phantom South Pole at index H; the phantom index folds onto row H-1.
    - Wx+b split: Wx (no bias) is computed at low resolution before splatting;
      +b is added after splatting, so zero-splat pixels get a baseline b
      rather than a dead zero.
    """

    def __init__(self, down_block: LatDeform2xDown, dim_in, dim_out, groups=32,
                 norm_type: str = "group"):
        super().__init__()
        self.down_block = down_block
        self.dim_out = dim_out
        self.n_points = down_block.n_points

        self.norm = make_norm2d(norm_type, dim_in, groups=groups)
        self.conv_w = nn.Conv2d(dim_in, down_block.n_points * dim_out,
                                kernel_size=1, bias=False)
        self.splat_bias = nn.Parameter(torch.zeros(1, dim_out, 1, 1))

        self.post = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1),
            make_norm2d(norm_type, dim_out, groups=groups),
            nn.GELU(),
        )

    @torch.compile
    def forward(self, x, H_out=None, W_out=None):
        B, _, H_p, W_p = x.shape
        H = H_out if H_out is not None else self.down_block.H
        W = W_out if W_out is not None else self.down_block.W
        N = self.n_points
        C = self.dim_out

        vals = self.conv_w(self.norm(x))
        vals = vals.reshape(B, C, N, H_p, W_p).permute(0, 1, 3, 4, 2)
        vals = vals.reshape(B, C, H_p * W_p * N)

        grid = self.down_block.get_sampling_grid(B, x.device).detach()
        grid = grid.reshape(B, H_p * W_p * N, 2)
        px = (grid[..., 0] + 1) * 0.5 * (W - 1)
        py = (grid[..., 1] + 1) * 0.5 * (H - 1)

        x0 = px.floor().long()
        y0 = py.floor().long()
        x1 = x0 + 1
        y1 = y0 + 1

        fx = (px - x0.float()).unsqueeze(1)
        fy = (py - y0.float()).unsqueeze(1)

        w00 = (1 - fx) * (1 - fy)
        w01 = (1 - fx) * fy
        w10 = fx * (1 - fy)
        w11 = fx * fy

        x0w = x0 % W
        x1w = x1 % W

        half_W = W // 2

        def _polar_wrap(yi, xi):
            past_n = yi < 0
            past_s = yi >= H
            yi = torch.where(past_n, -yi, yi)
            yi = torch.where(past_s, 2 * H - yi, yi)
            xi = torch.where(past_n | past_s, (xi + half_W) % W, xi)
            return yi.clamp(0, H - 1), xi

        num = x.new_zeros(B, C, H * W)
        den = x.new_zeros(B, 1, H * W)

        for wt, yi, xi in [
            (w00, y0, x0w),
            (w01, y1, x0w),
            (w10, y0, x1w),
            (w11, y1, x1w),
        ]:
            yi, xi = _polar_wrap(yi, xi)
            idx = (yi * W + xi).unsqueeze(1).expand_as(vals)
            idx1 = (yi * W + xi).unsqueeze(1)
            num.scatter_add_(2, idx, vals * wt)
            den.scatter_add_(2, idx1, wt)

        out = num / den.clamp(min=1e-8)
        out = out.reshape(B, C, H, W)
        out = out + self.splat_bias

        return self.post(out)


# =============================================================================
# Spectral 2x down / up — spherical-harmonic resolution change
# =============================================================================

class Spectral2xDown(nn.Module):
    """SHT-based 2x downsampler on S². Exact up to quadrature, pole-aware."""

    def __init__(self, dim_in, dim_out, high_res, low_res, groups=32,
                 grid_type: str = "equiangular", norm_type: str = "group"):
        super().__init__()
        H_hi, W_hi = high_res
        H_lo, W_lo = low_res
        self.H_hi, self.W_hi = H_hi, W_hi
        self.H_lo, self.W_lo = H_lo, W_lo

        lmax = H_lo
        mmax = W_lo // 2 + 1
        self.lmax = lmax
        self.mmax = mmax

        self.sht_hi = th.RealSHT(H_hi, W_hi, lmax=lmax, mmax=mmax, grid=grid_type).float()
        self.isht_lo = th.InverseRealSHT(H_lo, W_lo, lmax=lmax, mmax=mmax, grid=grid_type).float()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            make_norm2d(norm_type, dim_out, groups=groups),
            nn.GELU(),
        )

    def forward(self, x):
        dtype = x.dtype
        coef = self.sht_hi(x.float())
        y = self.isht_lo(coef).to(dtype)
        return self.proj(y)


class Spectral2xUp(nn.Module):
    """SHT-based 2x upsampler on S²."""

    def __init__(self, dim_in, dim_out, low_res, high_res, groups=32,
                 grid_type: str = "equiangular", norm_type: str = "group"):
        super().__init__()
        H_lo, W_lo = low_res
        H_hi, W_hi = high_res
        self.H_lo, self.W_lo = H_lo, W_lo
        self.H_hi, self.W_hi = H_hi, W_hi

        lmax = H_lo
        mmax = W_lo // 2 + 1
        self.lmax = lmax
        self.mmax = mmax

        self.pre = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            make_norm2d(norm_type, dim_out, groups=groups),
            nn.GELU(),
        )
        self.sht_lo = th.RealSHT(H_lo, W_lo, lmax=lmax, mmax=mmax, grid=grid_type).float()
        self.isht_hi = th.InverseRealSHT(H_hi, W_hi, lmax=lmax, mmax=mmax, grid=grid_type).float()

        self.post = nn.Conv2d(dim_out, dim_out, kernel_size=1)

    def forward(self, x, H_out=None, W_out=None):
        x = self.pre(x)
        dtype = x.dtype
        coef = self.sht_lo(x.float())
        y = self.isht_hi(coef).to(dtype)
        return self.post(y)


# =============================================================================
# Encoder / Decoder blocks
# =============================================================================

class LatDeformDownBlock(nn.Module):
    """Encoder stage: FlowerBlock at current resolution, then LatDeform 2x down."""

    def __init__(self, in_channels, out_channels, spatial_resolution,
                 num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range,
                 tangent=False, polar_fold=False,
                 drop_south_pole: bool = False, sec_max: float | None = None,
                 norm_type: str = "group",
                 **kwargs):
        super().__init__()
        self.flower = FlowerBlock(
            in_channels, spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate,
            tangent=tangent, polar_fold=polar_fold,
            norm_type=norm_type,
        )
        self.lat_deform = LatDeform2xDown(
            in_channels, out_channels, spatial_resolution,
            groups=groups,
            drop_south_pole=drop_south_pole, sec_max=sec_max,
            norm_type=norm_type,
        )

    def forward(self, x, meta=None):
        x = self.flower(x, meta) + x
        skip = x
        x_down = self.lat_deform(x)
        return skip, x_down


class LatDeformUpBlock(nn.Module):
    """Decoder stage: splat 2x up, cat skip, FlowerBlock at upsampled resolution."""

    def __init__(self, in_channels, splat_out_channels, total_channels,
                 down_block: LatDeform2xDown,
                 out_spatial_resolution, num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range,
                 tangent=False, polar_fold=False,
                 norm_type: str = "group",
                 **kwargs):
        super().__init__()
        self.splat = LatDeformSplat2xUp(
            down_block, in_channels, splat_out_channels, groups=groups,
            norm_type=norm_type,
        )
        self.flower = FlowerBlock(
            total_channels, out_spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate,
            tangent=tangent, polar_fold=polar_fold,
            norm_type=norm_type,
        )

    def forward(self, x, skip, meta=None):
        x = self.splat(x)
        x = torch.cat([x, skip], dim=1)
        x = self.flower(x, meta) + x
        return x


class SpectralDownBlock(nn.Module):
    """Encoder stage: FlowerBlock, then Spectral 2x down."""

    def __init__(self, in_channels, out_channels, spatial_resolution,
                 num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range,
                 tangent=False, polar_fold=False,
                 low_res=None, grid_type: str = "equiangular",
                 norm_type: str = "group",
                 **kwargs):
        super().__init__()
        assert low_res is not None, "SpectralDownBlock requires low_res"
        self.flower = FlowerBlock(
            in_channels, spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate,
            tangent=tangent, polar_fold=polar_fold,
            norm_type=norm_type,
        )
        self.spectral_down = Spectral2xDown(
            in_channels, out_channels,
            high_res=tuple(spatial_resolution), low_res=tuple(low_res),
            groups=groups, grid_type=grid_type,
            norm_type=norm_type,
        )

    @property
    def lat_deform(self):
        # Interface parity with LatDeformDownBlock so the decoder can read
        # encoder_blocks[idx].lat_deform — SpectralUpBlock ignores this.
        return self.spectral_down

    def forward(self, x, meta=None):
        x = self.flower(x, meta) + x
        skip = x
        x_down = self.spectral_down(x)
        return skip, x_down


class SpectralUpBlock(nn.Module):
    """Decoder stage: Spectral 2x up, cat skip, FlowerBlock."""

    def __init__(self, in_channels, splat_out_channels, total_channels,
                 down_block,  # ignored; kept for signature parity
                 out_spatial_resolution, num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range,
                 tangent=False, polar_fold=False,
                 low_res=None, grid_type: str = "equiangular",
                 norm_type: str = "group",
                 **kwargs):
        super().__init__()
        assert low_res is not None, "SpectralUpBlock requires low_res"
        self.spectral_up = Spectral2xUp(
            in_channels, splat_out_channels,
            low_res=tuple(low_res), high_res=tuple(out_spatial_resolution),
            groups=groups, grid_type=grid_type,
            norm_type=norm_type,
        )
        self.flower = FlowerBlock(
            total_channels, out_spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate,
            tangent=tangent, polar_fold=polar_fold,
            norm_type=norm_type,
        )

    def forward(self, x, skip, meta=None):
        x = self.spectral_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.flower(x, meta) + x
        return x


# =============================================================================
# FlowerUNet — the core U-Net scaffold used by Zinnia and Dahlia
# =============================================================================

class FlowerUNet(nn.Module):
    """U-Net with configurable resolution-change blocks and spherical warps."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        spatial_resolution: tuple[int, int],
        lifting_dim: int = 128,
        n_levels: int = 4,
        channel_multipliers: list[int] | None = None,
        num_heads: int = 32,
        groups: int = 32,
        dropout_rate: float = 0.0,
        coord_dim: int = 3,
        tangent: bool = False,
        polar_fold: bool = False,
        up_block_cls=None,
        down_block_cls=None,
        drop_south_pole: bool = False,
        bottleneck_blocks: int = 2,
        blocks_per_stage: int = 1,
        sec_max: float | None = None,
        resolution_schedule: list[tuple[int, int]] | None = None,
        grid_type: str = "equiangular",
        norm_type: str = "group",
    ):
        super().__init__()
        self.norm_type = norm_type
        if up_block_cls is None:
            up_block_cls = LatDeformUpBlock
        if down_block_cls is None:
            down_block_cls = LatDeformDownBlock

        if channel_multipliers is None:
            channel_multipliers = [2**i for i in range(n_levels)]
        assert len(channel_multipliers) == n_levels

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.spatial_resolution = list(spatial_resolution)
        self.n_levels = n_levels
        self.coord_dim = coord_dim

        encoder_channels = [lifting_dim * m for m in channel_multipliers]

        if resolution_schedule is None:
            min_divisor = 2 ** (n_levels - 1)
            for i, d in enumerate(spatial_resolution):
                assert d % min_divisor == 0, (
                    f"Dimension {i} ({d}) must be divisible by {min_divisor}"
                )
            schedule = [tuple(spatial_resolution)]
            for _ in range(n_levels - 1):
                schedule.append(tuple(d // 2 for d in schedule[-1]))
        else:
            assert len(resolution_schedule) == n_levels, (
                f"resolution_schedule length {len(resolution_schedule)} != "
                f"n_levels {n_levels}"
            )
            assert tuple(resolution_schedule[0]) == tuple(spatial_resolution), (
                f"resolution_schedule[0]={tuple(resolution_schedule[0])} must "
                f"equal spatial_resolution={tuple(spatial_resolution)}"
            )
            schedule = [tuple(r) for r in resolution_schedule]
        self.resolution_schedule = schedule

        self.lift = nn.Conv2d(dim_in + coord_dim, encoder_channels[0], kernel_size=1)

        coord_grid = sphere_coord_grid(*spatial_resolution,
                                       drop_south_pole=drop_south_pole,
                                       grid_type=grid_type)
        self.register_buffer("coord_grid", coord_grid)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_extra_blocks = nn.ModuleList()
        current_res = list(schedule[0])
        for i in range(n_levels - 1):
            lat_lon = make_lat_lon_grid(*current_res,
                                        drop_south_pole=drop_south_pole,
                                        grid_type=grid_type)
            lat_top = float(lat_lon[0, 0, 0, 1])
            lat_bot = float(lat_lon[0, -1, 0, 1])
            lat_range = lat_top - lat_bot

            low_res = schedule[i + 1]
            self.encoder_blocks.append(down_block_cls(
                encoder_channels[i], encoder_channels[i + 1], current_res,
                num_heads, groups, dropout_rate,
                lat_lon, lat_top, lat_range,
                tangent=tangent, polar_fold=polar_fold,
                drop_south_pole=drop_south_pole, sec_max=sec_max,
                low_res=low_res, grid_type=grid_type,
                norm_type=norm_type,
            ))
            n_extra = (blocks_per_stage - 1) if i > 0 else 0
            self.encoder_extra_blocks.append(nn.ModuleList([
                FlowerBlock(encoder_channels[i], current_res, num_heads,
                            lat_lon, lat_top, lat_range,
                            groups=groups, dropout_rate=dropout_rate,
                            tangent=tangent, polar_fold=polar_fold,
                            norm_type=norm_type)
                for _ in range(n_extra)
            ]))
            current_res = list(schedule[i + 1])

        # Bottleneck
        bottleneck_ch = encoder_channels[-1]
        bn_lat_lon = make_lat_lon_grid(*current_res,
                                       drop_south_pole=drop_south_pole,
                                       grid_type=grid_type)
        bn_lat_top = float(bn_lat_lon[0, 0, 0, 1])
        bn_lat_bot = float(bn_lat_lon[0, -1, 0, 1])
        bn_lat_range = bn_lat_top - bn_lat_bot
        self.bottleneck = nn.ModuleList([
            FlowerBlock(bottleneck_ch, current_res, num_heads,
                        bn_lat_lon, bn_lat_top, bn_lat_range,
                        groups=groups, dropout_rate=dropout_rate,
                        tangent=tangent, polar_fold=polar_fold,
                        norm_type=norm_type)
            for _ in range(bottleneck_blocks)
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_extra_blocks = nn.ModuleList()
        decoder_in_ch = bottleneck_ch
        for i in range(n_levels - 1):
            encoder_idx = n_levels - 2 - i
            skip_ch = encoder_channels[encoder_idx]
            splat_out_ch = skip_ch
            total_ch = splat_out_ch + skip_ch

            out_res = list(schedule[encoder_idx])
            low_res = tuple(current_res)
            lat_lon = make_lat_lon_grid(*out_res,
                                        drop_south_pole=drop_south_pole,
                                        grid_type=grid_type)
            lat_top = float(lat_lon[0, 0, 0, 1])
            lat_bot = float(lat_lon[0, -1, 0, 1])
            lat_range = lat_top - lat_bot

            self.decoder_blocks.append(up_block_cls(
                decoder_in_ch, splat_out_ch, total_ch,
                self.encoder_blocks[encoder_idx].lat_deform,
                out_res, num_heads, groups, dropout_rate,
                lat_lon, lat_top, lat_range,
                tangent=tangent, polar_fold=polar_fold,
                low_res=low_res, grid_type=grid_type,
                norm_type=norm_type,
            ))
            n_extra = (blocks_per_stage - 1) if i < n_levels - 2 else 0
            self.decoder_extra_blocks.append(nn.ModuleList([
                FlowerBlock(total_ch, out_res, num_heads,
                            lat_lon, lat_top, lat_range,
                            groups=groups, dropout_rate=dropout_rate,
                            tangent=tangent, polar_fold=polar_fold,
                            norm_type=norm_type)
                for _ in range(n_extra)
            ]))
            decoder_in_ch = total_ch
            current_res = out_res

        self.project = nn.Sequential(
            nn.Conv2d(decoder_in_ch, lifting_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(lifting_dim, dim_out, kernel_size=1),
        )

    def forward(self, x, meta=None):
        B = x.shape[0]

        grid = self.coord_grid.expand(B, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)

        x = self.lift(x)

        skips = []
        for block, extra_blocks in zip(self.encoder_blocks, self.encoder_extra_blocks):
            skip, x = block(x, meta)
            for fb in extra_blocks:
                skip = fb(skip, meta) + skip
            skips.append(skip)

        for block in self.bottleneck:
            x = block(x, meta) + x

        for block, extra_blocks, skip in zip(self.decoder_blocks, self.decoder_extra_blocks, reversed(skips)):
            x = block(x, skip, meta)
            for fb in extra_blocks:
                x = fb(x, meta) + x

        return self.project(x)


# =============================================================================
# Zinnia — Legendre–Gauss internal grid, SHT boundary resampling, tangent warp
# =============================================================================

class Zinnia(nn.Module):
    """U-Net with SHT down/upsampling on a Legendre–Gauss grid.

    The data grid is equiangular at I/O; internally everything runs on a
    Legendre–Gauss grid, with a single SHT+iSHT pair for ingress/egress.
    FlowerBlocks use TangentSelfWarp (exact Rodrigues rotation on S²).
    """

    def __init__(self, inp_shape=(721, 1440), out_shape=None,
                 inp_chans=69, out_chans=69,
                 lifting_dim=384, n_levels=4,
                 channel_multipliers=None,
                 num_heads=128, groups=32, dropout_rate=0.0,
                 H_lg=None, W_lg=None, f_in=1.0,
                 coord_dim=0,
                 **kwargs):
        super().__init__()

        H_eq, W_eq = inp_shape
        if H_lg is None:
            H_lg = H_eq - 1 if H_eq % 2 == 1 else H_eq
        if W_lg is None:
            W_lg = W_eq
        self.orig_H = H_eq

        L = int(H_lg * f_in)
        M = int((W_lg // 2 + 1) * f_in)

        self.sht_in   = th.RealSHT(H_eq, W_eq, lmax=L, mmax=M,
                                   grid="equiangular").float()
        self.isht_in  = th.InverseRealSHT(H_lg, W_lg, lmax=L, mmax=M,
                                          grid="legendre-gauss").float()
        self.sht_out  = th.RealSHT(H_lg, W_lg, lmax=L, mmax=M,
                                   grid="legendre-gauss").float()
        self.isht_out = th.InverseRealSHT(H_eq, W_eq, lmax=L, mmax=M,
                                          grid="equiangular").float()

        min_div = 2 ** (n_levels - 1)
        assert H_lg % min_div == 0, (
            f"H_lg={H_lg} must be divisible by 2^(n_levels-1)={min_div}"
        )
        assert W_lg % min_div == 0, (
            f"W_lg={W_lg} must be divisible by 2^(n_levels-1)={min_div}"
        )
        schedule = [(H_lg, W_lg)]
        for _ in range(n_levels - 1):
            schedule.append((schedule[-1][0] // 2, schedule[-1][1] // 2))
        self.resolution_schedule = schedule

        self.model = FlowerUNet(
            dim_in=inp_chans,
            dim_out=out_chans,
            spatial_resolution=(H_lg, W_lg),
            lifting_dim=lifting_dim,
            n_levels=n_levels,
            channel_multipliers=channel_multipliers,
            num_heads=num_heads,
            groups=groups,
            dropout_rate=dropout_rate,
            coord_dim=coord_dim,
            tangent=True,
            polar_fold=False,
            up_block_cls=SpectralUpBlock,
            down_block_cls=SpectralDownBlock,
            resolution_schedule=schedule,
            grid_type="legendre-gauss",
        )

    def forward(self, x, meta=None):
        dtype = x.dtype
        x = self.isht_in(self.sht_in(x.float())).to(dtype)
        y = self.model(x, meta)
        y = self.isht_out(self.sht_out(y.float())).to(dtype)
        return y


# =============================================================================
# Dahlia — equiangular grid, sec(lat)-scaled 3×3 sampling, spherical warp
# =============================================================================

class Dahlia(nn.Module):
    """U-Net with LatDeform 3×3 sec(lat)-scaled down/up on an equiangular grid.

    Config kwarg ``tangent`` selects TangentSelfWarp (True) or
    SphericalSelfWarp (False, default). With ``tangent=False`` a
    ``polar_fold`` flag adds 180°-shift polar reflection for cross-pole
    displacements. If the grid height is not divisible by 2^(n_levels-1),
    the south-pole row is dropped on entry and reconstructed (by zonal
    mean of row H-1) on egress.
    """

    def __init__(self, inp_shape=(121, 240), out_shape=None,
                 inp_chans=69, out_chans=69,
                 lifting_dim=128, n_levels=4,
                 channel_multipliers=None,
                 num_heads=32, groups=32, dropout_rate=0.0,
                 tangent=False, polar_fold=True,
                 coord_dim=0, sec_max=None,
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
            up_block_cls=LatDeformUpBlock,
            down_block_cls=LatDeformDownBlock,
            drop_south_pole=self.drop_south_pole,
            sec_max=sec_max,
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
