"""Spherical warp primitives used by Zinnia and Dahlia.

- ``custom_grid_sample2d``: grid_sample with periodic and polar boundaries.
- ``SphericalSelfWarp``: flow in (lon, lat) radians.
- ``TangentSelfWarp``: flow in the local tangent plane, projected via Rodrigues.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


flow_fold_heads = {
    2: 'B (heads dir) H W -> (B heads) H W dir',
    3: 'B (heads dir) D H W -> (B heads) D H W dir',
}

value_fold_heads = {
    2: 'B (heads C_i) H W -> (B heads) C_i H W',
    3: 'B (heads C_i) D H W -> (B heads) C_i D H W',
}

value_unfold_heads = {
    2: '(B heads) C_i H W -> B (heads C_i) H W',
    3: '(B heads) C_i D H W -> B (heads C_i) D H W',
}


@torch.compile
def custom_grid_sample2d(
    input, grid, mode="bilinear", padding_modes=("zeros", "zeros"), align_corners=False
):
    """grid_sample with per-dimension periodic and polar boundaries.

    padding_modes is (padding_h, padding_w); each in
    {'zeros', 'border', 'reflection', 'periodic', 'polar', 'polar_dropped_south'}.
    'polar' reflects gy past ±1 and shifts gx by 180° (half-period), then
    clamps H with 'border'. 'polar_dropped_south' is the node-centred variant
    for grids where the south-pole row has been dropped.
    """
    padding_h, padding_w = padding_modes

    if padding_h.lower() == "polar":
        if padding_w.lower() != "periodic":
            raise ValueError("Polar boundary on H requires periodic W (longitude).")
        gx, gy = grid[..., 0], grid[..., 1]
        past_north = gy > 1
        past_south = gy < -1
        gy = torch.where(past_north,  2 - gy, gy)
        gy = torch.where(past_south, -2 - gy, gy)
        gx = torch.where(past_north | past_south, gx + 1.0, gx)
        grid = torch.stack([gx, gy], dim=-1)
        padding_h = "border"
    elif padding_h.lower() == "polar_dropped_south":
        if padding_w.lower() != "periodic":
            raise ValueError("polar_dropped_south on H requires periodic W (longitude).")
        gx, gy = grid[..., 0], grid[..., 1]
        south_pole_gy = 1.0 + 2.0 / (input.shape[2] - 1)
        past_north = gy < -1.0
        past_south = gy > south_pole_gy
        gy = torch.where(past_north, -2.0 - gy, gy)
        gy = torch.where(past_south, (2.0 * south_pole_gy) - gy, gy)
        gx = torch.where(past_north | past_south, gx + 1.0, gx)
        grid = torch.stack([gx, gy], dim=-1)
        padding_h = "border"

    if padding_w.lower() != "periodic" and padding_h.lower() != "periodic":
        if padding_w != padding_h:
            raise ValueError(
                "When using different paddings for x and y dimensions, "
                "one should be `periodic`."
            )
        return F.grid_sample(
            input, grid, mode=mode, padding_mode=padding_w, align_corners=align_corners
        )

    periodic_w = padding_w == "periodic"
    periodic_h = padding_h == "periodic"
    final_padding_w = "border" if periodic_w else padding_w
    final_padding_h = "border" if periodic_h else padding_h

    N, C, H, W = input.shape

    if periodic_h and periodic_w:
        top_row = input[:, :, 0:1, :]
        left_col = input[:, :, :, 0:1]
        top_left_corner = input[:, :, 0:1, 0:1]
        padded_h = torch.cat([input, top_row], dim=2)
        left_col_extended = torch.cat([left_col, top_left_corner], dim=2)
        padded_input = torch.cat([padded_h, left_col_extended], dim=3)
        H_padded = H + 1
        W_padded = W + 1
    elif periodic_h:
        top_row = input[:, :, 0:1, :]
        padded_input = torch.cat([input, top_row], dim=2)
        H_padded = H + 1
        W_padded = W
    elif periodic_w:
        left_col = input[:, :, :, 0:1]
        padded_input = torch.cat([input, left_col], dim=3)
        H_padded = H
        W_padded = W + 1

    gx, gy = grid[..., 0], grid[..., 1]

    if align_corners:
        px = (gx + 1) * (W - 1) / 2
        py = (gy + 1) * (H - 1) / 2
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        gx_new = 2 * px_wrapped / W - 1 if periodic_w else (2 * px_wrapped / (W_padded - 1) - 1 if W_padded > 1 else gx)
        gy_new = 2 * py_wrapped / H - 1 if periodic_h else (2 * py_wrapped / (H_padded - 1) - 1 if H_padded > 1 else gy)
    else:
        px = ((gx + 1) * W - 1) / 2
        py = ((gy + 1) * H - 1) / 2
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        gx_new = 2 * (px_wrapped + 0.5) / (W + 1) - 1 if periodic_w else (2 * (px_wrapped + 0.5) / W_padded - 1 if W_padded > 0 else gx)
        gy_new = 2 * (py_wrapped + 0.5) / (H + 1) - 1 if periodic_h else (2 * (py_wrapped + 0.5) / H_padded - 1 if H_padded > 0 else gy)

    grid_new = torch.stack([gx_new, gy_new], dim=-1)

    if periodic_w and periodic_h:
        final_padding = "border"
    elif periodic_w:
        final_padding = final_padding_h
    elif periodic_h:
        final_padding = final_padding_w

    return F.grid_sample(
        padded_input, grid_new, mode=mode,
        padding_mode=final_padding, align_corners=align_corners,
    )


class SphericalSelfWarp(nn.Module):
    """SelfWarp with flow in spherical (lon, lat) coordinates.

    Displacement heads predict (Δlon, Δlat) in radians. Conversion to
    grid_sample [-1, 1] coords (align_corners=True, equiangular padded grid):
        gx = lon / π - 1               (periodic)
        gy = 2 * (lat_top - lat) / lat_range - 1

    When ``polar_fold=True``, displacements crossing a pole are reflected
    with a 180° longitude shift directly in (lat, lon) space.
    """

    def __init__(self, in_channels, out_channels, spatial_resolution, num_heads,
                 lat_lon_grid, lat_top, lat_range, polar_fold=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.polar_fold = polar_fold

        self.flow_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2 * num_heads, kernel_size=1),
        )
        self.value_head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.register_buffer("base_grid_sph", lat_lon_grid)          # (1, H_p, W_p, 2)
        self.register_buffer("lat_top",   torch.tensor(float(lat_top)))
        self.register_buffer("lat_range", torch.tensor(float(lat_range)))

    def forward(self, u, meta=None):
        flow  = self.flow_head(u)
        value = self.value_head(u)

        flow = rearrange(flow, flow_fold_heads[2], dir=2, heads=self.num_heads)

        lon = self.base_grid_sph[..., 0] + flow[..., 0]
        lat = self.base_grid_sph[..., 1] + flow[..., 1]

        lon = lon % (2 * math.pi)

        gx = lon / math.pi - 1
        gy = 2.0 * (self.lat_top - lat) / self.lat_range - 1.0
        grid = torch.stack([gx, gy], dim=-1)

        value = rearrange(value, value_fold_heads[2], heads=self.num_heads)
        padding = ("polar", "periodic") if self.polar_fold else ("zeros", "periodic")
        u_warp = custom_grid_sample2d(
            value, grid, mode="bilinear", padding_modes=padding, align_corners=True,
        )
        return rearrange(u_warp, value_unfold_heads[2], heads=self.num_heads)


def _lat_to_gy(lat, lat_asc, gy_asc):
    """Monotonic lat → gy lookup. Reduces to the analytic affine formula on
    equiangular grids and handles Legendre–Gauss nodes correctly.

    ``lat_asc`` / ``gy_asc`` are grid-row lat/gy values in ASCENDING lat order.
    """
    idx = torch.searchsorted(lat_asc, lat, right=False).clamp(1, lat_asc.numel() - 1)
    lo = lat_asc[idx - 1]
    hi = lat_asc[idx]
    g_lo = gy_asc[idx - 1]
    g_hi = gy_asc[idx]
    t = (lat - lo) / (hi - lo).clamp_min(1e-12)
    return g_lo + t * (g_hi - g_lo)


@torch.compile
def _tangent_warp_coords(du, dv, e_east, e_north, base_pos_3d, lat_asc, gy_asc):
    """Rodrigues rotation + spherical→grid_sample conversion."""
    d_3d = du * e_east + dv * e_north
    theta = torch.sqrt(du ** 2 + dv ** 2 + 1e-12)
    sinc_t = torch.sin(theta) / theta
    pos_3d_new = base_pos_3d * torch.cos(theta) + d_3d * sinc_t

    r = torch.sqrt(pos_3d_new[..., 0] ** 2 + pos_3d_new[..., 1] ** 2)
    lat_new = torch.atan2(pos_3d_new[..., 2], r)
    lon_new = torch.atan2(pos_3d_new[..., 1], pos_3d_new[..., 0]) % (2 * math.pi)

    gx = lon_new / math.pi - 1
    gy = _lat_to_gy(lat_new, lat_asc, gy_asc)
    return torch.stack([gx, gy], dim=-1)


class TangentSelfWarp(nn.Module):
    """SelfWarp with flow in the local tangent plane, projected via 3D Rodrigues.

    Displacement heads predict (du, dv) = (East, North) shifts. These are
    projected into 3D Cartesian space using precomputed tangent basis vectors,
    added to the 3D grid positions, and normalised back onto the unit sphere.
    Pole crossing is handled naturally by the 3D projection.
    """

    def __init__(self, in_channels, out_channels, spatial_resolution, num_heads,
                 lat_lon_grid, lat_top, lat_range, **kwargs):
        super().__init__()
        self.num_heads = num_heads

        self.flow_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2 * num_heads, kernel_size=1),
        )
        self.value_head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # lat_lon_grid: (1, H_p, W_p, 2) with [..., 0]=lon, [..., 1]=lat in radians
        lon = lat_lon_grid[..., 0]
        lat = lat_lon_grid[..., 1]

        base_pos_3d = torch.stack([
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
        ], dim=-1)
        self.register_buffer("base_pos_3d", base_pos_3d)

        e_east = torch.stack([
            -torch.sin(lon),
             torch.cos(lon),
             torch.zeros_like(lon),
        ], dim=-1)
        self.register_buffer("e_east", e_east)

        e_north = torch.stack([
            -torch.sin(lat) * torch.cos(lon),
            -torch.sin(lat) * torch.sin(lon),
             torch.cos(lat),
        ], dim=-1)
        self.register_buffer("e_north", e_north)

        # Per-row lat → gy lookup (ascending-lat order). For equiangular grids
        # this reduces to gy = 2*(lat_top - lat)/lat_range - 1 to FP precision;
        # for Legendre–Gauss nodes it gives correct per-row coordinates.
        lat_nodes_desc = lat_lon_grid[0, :, 0, 1].contiguous().float()
        H_rows = lat_nodes_desc.numel()
        gy_nodes_desc = torch.linspace(-1.0, 1.0, H_rows, dtype=torch.float32)
        lat_asc = lat_nodes_desc.flip(0).contiguous()
        gy_asc = gy_nodes_desc.flip(0).contiguous()
        self.register_buffer("lat_asc", lat_asc)
        self.register_buffer("gy_asc", gy_asc)

    def forward(self, u, meta=None):
        flow  = self.flow_head(u)
        value = self.value_head(u)

        flow = rearrange(flow, flow_fold_heads[2], dir=2, heads=self.num_heads)

        du = flow[..., 0:1]
        dv = flow[..., 1:2]

        value = rearrange(value, value_fold_heads[2], heads=self.num_heads)

        grid = _tangent_warp_coords(du, dv, self.e_east, self.e_north,
                                    self.base_pos_3d, self.lat_asc, self.gy_asc)
        u_warp = custom_grid_sample2d(
            value, grid, mode="bilinear",
            padding_modes=("border", "periodic"), align_corners=True,
        )

        return rearrange(u_warp, value_unfold_heads[2], heads=self.num_heads)
