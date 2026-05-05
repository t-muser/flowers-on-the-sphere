"""DataModule for the MITgcm global-ocean cs32x15 dataset.

The dataset on disk is the **native cubed-sphere** Zarr produced by
:mod:`datagen.mitgcm.global_ocean.solver` — a single ``data`` variable
on dims ``(time, field, face, y, x)`` with five fields in the canonical
order :data:`datagen.mitgcm.global_ocean.regrid.FIELD_ORDER`. To train
models that expect a regular lat/lon grid, this module:

1. Rotates the ``(u_k2, v_k2)`` velocity pair from face-aligned to
   geographic ``(east, north)`` using ``angle_cs/sn`` from ``grid.zarr``.
2. Regrids each scalar channel from cs32 → ``(nlat, nlon)`` lat/lon
   with the precomputed weights from
   :mod:`datagen.mitgcm.global_ocean.regrid` (which wraps
   :mod:`datagen.cpl_aim_ocn.regrid`).
3. Concatenates static depth and per-window day-of-year ``(sin, cos)``
   into the input tensor as conditioning channels.
4. Returns a per-channel ``valid_mask`` so the loss can drop land cells.

Public surface mirrors :class:`fots.data.cpl_aim_ocn.CplAimOcnDataModule`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from datagen.mitgcm.global_ocean.regrid import (
    FIELD_ORDER,
    GlobalOceanLatLon,
    apply_dynamic,
    build as build_lat_lon,
    field_masks_ll,
)
from fots.data.datamodule import AbstractDataModule
from fots.data.zarr_dataset import ZarrMetadata

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"run_(\d+)\.zarr")

DEFAULT_NLAT = 64
DEFAULT_NLON = 128
DEFAULT_FIELD_NAMES: tuple[str, ...] = FIELD_ORDER
N_DYNAMIC = len(FIELD_ORDER)
N_STATIC_INPUT = 1   # depth
N_TIME_INPUT = 2     # sin(doy), cos(doy)
N_INPUT_CHANNELS = N_DYNAMIC + N_STATIC_INPUT + N_TIME_INPUT  # 8

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.0


def _scan_run_ids(split_dir: Path) -> list[int]:
    ids = []
    for p in split_dir.iterdir():
        m = _RUN_RE.fullmatch(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _resolve_split_run_ids(root: Path) -> dict[str, list[int]]:
    splits_path = root / "splits.json"
    if splits_path.is_file():
        with open(splits_path) as f:
            splits = json.load(f)
        return {s: sorted(int(i) for i in splits[s]) for s in ("train", "val", "test")}
    return {s: _scan_run_ids(root / s) for s in ("train", "val", "test")}


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Map raw depth (m, 0=land) to a roughly [0, 1] static channel.

    log10-scaled so a 50 m shelf and a 5000 m abyss are both informative,
    not buried by the dynamic range.
    """
    out = np.log10(np.clip(depth, 1.0, None)) / 4.0
    return out.astype(np.float32)


def _doy_channels(time_seconds: np.ndarray) -> np.ndarray:
    """Day-of-year sin/cos features for a vector of time stamps in seconds.

    Uses ``time/86400 % 365`` as the phase. After the spinup-trim rebase
    the trajectory's ``time[0] == 0``; the absolute pickup epoch is not
    exposed by MITgcm so we treat the trajectory start as Jan 1 (the
    constant offset is the same for every run, so the model just learns
    a shifted phase if that assumption is wrong).
    """
    days = time_seconds / _SECONDS_PER_DAY
    phase = 2 * np.pi * (days % _DAYS_PER_YEAR) / _DAYS_PER_YEAR
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)  # (T, 2)


class _GlobalOceanWindowDataset(Dataset):
    """Sliding-window sampler over per-run cs32 global-ocean zarrs.

    Each ``__getitem__`` opens the relevant run (cached per-worker),
    rotates u/v on the cs grid, regrids cs32 → lat/lon with the
    precomputed weights, z-scores the dynamic channels, concatenates the
    static depth + day-of-year channels, and returns

    ::

        {"input_fields":  Tensor(T_in,  C_in,  nlat, nlon),
         "output_fields": Tensor(T_out, C_out, nlat, nlon),
         "valid_mask":    Tensor(C_out, nlat, nlon)}

    where ``C_in = 8`` (5 dyn + 1 depth + 2 doy) and ``C_out = 5``.
    """

    def __init__(
        self,
        split_dir: Path,
        run_ids: list[int],
        *,
        n_steps_input: int,
        n_steps_output: int,
        time_steps_per_run: int,
        grid_ll: GlobalOceanLatLon,
        full_trajectory: bool = False,
        max_rollout_steps: Optional[int] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ):
        self.split_dir = Path(split_dir)
        self.run_ids = list(run_ids)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.time_steps_per_run = int(time_steps_per_run)
        self.full_trajectory = bool(full_trajectory)
        self._grid_ll = grid_ll

        if self.full_trajectory:
            if max_rollout_steps is None:
                max_rollout_steps = self.time_steps_per_run - self.n_steps_input
            self.n_steps_output = int(max_rollout_steps)

        total_window = self.n_steps_input + self.n_steps_output
        if total_window > self.time_steps_per_run:
            raise ValueError(
                f"window {total_window} exceeds trajectory length "
                f"{self.time_steps_per_run}"
            )
        self._windows_per_run = (
            1 if self.full_trajectory
            else self.time_steps_per_run - total_window + 1
        )

        # Per-channel stats for the *dynamic* fields, broadcast over (T, C, H, W).
        self._norm_mean = (
            norm_mean.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_mean is not None else None
        )
        self._norm_std = (
            norm_std.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_std is not None else None
        )

        # Static channel: depth on lat/lon, broadcast to (1, 1, H, W).
        depth_norm = _normalize_depth(grid_ll.depth_ll)
        self._static_depth = depth_norm[None, None, :, :].astype(np.float32)

        # Per-channel mask in C-axis order matching FIELD_ORDER, on lat/lon.
        masks = field_masks_ll(grid_ll)
        mask_stack = np.stack(
            [masks[name].astype(np.float32) for name in FIELD_ORDER], axis=0
        )  # (5, H, W)
        self._valid_mask_np = mask_stack
        # Broadcast-friendly version for in-loader masking of dynamic input/output.
        self._mask_dyn_btch = mask_stack[None, :, :, :]  # (1, 5, H, W)

        # Per-process zarr handle cache; cleared on pickle.
        self._zarr_cache: dict[int, xr.Dataset] = {}

    def __len__(self) -> int:
        return len(self.run_ids) * self._windows_per_run

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zarr_cache"] = {}
        return state

    def _open_run(self, run_id: int) -> xr.Dataset:
        ds = self._zarr_cache.get(run_id)
        if ds is None:
            path = self.split_dir / f"run_{run_id:04d}.zarr"
            ds = xr.open_zarr(str(path), consolidated=True)
            self._zarr_cache[run_id] = ds
        return ds

    def _window_to_arrays(
        self, ds: xr.Dataset, t0: int, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice → rotate u/v → regrid → return (dynamic_lat_lon, time_seconds).

        Output dynamic shape: ``(T, 5, nlat, nlon)`` float32.
        """
        window = ds.isel(time=slice(t0, t0 + T))
        cube = window["data"].values  # (T, 5, face, y, x)
        latlon = apply_dynamic(cube, self._grid_ll)  # (T, 5, nlat, nlon)
        time_seconds = window["time"].values.astype(np.float64)
        return latlon, time_seconds

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        run_idx, window_idx = divmod(idx, self._windows_per_run)
        run_id = self.run_ids[run_idx]
        t0 = 0 if self.full_trajectory else window_idx
        ds = self._open_run(run_id)

        x_dyn, x_time = self._window_to_arrays(ds, t0, self.n_steps_input)
        y_dyn, _ = self._window_to_arrays(
            ds, t0 + self.n_steps_input, self.n_steps_output,
        )

        # Zero land cells before z-score so normalised land stays at the
        # field mean (i.e. exactly zero), avoiding step gradients into the
        # model. Loss masking handles output-side correctness.
        x_dyn = x_dyn * self._mask_dyn_btch
        y_dyn = y_dyn * self._mask_dyn_btch

        if self._norm_mean is not None:
            x_dyn = (x_dyn - self._norm_mean) / self._norm_std
            y_dyn = (y_dyn - self._norm_mean) / self._norm_std
            # Re-zero land after normalisation (mean ≠ 0 in raw units).
            x_dyn = x_dyn * self._mask_dyn_btch
            y_dyn = y_dyn * self._mask_dyn_btch

        # Static depth channel — same value at every input timestep.
        T_in = x_dyn.shape[0]
        H, W = x_dyn.shape[-2:]
        depth_t = np.broadcast_to(self._static_depth, (T_in, 1, H, W))

        # Day-of-year channels — same value across (H, W), varies per t.
        doy = _doy_channels(x_time)  # (T_in, 2)
        doy_b = np.broadcast_to(doy[:, :, None, None], (T_in, 2, H, W))

        x_input = np.concatenate([x_dyn, depth_t, doy_b], axis=1).astype(np.float32)

        return {
            "input_fields":  torch.tensor(x_input),
            "output_fields": torch.tensor(y_dyn.astype(np.float32)),
            "valid_mask":    torch.tensor(self._valid_mask_np),
        }


class GlobalOceanDataModule(AbstractDataModule):
    """Data module for the MITgcm global-ocean cs32x15 dataset.

    Same overall surface as
    :class:`fots.data.cpl_aim_ocn.CplAimOcnDataModule`. Adds:

    * Vector rotation for ``(u_k2, v_k2)`` before regrid.
    * Static ``depth`` channel and per-timestep ``sin/cos(doy)`` channels in
      ``input_fields``.
    * Per-channel ``valid_mask`` in each batch dict for masked loss
      computation.
    """

    def __init__(
        self,
        root: str,
        time_steps_per_run: int,
        dim_in: int = N_INPUT_CHANNELS,
        dim_out: int = N_DYNAMIC,
        spatial_resolution: tuple[int, int] = (DEFAULT_NLAT, DEFAULT_NLON),
        grid: str = "equiangular",
        field_names: tuple[str, ...] = DEFAULT_FIELD_NAMES,
        batch_size: int = 2,
        n_steps_input: int = 4,
        n_steps_output: int = 1,
        max_rollout_steps: Optional[int] = None,
        num_workers: int = 0,
        dataset_name: str = "global_ocean",
        rank: int = 0,
        world_size: int = 1,
        regrid_method: str = "idw",
        regrid_k: int = 4,
    ):
        self.root = Path(root)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.max_rollout_steps = max_rollout_steps
        self.num_workers = int(num_workers)
        self.dataset_name = dataset_name
        self.time_steps_per_run = int(time_steps_per_run)

        nlat, nlon = int(spatial_resolution[0]), int(spatial_resolution[1])
        self._metadata = ZarrMetadata(
            dataset_name=dataset_name,
            dim_in=int(dim_in),
            dim_out=int(dim_out),
            spatial_resolution=(nlat, nlon),
            grid=grid,
            field_names=tuple(field_names),
        )

        split_run_ids = _resolve_split_run_ids(self.root)
        logger.info(
            "%s splits from %s: train=%d val=%d test=%d",
            dataset_name, self.root,
            len(split_run_ids["train"]),
            len(split_run_ids["val"]),
            len(split_run_ids["test"]),
        )

        # Build lat/lon grid info once. The cs grid is fixed across runs.
        grid_zarr = self.root / "grid.zarr"
        if not grid_zarr.exists():
            raise FileNotFoundError(
                f"{grid_zarr} not found — run "
                f"`datagen.mitgcm.global_ocean.scripts.extract_grid` first."
            )
        self._grid_ll = build_lat_lon(
            grid_zarr, nlat=nlat, nlon=nlon,
            method=regrid_method, k=regrid_k,
        )
        logger.info(
            "%s regrid weights: cs32 → (%d, %d) method=%s k=%d",
            dataset_name, nlat, nlon, regrid_method, self._grid_ll.weights.k,
        )

        # Z-score stats over the dynamic fields only.
        self._norm_mean_np: Optional[np.ndarray] = None
        self._norm_std_np: Optional[np.ndarray] = None
        self._norm_mean_t: Optional[torch.Tensor] = None
        self._norm_std_t: Optional[torch.Tensor] = None
        stats_path = self.root / "stats.json"
        if stats_path.is_file():
            with open(stats_path) as f:
                stats = json.load(f)
            try:
                means = np.array([float(stats[n]["mean"]) for n in FIELD_ORDER],
                                 dtype=np.float32)
                stds = np.array([float(stats[n]["std"]) for n in FIELD_ORDER],
                                dtype=np.float32)
            except KeyError as e:
                logger.warning("stats.json missing entry for %s; skipping normalization", e)
            else:
                self._norm_mean_np = means
                self._norm_std_np = stds
                self._norm_mean_t = torch.from_numpy(means).reshape(1, -1, 1, 1)
                self._norm_std_t = torch.from_numpy(stds).reshape(1, -1, 1, 1)
        else:
            logger.info("%s: no stats.json at %s; training on raw fields",
                        dataset_name, stats_path)

        def _make(split: str, full_trajectory: bool = False) -> _GlobalOceanWindowDataset:
            return _GlobalOceanWindowDataset(
                split_dir=self.root / split,
                run_ids=split_run_ids[split],
                n_steps_input=self.n_steps_input,
                n_steps_output=self.n_steps_output,
                time_steps_per_run=self.time_steps_per_run,
                grid_ll=self._grid_ll,
                full_trajectory=full_trajectory,
                max_rollout_steps=self.max_rollout_steps,
                norm_mean=self._norm_mean_np,
                norm_std=self._norm_std_np,
            )

        self.train_dataset = _make("train")
        self.val_dataset = _make("val")
        self.test_dataset = _make("test")
        self.rollout_val_dataset = _make("val", full_trajectory=True)
        self.rollout_test_dataset = _make("test", full_trajectory=True)

    def denormalize_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_mean_t is None:
            return x
        mean = self._norm_mean_t.to(x.device, x.dtype)
        std = self._norm_std_t.to(x.device, x.dtype)
        if x.dim() == 5:  # (B, T, C, H, W)
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return x * std + mean

    @property
    def metadata(self) -> ZarrMetadata:
        return self._metadata

    def _loader(
        self, dataset: Dataset, *,
        shuffle: bool, batch_size: Optional[int] = None, distribute: bool = True,
    ) -> DataLoader:
        bs = batch_size if batch_size is not None else self.batch_size
        if distribute and self.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=shuffle, drop_last=True,
            )
            return DataLoader(
                dataset, batch_size=bs, sampler=sampler,
                num_workers=self.num_workers, pin_memory=False, drop_last=True,
            )
        return DataLoader(
            dataset, batch_size=bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=False, drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def rollout_val_dataloader(self) -> DataLoader:
        return self._loader(self.rollout_val_dataset, shuffle=False,
                            batch_size=1, distribute=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

    def rollout_test_dataloader(self) -> DataLoader:
        return self._loader(self.rollout_test_dataset, shuffle=False,
                            batch_size=1, distribute=False)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.root}>"
