"""DataModule for the MITgcm global-ocean 3-D dataset.

The 3-D variant of the dataset is produced by
:func:`datagen.mitgcm.global_ocean.solver.write_cubed_sphere_zarr_3d`. Each
``run.zarr`` exposes per-variable arrays with a depth axis:

* ``theta``, ``salt``, ``u``, ``v``: ``(time, level, face, y, x)``
* ``eta``: ``(time, face, y, x)``

This DataModule mirrors :class:`fots.data.global_ocean.GlobalOceanDataModule`
but folds the depth axis into the channel axis at regrid time. The output
batches have shape ``(B, T, C, nlat, nlon)`` with
``C = len(FIELDS_3D) * Nlevel + len(FIELDS_2D)``.
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
    FIELDS_2D,
    FIELDS_3D,
    GlobalOceanLatLon,
    apply_dynamic_3d,
    build as build_lat_lon,
    field_masks_3d_ll,
    field_names_3d,
)
from fots.data.datamodule import AbstractDataModule
from fots.data.zarr_dataset import ZarrMetadata

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"run_(\d+)\.zarr")

DEFAULT_NLAT = 64
DEFAULT_NLON = 128
N_STATIC_INPUT = 1   # depth
N_TIME_INPUT = 2     # sin(doy), cos(doy)

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
    out = np.log10(np.clip(depth, 1.0, None)) / 4.0
    return out.astype(np.float32)


def _doy_channels(time_seconds: np.ndarray) -> np.ndarray:
    days = time_seconds / _SECONDS_PER_DAY
    phase = 2 * np.pi * (days % _DAYS_PER_YEAR) / _DAYS_PER_YEAR
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)


def _peek_levels(any_run_zarr: Path) -> np.ndarray:
    """Read the ``level`` coord from any run.zarr to learn Nlevel."""
    ds = xr.open_zarr(str(any_run_zarr), consolidated=False)
    return ds["level"].values.astype(np.int64)


class _GlobalOcean3DWindowDataset(Dataset):
    """Sliding-window sampler over per-run 3-D global-ocean zarrs."""

    def __init__(
        self,
        split_dir: Path,
        run_ids: list[int],
        *,
        n_steps_input: int,
        n_steps_output: int,
        time_steps_per_run: int,
        grid_ll: GlobalOceanLatLon,
        level_idx: np.ndarray,
        full_trajectory: bool = False,
        max_rollout_steps: Optional[int] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        impute_land: bool = True,
    ):
        self.split_dir = Path(split_dir)
        self.run_ids = list(run_ids)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.time_steps_per_run = int(time_steps_per_run)
        self.full_trajectory = bool(full_trajectory)
        self._grid_ll = grid_ll
        self._level_idx = np.asarray(level_idx, dtype=np.int64)
        self._impute_land = bool(impute_land)

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

        # Per-channel z-score broadcast over (T, C, H, W).
        self._norm_mean = (
            norm_mean.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_mean is not None else None
        )
        self._norm_std = (
            norm_std.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_std is not None else None
        )

        depth_norm = _normalize_depth(grid_ll.depth_ll)
        self._static_depth = depth_norm[None, None, :, :].astype(np.float32)

        # Per-channel mask in C-axis order matching field_names_3d.
        mask_stack = field_masks_3d_ll(grid_ll, self._level_idx).astype(np.float32)
        self._valid_mask_np = mask_stack
        self._mask_dyn_btch = mask_stack[None, :, :, :]  # (1, C, H, W)

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
        window = ds.isel(time=slice(t0, t0 + T))
        # Subset by configured levels at load time so we only pay the read
        # + regrid cost for the levels the model actually consumes.
        window_3d = window.sel(level=list(int(v) for v in self._level_idx))
        fields_3d = {name: window_3d[name].values for name in FIELDS_3D}
        fields_2d = {name: window[name].values for name in FIELDS_2D}
        latlon = apply_dynamic_3d(
            fields_3d, fields_2d, self._grid_ll,
            level_idx=self._level_idx, impute_land=self._impute_land,
        )
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

        x_dyn = x_dyn * self._mask_dyn_btch
        y_dyn = y_dyn * self._mask_dyn_btch

        if self._norm_mean is not None:
            x_dyn = (x_dyn - self._norm_mean) / self._norm_std
            y_dyn = (y_dyn - self._norm_mean) / self._norm_std
            x_dyn = x_dyn * self._mask_dyn_btch
            y_dyn = y_dyn * self._mask_dyn_btch

        T_in = x_dyn.shape[0]
        H, W = x_dyn.shape[-2:]
        depth_t = np.broadcast_to(self._static_depth, (T_in, 1, H, W))

        doy = _doy_channels(x_time)
        doy_b = np.broadcast_to(doy[:, :, None, None], (T_in, 2, H, W))

        x_input = np.concatenate([x_dyn, depth_t, doy_b], axis=1).astype(np.float32)

        return {
            "input_fields":  torch.tensor(x_input),
            "output_fields": torch.tensor(y_dyn.astype(np.float32)),
            "valid_mask":    torch.tensor(self._valid_mask_np),
        }


class GlobalOcean3DDataModule(AbstractDataModule):
    """3-D variant of :class:`fots.data.global_ocean.GlobalOceanDataModule`.

    The level axis is folded into the channel axis at regrid time, so
    downstream models see a flat ``(T, C, H, W)`` stack with
    ``C = len(FIELDS_3D) * Nlevel + len(FIELDS_2D)`` dynamic channels plus
    the same ``depth + sin/cos(doy)`` static channels.
    """

    def __init__(
        self,
        root: str,
        time_steps_per_run: int,
        dim_in: Optional[int] = None,
        dim_out: Optional[int] = None,
        spatial_resolution: tuple[int, int] = (DEFAULT_NLAT, DEFAULT_NLON),
        grid: str = "equiangular",
        batch_size: int = 2,
        n_steps_input: int = 4,
        n_steps_output: int = 1,
        max_rollout_steps: Optional[int] = None,
        num_workers: int = 0,
        dataset_name: str = "global_ocean_3d",
        rank: int = 0,
        world_size: int = 1,
        regrid_method: str = "idw",
        regrid_k: int = 4,
        levels: Optional[list[int]] = None,
        impute_land: bool = True,
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

        split_run_ids = _resolve_split_run_ids(self.root)
        logger.info(
            "%s splits from %s: train=%d val=%d test=%d",
            dataset_name, self.root,
            len(split_run_ids["train"]),
            len(split_run_ids["val"]),
            len(split_run_ids["test"]),
        )

        # Look at any available run to discover the on-disk level coord, then
        # subset by ``levels`` if requested.
        first_split = next(
            s for s in ("train", "val", "test") if split_run_ids[s]
        )
        first_id = split_run_ids[first_split][0]
        peek_path = self.root / first_split / f"run_{first_id:04d}.zarr"
        all_levels = _peek_levels(peek_path)
        if levels is None:
            self._level_idx = all_levels
        else:
            sel = np.asarray(levels, dtype=np.int64)
            missing = [int(x) for x in sel if int(x) not in set(int(v) for v in all_levels)]
            if missing:
                raise ValueError(
                    f"levels {missing} not present in run.zarr "
                    f"(available: {list(all_levels)})"
                )
            self._level_idx = sel
        Nlevel = int(self._level_idx.size)
        self._impute_land = bool(impute_land)
        n_dynamic = len(FIELDS_3D) * Nlevel + len(FIELDS_2D)
        n_input = n_dynamic + N_STATIC_INPUT + N_TIME_INPUT

        self._field_names = field_names_3d(self._level_idx)
        self._metadata = ZarrMetadata(
            dataset_name=dataset_name,
            dim_in=int(dim_in) if dim_in is not None else n_input,
            dim_out=int(dim_out) if dim_out is not None else n_dynamic,
            spatial_resolution=(nlat, nlon),
            grid=grid,
            field_names=self._field_names,
        )

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
        if self._grid_ll.mask_c_3d_ll is None or self._grid_ll.mask_w_3d_ll is None:
            raise RuntimeError(
                f"{grid_zarr} has no per-level masks; rebuild it with the "
                "updated extract_grid.py before training the 3-D model."
            )
        logger.info(
            "%s regrid weights: cs32 → (%d, %d) method=%s k=%d  "
            "Nlevel=%d  C_dyn=%d",
            dataset_name, nlat, nlon, regrid_method, self._grid_ll.weights.k,
            Nlevel, n_dynamic,
        )

        self._norm_mean_np: Optional[np.ndarray] = None
        self._norm_std_np: Optional[np.ndarray] = None
        self._norm_mean_t: Optional[torch.Tensor] = None
        self._norm_std_t: Optional[torch.Tensor] = None
        stats_path = self.root / "stats.json"
        if stats_path.is_file():
            with open(stats_path) as f:
                stats = json.load(f)
            try:
                means = np.array(
                    [float(stats[n]["mean"]) for n in self._field_names],
                    dtype=np.float32,
                )
                stds = np.array(
                    [float(stats[n]["std"]) for n in self._field_names],
                    dtype=np.float32,
                )
            except KeyError as e:
                logger.warning(
                    "stats.json missing entry for %s; skipping normalization", e,
                )
            else:
                self._norm_mean_np = means
                self._norm_std_np = stds
                self._norm_mean_t = torch.from_numpy(means).reshape(1, -1, 1, 1)
                self._norm_std_t = torch.from_numpy(stds).reshape(1, -1, 1, 1)
        else:
            logger.info("%s: no stats.json at %s; training on raw fields",
                        dataset_name, stats_path)

        def _make(split: str, full_trajectory: bool = False) -> _GlobalOcean3DWindowDataset:
            return _GlobalOcean3DWindowDataset(
                split_dir=self.root / split,
                run_ids=split_run_ids[split],
                n_steps_input=self.n_steps_input,
                n_steps_output=self.n_steps_output,
                time_steps_per_run=self.time_steps_per_run,
                grid_ll=self._grid_ll,
                level_idx=self._level_idx,
                impute_land=self._impute_land,
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
        if x.dim() == 5:
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
