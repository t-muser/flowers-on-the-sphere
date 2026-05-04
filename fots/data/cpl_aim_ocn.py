"""DataModule for the coupled AIM+ocean cs32 dataset.

The dataset on disk is the **native cubed-sphere** Zarr produced by
:func:`datagen.cpl_aim_ocn.zarr_writer.write_cs32_zarr` — separate
``atm_*`` / ``ocn_*`` data variables on dims ``face, j, i`` (3-D atm
fields additionally on ``Zsigma``). To train models that expect a
regular lat/lon grid, this module:

1. Stacks every cs32 stream into the canonical 35-channel axis defined
   by :func:`datagen.cpl_aim_ocn.channels.channel_names` (3-D atm
   streams contribute one channel per σ-level).
2. Regrids each sample on the fly from cs32 → ``(nlat, nlon)`` lat/lon
   using :mod:`datagen.cpl_aim_ocn.regrid`. Regrid weights are built
   once from a representative run's XC/YC and reused thereafter.
3. Applies per-channel z-score normalization from ``<root>/stats.json``
   (written by ``scripts/finalize_dataset.py``).

Public surface mirrors :class:`fots.data.zarr_dataset.ZarrDataModule`,
so swapping the ``_target_`` in a config is enough to switch datasets.
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

from datagen.cpl_aim_ocn.channels import channel_names, expand_to_channels
from datagen.cpl_aim_ocn.regrid import (
    DEFAULT_NLAT,
    DEFAULT_NLON,
    RegridWeights,
    apply_weights,
    build_weights,
)
from fots.data.datamodule import AbstractDataModule
from fots.data.zarr_dataset import ZarrMetadata

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"run_(\d+)\.zarr")


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


class _CplAimOcnWindowDataset(Dataset):
    """Sliding-window sampler over per-run cs32 zarrs.

    Each ``__getitem__`` opens the relevant run (cached per-worker),
    slices the time axis, expands the cs32 streams into the canonical
    35-channel stack, regrids cs32 → lat/lon with the precomputed
    weights, and z-scores per channel. Returned tensors have shape
    ``(T, 35, nlat, nlon)``.
    """

    def __init__(
        self,
        split_dir: Path,
        run_ids: list[int],
        *,
        n_steps_input: int,
        n_steps_output: int,
        time_steps_per_run: int,
        weights: RegridWeights,
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
        self._weights = weights

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

        # Per-channel stats reshaped for broadcast over (T, C, H, W).
        self._norm_mean = (
            norm_mean.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_mean is not None else None
        )
        self._norm_std = (
            norm_std.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_std is not None else None
        )

        # Per-process zarr handle cache; cleared on pickle so DataLoader
        # workers each open their own file handles.
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

    def _window_to_array(self, ds: xr.Dataset, t0: int, T: int) -> np.ndarray:
        """Slice, expand, regrid → ``(T, channel, nlat, nlon)`` float32."""
        window = ds.isel(time=slice(t0, t0 + T))
        stacked = expand_to_channels(window)
        cube = stacked.transpose("time", "channel", "face", "j", "i").values
        regridded = apply_weights(cube, self._weights)
        return regridded.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        run_idx, window_idx = divmod(idx, self._windows_per_run)
        run_id = self.run_ids[run_idx]
        t0 = 0 if self.full_trajectory else window_idx
        ds = self._open_run(run_id)

        x = self._window_to_array(ds, t0, self.n_steps_input)
        y = self._window_to_array(
            ds, t0 + self.n_steps_input, self.n_steps_output,
        )

        if self._norm_mean is not None:
            x = (x - self._norm_mean) / self._norm_std
            y = (y - self._norm_mean) / self._norm_std

        # `torch.tensor` allocates a fresh resizable storage so the
        # DataLoader's pin_memory collator works (see comment in
        # zarr_dataset._ZarrWindowDataset).
        return {
            "input_fields": torch.tensor(x),
            "output_fields": torch.tensor(y),
        }


class CplAimOcnDataModule(AbstractDataModule):
    """Data module for the coupled AIM+ocean cs32 dataset.

    The dataset root has the same train/val/test layout as
    :class:`fots.data.zarr_dataset.ZarrDataModule`, but the per-run
    zarrs are cs32 native rather than ``(time, field, lat, lon)``.

    Parameters
    ----------
    root
        Dataset root with ``train/``, ``val/``, ``test/`` subdirectories
        containing ``run_XXXX.zarr`` cs32 stores.
    time_steps_per_run
        Trajectory length in each store (data-phase snapshot count).
    dim_in, dim_out
        Per-snapshot channel counts. Should both equal 35 (the canonical
        :func:`channel_names`).
    spatial_resolution
        ``(nlat, nlon)`` target grid for the load-time regrid.
        Defaults to (64, 128) — equiangular at ~2.8°.
    grid
        Grid type tag forwarded to ``ZarrMetadata``.
    field_names
        Channel names; should equal :func:`channel_names`. Used for the
        stats lookup and exposed via :attr:`metadata`.
    batch_size, n_steps_input, n_steps_output, max_rollout_steps,
    num_workers, dataset_name, rank, world_size
        Same semantics as :class:`ZarrDataModule`.
    regrid_method, regrid_k
        Forwarded to :func:`datagen.cpl_aim_ocn.regrid.build_weights`.
        Default ``"nearest"`` matches the stats pass in
        ``finalize_dataset.py``.
    """

    def __init__(
        self,
        root: str,
        time_steps_per_run: int,
        dim_in: int,
        dim_out: int,
        spatial_resolution: tuple[int, int] = (DEFAULT_NLAT, DEFAULT_NLON),
        grid: str = "equiangular",
        field_names: tuple[str, ...] = (),
        batch_size: int = 2,
        n_steps_input: int = 2,
        n_steps_output: int = 1,
        max_rollout_steps: Optional[int] = None,
        num_workers: int = 0,
        dataset_name: str = "cpl_aim_ocn",
        rank: int = 0,
        world_size: int = 1,
        regrid_method: str = "nearest",
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

        if not field_names:
            field_names = channel_names()
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

        # Build regrid weights from the first run we can open. The cs32
        # grid is fixed by code_atm/SIZE.h so any run produces the same
        # XC/YC.
        weights = self._build_weights(
            split_run_ids, nlat=nlat, nlon=nlon,
            method=regrid_method, k=regrid_k,
        )
        logger.info(
            "%s regrid weights: cs32 → (%d, %d) method=%s k=%d",
            dataset_name, nlat, nlon, regrid_method, weights.k,
        )

        # Per-channel z-score stats from <root>/stats.json (in field-first
        # format).
        self._norm_mean_np: Optional[np.ndarray] = None
        self._norm_std_np: Optional[np.ndarray] = None
        self._norm_mean_t: Optional[torch.Tensor] = None
        self._norm_std_t: Optional[torch.Tensor] = None
        stats_path = self.root / "stats.json"
        if field_names and stats_path.is_file():
            with open(stats_path) as f:
                stats = json.load(f)
            try:
                means = np.array([float(stats[n]["mean"]) for n in field_names],
                                 dtype=np.float32)
                stds = np.array([float(stats[n]["std"]) for n in field_names],
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

        def _make(split: str, full_trajectory: bool = False) -> _CplAimOcnWindowDataset:
            return _CplAimOcnWindowDataset(
                split_dir=self.root / split,
                run_ids=split_run_ids[split],
                n_steps_input=self.n_steps_input,
                n_steps_output=self.n_steps_output,
                time_steps_per_run=self.time_steps_per_run,
                weights=weights,
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

    def _build_weights(
        self, split_run_ids: dict[str, list[int]], *,
        nlat: int, nlon: int, method: str, k: int,
    ) -> RegridWeights:
        for split in ("train", "val", "test"):
            for rid in split_run_ids[split]:
                path = self.root / split / f"run_{rid:04d}.zarr"
                if path.exists():
                    ds = xr.open_zarr(str(path), consolidated=True)
                    return build_weights(
                        ds["XC"].values, ds["YC"].values,
                        nlat=nlat, nlon=nlon, method=method, k=k,
                    )
        raise FileNotFoundError(
            f"No run zarrs under {self.root}/{{train,val,test}} — "
            f"cannot read XC/YC to build regrid weights"
        )

    def denormalize_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_mean_t is None:
            return x
        mean = self._norm_mean_t.to(x.device, x.dtype)
        std = self._norm_std_t.to(x.device, x.dtype)
        if x.dim() == 5:  # (B, T, C, H, W) — broadcast over T
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
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=True,
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
