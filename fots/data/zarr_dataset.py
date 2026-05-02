"""Shared zarr-based dataset and data module for our custom PDE ensembles.

All datasets share the same on-disk layout::

    <root>/train/run_XXXX.zarr
    <root>/val/run_XXXX.zarr
    <root>/test/run_XXXX.zarr

Each zarr store has a ``fields`` variable with dims ``(time, field, lat, lon)``.
If ``<root>/stats.json`` exists the loader z-scores each field (raw fields
have radically different scales — h has σ≈530, vorticity σ≈2.5e-5 — so
unnormalised MSE is dominated by h and learns nothing about vorticity).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from fots.data.datamodule import AbstractDataModule

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"run_(\d+)\.zarr")


def _scan_run_ids(split_dir: Path) -> list[int]:
    """Return sorted run IDs found in ``split_dir`` as ``run_XXXX.zarr``."""
    ids = []
    for p in split_dir.iterdir():
        m = _RUN_RE.fullmatch(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


@dataclass
class ZarrMetadata:
    """Metadata surface consumed by ``fots.train.build_model``."""

    dataset_name: str
    dim_in: int
    dim_out: int
    spatial_resolution: tuple[int, int]
    grid: str = "equiangular"
    field_names: tuple[str, ...] = field(default_factory=tuple)
    n_spatial_dims: int = 2


class _ZarrWindowDataset(Dataset):
    """Sliding windows over per-run zarr stores under a single directory.

    Each store must expose a ``fields`` variable with dims
    ``(time, field, lat, lon)``. Samples are dicts::

        {"input_fields": Tensor(T_in, C, H, W),
         "output_fields": Tensor(T_out, C, H, W)}

    ``full_trajectory=True`` collapses indexing to one sample per run
    (``t0=0``, ``T_out=max_rollout_steps``) for rollout evaluation.
    """

    def __init__(
        self,
        split_dir: Path,
        run_ids: Sequence[int],
        n_steps_input: int,
        n_steps_output: int,
        *,
        time_steps_per_run: int,
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
        # Per-channel z-score stats (shape (1, C, 1, 1) for broadcast over
        # (T, C, H, W)). None disables normalization.
        self._norm_mean = (
            norm_mean.reshape(1, -1, 1, 1).astype(np.float32) if norm_mean is not None else None
        )
        self._norm_std = (
            norm_std.reshape(1, -1, 1, 1).astype(np.float32) if norm_std is not None else None
        )

        if self.full_trajectory:
            if max_rollout_steps is None:
                max_rollout_steps = self.time_steps_per_run - self.n_steps_input
            self.n_steps_output = int(max_rollout_steps)

        total_window = self.n_steps_input + self.n_steps_output
        if total_window > self.time_steps_per_run:
            raise ValueError(
                f"window {total_window} (n_steps_input={self.n_steps_input} + "
                f"n_steps_output={self.n_steps_output}) exceeds trajectory "
                f"length {self.time_steps_per_run}"
            )

        self._windows_per_run = (
            1 if self.full_trajectory
            else self.time_steps_per_run - total_window + 1
        )
        # Per-process cache; cleared on pickle so DataLoader workers each open
        # their own file handles.
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        run_idx, window_idx = divmod(idx, self._windows_per_run)
        run_id = self.run_ids[run_idx]
        t0 = 0 if self.full_trajectory else window_idx
        fields = self._open_run(run_id)["fields"]

        x = fields.isel(time=slice(t0, t0 + self.n_steps_input)).to_numpy().astype(np.float32, copy=False)
        y = fields.isel(
            time=slice(t0 + self.n_steps_input, t0 + self.n_steps_input + self.n_steps_output)
        ).to_numpy().astype(np.float32, copy=False)

        if self._norm_mean is not None:
            x = (x - self._norm_mean) / self._norm_std
            y = (y - self._norm_mean) / self._norm_std

        # `torch.from_numpy(x)` aliases x's buffer; if x came out of an xarray
        # zarr-backed array its storage is non-resizable and the DataLoader's
        # pin_memory collator will trip "Trying to resize storage that is not
        # resizable". `torch.tensor(x)` allocates a fresh resizable storage.
        return {
            "input_fields": torch.tensor(x),
            "output_fields": torch.tensor(y),
        }


class ZarrDataModule(AbstractDataModule):
    """Data module for zarr-based PDE datasets with train/val/test directories.

    Parameters
    ----------
    root:
        Dataset root with ``train/``, ``val/``, ``test/`` subdirectories
        each containing ``run_XXXX.zarr`` stores.
    time_steps_per_run:
        Trajectory length (number of time steps) in each zarr store.
    dim_in, dim_out:
        Number of field channels per input/output snapshot.
    spatial_resolution:
        ``(lat, lon)`` grid size.
    grid:
        Grid type, e.g. ``"equiangular"`` or ``"legendre-gauss"``.
    field_names:
        Human-readable names for each field channel (for logging/metadata only).
    batch_size:
        Batch size for all dataloaders.
    n_steps_input, n_steps_output:
        Input and target window lengths in snapshots.
    max_rollout_steps:
        Target length for rollout evaluation. Defaults to the full remaining
        trajectory after the input window.
    num_workers:
        DataLoader worker count.
    dataset_name:
        Tag used for experiment naming and logging.
    """

    def __init__(
        self,
        root: str,
        time_steps_per_run: int,
        dim_in: int,
        dim_out: int,
        spatial_resolution: tuple[int, int],
        grid: str = "equiangular",
        field_names: tuple[str, ...] = (),
        batch_size: int = 2,
        n_steps_input: int = 2,
        n_steps_output: int = 1,
        max_rollout_steps: Optional[int] = None,
        num_workers: int = 0,
        dataset_name: str = "zarr",
        rank: int = 0,
        world_size: int = 1,
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
        self._metadata = ZarrMetadata(
            dataset_name=dataset_name,
            dim_in=int(dim_in),
            dim_out=int(dim_out),
            spatial_resolution=tuple(spatial_resolution),
            grid=grid,
            field_names=tuple(field_names),
        )

        # Auto-load <root>/stats.json if present, ordered to match field_names.
        # _norm_*_t are torch tensors used by denormalize_fn; _norm_*_np feed
        # the per-sample numpy normalization in _ZarrWindowDataset.
        self._norm_mean_np: Optional[np.ndarray] = None
        self._norm_std_np: Optional[np.ndarray] = None
        self._norm_mean_t: Optional[torch.Tensor] = None
        self._norm_std_t: Optional[torch.Tensor] = None
        stats_path = self.root / "stats.json"
        if field_names and stats_path.is_file():
            with open(stats_path) as f:
                stats = json.load(f)
            try:
                means = np.array([float(stats[n]["mean"]) for n in field_names], dtype=np.float32)
                stds = np.array([float(stats[n]["std"]) for n in field_names], dtype=np.float32)
            except KeyError as e:
                logger.warning("stats.json missing entry for %s; skipping normalization", e)
            else:
                self._norm_mean_np = means
                self._norm_std_np = stds
                self._norm_mean_t = torch.from_numpy(means).reshape(1, -1, 1, 1)
                self._norm_std_t = torch.from_numpy(stds).reshape(1, -1, 1, 1)
                logger.info(
                    "%s normalization stats from %s: mean=%s std=%s",
                    dataset_name, stats_path,
                    np.array2string(means, precision=3),
                    np.array2string(stds, precision=3),
                )
        else:
            logger.info("%s: no stats.json at %s; training on raw fields", dataset_name, stats_path)

        split_run_ids = {
            split: _scan_run_ids(self.root / split)
            for split in ("train", "val", "test")
        }
        logger.info(
            "%s splits from %s: train=%d val=%d test=%d",
            dataset_name, self.root,
            len(split_run_ids["train"]),
            len(split_run_ids["val"]),
            len(split_run_ids["test"]),
        )

        def _make(split: str, full_trajectory: bool = False) -> _ZarrWindowDataset:
            return _ZarrWindowDataset(
                split_dir=self.root / split,
                run_ids=split_run_ids[split],
                n_steps_input=self.n_steps_input,
                n_steps_output=self.n_steps_output,
                time_steps_per_run=self.time_steps_per_run,
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
        """Map a normalized (B, C, H, W) tensor back to physical units.

        Used by ``Trainer.validation_loop`` / ``rollout_loop`` so the
        spherical metric suite is reported in the same units as the
        original Dedalus snapshots. Pass-through if no stats were found.
        """
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
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        batch_size: Optional[int] = None,
        distribute: bool = True,
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
                dataset,
                batch_size=bs,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
            )
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def rollout_val_dataloader(self) -> DataLoader:
        # Rollout eval runs on rank 0 only — keep batch_size=1 and skip
        # distribution so the trainer can simply gate the call on is_main.
        return self._loader(self.rollout_val_dataset, shuffle=False, batch_size=1, distribute=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

    def rollout_test_dataloader(self) -> DataLoader:
        return self._loader(self.rollout_test_dataset, shuffle=False, batch_size=1, distribute=False)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.root}>"
