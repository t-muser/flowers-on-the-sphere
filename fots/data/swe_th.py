"""On-the-fly shallow-water-equations datamodule.

Wraps ``torch_harmonics.examples.pde_dataset.PdeDataset``, which uses a
spectral ``ShallowWaterSolver`` to generate `(input, target)` pairs on
the fly. For the smoke run the dataset is small and lives entirely on
CPU.

The dataset yields tensors of shape ``(C, H, W)`` per sample; the
datamodule forwards them verbatim through ``DataLoader`` so a batch is
``(B, C, H, W)``. For ``n_steps_output > 1``, the target covers the
multi-step rollout baked into ``PdeDataset.nsteps`` — single-step here,
rollout variants reuse the same dataset for the smoke run.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from fots.data.datamodule import AbstractDataModule


@dataclass
class SweThMetadata:
    """Minimal dataset metadata surfaced to ``fots.train``."""

    dataset_name: str
    dim_in: int
    dim_out: int
    spatial_resolution: tuple[int, int]
    grid: str = "equiangular"
    field_names: tuple[str, ...] = ("geopotential", "vorticity", "divergence")
    n_spatial_dims: int = 2


class SweThDataModule(AbstractDataModule):
    """On-the-fly SWE datamodule backed by ``PdeDataset``.

    Parameters
    ----------
    nlat, nlon:
        Spatial grid resolution. Passed straight to ``PdeDataset(dims=(nlat, nlon))``.
    dt:
        Physical time step (seconds) between the input snapshot and the target.
    n_steps_input, n_steps_output:
        Number of steps in the input and output; for the smoke run both are 1.
        ``n_steps_output`` is recorded in metadata but the dataset itself
        advances by a single ``dt`` per sample (multi-step rollouts are
        handled by the trainer, not by PdeDataset).
    batch_size, num_workers:
        Torch DataLoader settings.
    train_size, val_size:
        Number of on-the-fly samples per epoch for train/val. Rollout and
        test loaders reuse the val dataset in this scaffolding pass.
    grid:
        "equiangular" or "legendre-gauss" — passed to ``PdeDataset``.
    normalize:
        Whether ``PdeDataset`` should z-score-normalize samples internally.
    seed:
        Torch generator seed for dataloader shuffling (train only).
    """

    # Shallow-water output has 3 fields: (vorticity, divergence, layer-depth)
    N_FIELDS: int = 3

    def __init__(
        self,
        nlat: int = 64,
        nlon: int = 128,
        dt: float = 3600.0,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        batch_size: int = 2,
        num_workers: int = 0,
        train_size: int = 16,
        val_size: int = 4,
        grid: str = "equiangular",
        normalize: bool = True,
        seed: int = 0,
        dataset_name: str = "swe_th",
    ):
        from torch_harmonics.examples.pde_dataset import PdeDataset

        self.dataset_name = dataset_name
        self.nlat = nlat
        self.nlon = nlon
        self.dt = dt
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.grid = grid

        nsteps = max(1, int(n_steps_output))
        self._train_ds = PdeDataset(
            dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid,
            initial_condition="random", num_examples=train_size,
            device=torch.device("cpu"), normalize=normalize,
        )
        self._val_ds = PdeDataset(
            dt=dt, nsteps=nsteps, dims=(nlat, nlon), grid=grid,
            initial_condition="random", num_examples=val_size,
            device=torch.device("cpu"), normalize=normalize,
        )
        self._seed = seed

    @property
    def metadata(self) -> SweThMetadata:
        return SweThMetadata(
            dataset_name=self.dataset_name,
            dim_in=self.N_FIELDS,
            dim_out=self.N_FIELDS,
            spatial_resolution=(self.nlat, self.nlon),
            grid=self.grid,
            n_spatial_dims=2,
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        gen = torch.Generator()
        gen.manual_seed(self._seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=gen if shuffle else None,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val_ds, shuffle=False)

    def rollout_val_dataloader(self) -> DataLoader:
        return self._loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._val_ds, shuffle=False)

    def rollout_test_dataloader(self) -> DataLoader:
        return self._loader(self._val_ds, shuffle=False)
