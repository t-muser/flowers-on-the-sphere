"""Probe HDF5 spherical datasets end-to-end in the pt-25.04 container.

For each dataset config:
  1. inspect one train .h5 file: groups, the t0/t1 fields, trajectory length,
     spatial resolution, component_names attrs,
  2. instantiate the NotWellDataModule via Hydra (same path fots.train uses),
  3. pull one batch from the train loader and print tensor shapes.

Run via scripts/probe_hdf5_modulus.sbatch (the_well + h5py installed into the
~/.local userbase first). Uses the train_4-to-1 window (n_steps_input=4,
n_steps_output=2) merged on top of each data config, matching real training.
"""
import sys

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import os

# Datasets to probe; override with PROBE_DATASETS="cahn_hilliard mickelin ...".
_DEFAULT = ["mickelin", "shock_caps"]
CONFIGS = {
    ds: f"configs/data/{ds}_hdf5.yaml"
    for ds in os.environ.get("PROBE_DATASETS", " ".join(_DEFAULT)).split()
}
TRAIN_CONFIG = "configs/train_4-to-1.yaml"


def _h5_tree(name, obj):
    indent = "  " * (name.count("/") + 1)
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}  shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{indent}{name}/")


def inspect_file(path):
    print(f"--- h5 structure: {path}")
    with h5py.File(path, "r") as f:
        print("  root attrs:", dict(f.attrs))
        f.visititems(_h5_tree)
        # Look for component_names on any field
        def _show_comp(name, obj):
            if isinstance(obj, h5py.Dataset) and "component_names" in obj.attrs:
                print(f"  {name}.component_names = {list(obj.attrs['component_names'])}")
        f.visititems(_show_comp)


def main():
    import glob, os

    for label, cfg_path in CONFIGS.items():
        print("=" * 70)
        print(f"DATASET: {label}  ({cfg_path})")
        print("=" * 70)

        data_cfg = OmegaConf.load(cfg_path)
        path = data_cfg.data.path
        train_files = sorted(glob.glob(os.path.join(path, "data", "train", "*.h5")))
        print(f"train files: {len(train_files)}")
        if not train_files:
            print("  !! no train files found, skipping")
            continue
        inspect_file(train_files[0])

        # Merge train config on top (gives n_steps_input=4, n_steps_output=2,
        # max_rollout_steps, world_size/rank defaults) exactly like fots.train.
        train_cfg = OmegaConf.load(TRAIN_CONFIG)
        merged = OmegaConf.merge(data_cfg, train_cfg)
        # Validate the experiment/wandb name resolution (post well_dataset_name
        # -> dataset_name rename): both must come out non-None.
        from fots.utils import get_experiment_name
        wandb_name = OmegaConf.select(merged, "data.dataset_name") or OmegaConf.select(
            merged, "data.well_dataset_name"
        )
        print("  data.dataset_name (wandb tag/group):", wandb_name)
        print("  experiment_name:", get_experiment_name(merged))
        assert wandb_name is not None, "dataset_name did not resolve — wandb tag would be empty"
        # Hydra instantiate of the datamodule (the part that fails fast if the
        # spec/loader disagree, or if the renamed kwarg no longer binds).
        print("--- instantiating NotWellDataModule via Hydra ---")
        dm = instantiate(merged.data, _convert_="all")
        md = dm.train_dataset.metadata
        print("  dataset_name:", md.dataset_name)
        print("  n_spatial_dims:", md.n_spatial_dims)
        print("  spatial_resolution:", md.spatial_resolution)
        print("  field_names:", dict(md.field_names))
        print("  scalar_names:", md.scalar_names)
        print("  boundary_condition_types:", md.boundary_condition_types)
        print("  n_files:", md.n_files)
        print("  train samples (len):", len(dm.train_dataset))

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        print("--- one train batch ---")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} {v}")
        print(f"[probe] {label} OK\n")


if __name__ == "__main__":
    main()
