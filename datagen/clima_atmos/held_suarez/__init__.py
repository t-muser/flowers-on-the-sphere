"""Held-Suarez producer driven by ClimaAtmos.jl.

Replaces the cubed-sphere MITgcm path (``datagen/mitgcm/held_suarez_cs/``)
which suffered from a finite-volume cube-corner instability at our target
resolution. ClimaAtmos uses a spectral-element method on the cubed sphere,
so the corner pathology does not arise there.

Layout
------
- ``run_hs.jl``    — Julia driver: reads a corner JSON, builds a TOML
  override dict, constructs ``ClimaAtmos.AtmosSimulation``, runs it, and
  writes NetCDF on a remapped lat-lon grid.
- ``postprocess.py`` — NetCDF → ``(time, level, lat, lon)`` Zarr via
  ``datagen.resample.write_latlon_zarr_3d`` (vertically interpolated to
  8 ERA5 hPa levels).
- ``scripts/{run,preflight,finalize_dataset}.py`` — CLI shape mirrors
  ``datagen/mitgcm/held_suarez/scripts/`` so SLURM array drivers are
  near-identical.
- ``env/Project.toml`` — pinned Julia environment.
- ``slurm/*.sbatch``  — phase-0/phase-1 jobs (smoke, scoping, preflight,
  production). All Julia and solver work happens on compute nodes.
"""
