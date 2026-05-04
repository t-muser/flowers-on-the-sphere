# Coupled AIM Atmosphere + Ocean GCM on the Cubed Sphere

**One-line description.** A 180-member parametric ensemble of
year-long, daily-snapshot integrations of MITgcm's *Atmospheric
Intermediate Model* (AIM, 5 σ-levels) coupled to a 15-level cs32
primitive-equation ocean with thermodynamic sea ice — three MPI
binaries running concurrently as one MPMD job and exchanging fluxes
through MITgcm's in-tree `atm_ocn_coupler` package.

**Extended description.** Each run is a single time-evolving climate
state: the AIM atmosphere (Molteni 2003 spectral physics, dry
dynamical core in σ-coordinates, simplified radiation, convection,
land-surface, and thermodynamic sea-ice schemes) is coupled every
ocean time-step (1 hour) to a free-surface, 15-level cs32 ocean using
GM-Redi mesoscale parameterisation. The ocean restarts from a
parent `global_ocean.cs32x15` spin-up pickup at iteration 72000;
the atmosphere cold-starts from prescribed climatology with a small
seed-controlled potential-temperature perturbation. Each ensemble
member runs **30 days of coupled spin-up** followed by **365 days of
data collection** at daily diagnostic cadence, producing a single
`run.zarr` per member. Ensemble diversity comes from a 4-axis grid
over atmospheric CO₂ concentration `co2_ppm`, total-solar-irradiance
multiplier `solar_scale`, ocean GM-Redi background diffusivity
`gm_kappa`, and an IC-perturbation seed.

The on-disk artifact is the **native cubed-sphere zarr** with separate
`atm_*` / `ocn_*` data variables on `(face=6, j=32, i=32)` (3-D atm
fields additionally on `Zsigma=5`). For models that expect a regular
lat/lon grid, the `fots.data.cpl_aim_ocn.CplAimOcnDataModule` regrids
each sample on the fly using a precomputed cKDTree.

## Associated resources

- **MITgcm verification.** [`verification/cpl_aim+ocn`](https://github.com/MITgcm/MITgcm/tree/master/verification/cpl_aim+ocn)
  — upstream coupled-model integration test that this dataset
  vendors and parameterises.
- **AIM atmosphere.** Molteni, F. (2003). *Atmospheric simulations
  using a GCM with simplified physical parametrizations. I:
  model climatology and variability in multi-decadal experiments.*
  Climate Dynamics, 20(2-3), 175-191.
- **Coupler.** MITgcm's in-tree `atm_ocn_coupler` package
  (`pkg/atm_ocn_coupler/`).
- **Data generator.** Till Muser (University of Basel), 2026.
- **Code.** [`datagen/cpl_aim_ocn/`](../datagen/cpl_aim_ocn).

## Configuration grid

180 runs total, tensor product of:

| Axis           | Values                       | Count |
|----------------|------------------------------|-------|
| `co2_ppm`      | 280, 348, 560, 1120          |   4   |
| `solar_scale`  | 0.97, 1.00, 1.03             |   3   |
| `gm_kappa`     | 500, 1000, 2000 (m²/s)       |   3   |
| `seed`         | 0, 1, 2, 3, 4                |   5   |

Row-major iteration: `co2_ppm` slowest, `seed` fastest. The first
five runs are the same physics replayed under five different IC
perturbations.

## Output channels (35)

After load-time regrid + flatten, every snapshot exposes 35 scalar
channels on a `(64, 128)` equiangular lat/lon grid. The canonical
order is in [`datagen/cpl_aim_ocn/channels.py`](../datagen/cpl_aim_ocn/channels.py):

**Atm 2-D (9):** `atm_TS`, `atm_QS`, `atm_PRECON`, `atm_PRECLS`,
`atm_WINDS`, `atm_UFLUX`, `atm_VFLUX`, `atm_SI_Fract`, `atm_SI_Thick`.

**Atm 3-D, σ-split (4 × 5 = 20):** `atm_UVEL_s{1..5}`,
`atm_VVEL_s{1..5}`, `atm_THETA_s{1..5}`, `atm_SALT_s{1..5}` (σ index
1 = top of atmosphere, 5 = surface, matching MITgcm's
pressure-coordinate convention).

**Ocn 2-D (6):** `ocn_THETA` (SST), `ocn_SALT` (SSS), `ocn_UVEL`
(surface u), `ocn_VVEL` (surface v), `ocn_ETAN` (SSH),
`ocn_MXLDEPTH` (mixed-layer depth).

The four ocean dynamical-core variables are sliced to surface
(`k=0`) inside the diagnostics namelist itself; `ETAN` and
`MXLDEPTH` are natively 2-D.

## End-to-end recipe

All commands assume the repo root and the SLURM cluster scicore.
`DATA_ROOT` defaults to `$SCRATCH/flowers-data`.

```bash
# 1. Compile the three coupled binaries (cpl, ocn, atm) and stage inputs.
sbatch datagen/cpl_aim_ocn/slurm/build.sbatch

# 2. End-to-end smoke test (5 simulated hours, ~1 minute wall-clock).
sbatch datagen/cpl_aim_ocn/slurm/smoke.sbatch

# 3. Generate the 180 per-run JSON configs (login node, fast).
uv run --project datagen \
    python -m datagen.cpl_aim_ocn.scripts.generate_sweep

# 4. Run the full sweep (180 array jobs × 3 MPI tasks each).
sbatch datagen/cpl_aim_ocn/slurm/sweep.sbatch

# 5. Assemble the published dataset (split + stats).
sbatch datagen/cpl_aim_ocn/slurm/finalize.sbatch
```

After step 5 the on-disk layout is::

    $DATA_ROOT/cpl_aim_ocn/
        runs/
            run_0000/run.zarr      (raw cs32 zarr per run)
            run_0001/run.zarr
            ...
        train/
            run_0000.zarr -> ../runs/run_XXXX/run.zarr   (symlinks)
            ...
        val/
            ...
        test/
            ...
        splits.json                  random 80/10/10, seed=42
        stats.json                   per-channel z-score stats

## Training

Use the dataset like any other in the repo:

```bash
uv run python -m fots.train data=cpl_aim_ocn model=zinnia_v5
```

The `CplAimOcnDataModule` builds cs32 → lat/lon regrid weights once
at instantiation (cKDTree on the 6 × 32 × 32 cell centres) and
applies them per sample during loading. Stats are loaded from
`stats.json` and applied as standard z-score normalization. The
returned tensor shape is `(B, T, 35, 64, 128)`.

## Provenance

Vendored upstream-MITgcm files (`code_atm/`, `code_ocn/`, `code_cpl/`,
`shared_code/`, `templates/*`, `build_*/genmake_local`) are
documented in
[`datagen/cpl_aim_ocn/PROVENANCE.md`](../datagen/cpl_aim_ocn/PROVENANCE.md)
including the upstream commit hash and the two intentional deviations
(`numlists=30` in `DIAGNOSTICS_SIZE.h` for both atm and ocn). Re-vendor
with the recipe at the bottom of that file when MITgcm bumps the AIM
or coupler packages.
