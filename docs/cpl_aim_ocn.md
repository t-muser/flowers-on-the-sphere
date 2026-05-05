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
over atmospheric CO₂ concentration `co2_ppm`, AIM area-mean solar
forcing multiplier `solar_scale`, ocean GM-Redi background diffusivity
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

| Axis           | Values                       | Count | Rendered meaning |
|----------------|------------------------------|-------|------------------|
| `co2_ppm`      | 280, 348, 560, 1120          |   4   | User-facing ppm; converted to AIM mole fraction in `aim_fixed_pCO2`. |
| `solar_scale`  | 0.97, 1.00, 1.03             |   3   | Multiplier on AIM's area-mean `SOLC=342 W/m²`. |
| `gm_kappa`     | 500, 1000, 2000 (m²/s)       |   3   | Written to `GM_background_K` and `GM_isopycK`. |
| `seed`         | 0, 1, 2, 3, 4                |   5   | Seed for the atmospheric theta perturbation. |

Row-major iteration: `co2_ppm` slowest, `seed` fastest. The first
five runs are the same physics replayed under five different IC
perturbations.

### Parameter units in MITgcm namelists

The JSON sweep files intentionally use readable physical units, but
MITgcm's AIM namelists expect two values in AIM-internal units. The
renderer performs these conversions in
[`datagen/cpl_aim_ocn/namelist.py`](../datagen/cpl_aim_ocn/namelist.py),
and the tests lock them down.

**CO₂.** `co2_ppm` is stored in parts per million in the JSON configs.
When `aim_select_pCO2=1`, AIM expects `aim_fixed_pCO2` as a dry-air
mole fraction, not ppm. The renderer writes

```text
aim_fixed_pCO2 = co2_ppm * 1e-6
```

Examples:

| JSON `co2_ppm` | Rendered `aim_fixed_pCO2` |
|---------------:|--------------------------:|
| 280            | 0.00028                   |
| 320            | 0.00032                   |
| 348            | 0.000348                  |
| 560            | 0.00056                   |
| 1120           | 0.00112                   |

Writing the ppm value directly would make AIM interpret, for example,
`280` as a mole fraction instead of 280 ppm, producing a nonsensical
radiation state.

**Solar forcing.** `solar_scale` multiplies AIM's area-mean incoming
solar constant. This is not the top-of-atmosphere total solar irradiance
near 1365 W/m². AIM's upstream default is

```text
SOLC = 342.0 W/m²
```

so the renderer writes

```text
SOLC = solar_scale * 342.0
```

Examples:

| JSON `solar_scale` | Rendered `SOLC` |
|-------------------:|----------------:|
| 0.97               | 331.74          |
| 1.00               | 342.00          |
| 1.03               | 352.26          |

This convention preserves the original AIM calibration. A previous
prototype used `solar_scale * 1365`, which over-forced the shortwave
radiation by about a factor of four.

**GM-Redi.** The ocean sweep value `gm_kappa` is already in MITgcm's
expected units (m²/s). It is written to both `GM_background_K` and
`GM_isopycK`. Upstream uses 800 m²/s; the production grid intentionally
uses 500, 1000, and 2000 m²/s.

### First-run and reference-control values

`run_0000.json` is not an upstream-reference control. It is the first
corner of the production tensor grid:

```json
{
  "co2_ppm": 280.0,
  "solar_scale": 0.97,
  "gm_kappa": 500.0,
  "seed": 0,
  "spinup_days": 30.0,
  "data_days": 365.0,
  "snapshot_interval_days": 1.0
}
```

Its rendered `atm/data.aimphys` contains:

```text
aim_select_pCO2=1,
aim_fixed_pCO2=0.00028,
SOLC=331.74,
```

A useful reference-like control point for debugging against the upstream
AIM/ocean calibration is:

```json
{
  "co2_ppm": 320.0,
  "solar_scale": 1.0,
  "gm_kappa": 800.0,
  "seed": 0
}
```

That point renders `aim_fixed_pCO2=0.00032`, `SOLC=342`, and
`GM_background_K=GM_isopycK=800`. It is not part of the 180-member
production grid unless added explicitly as a separate control run.

## Runtime design

Each ensemble member has two MITgcm launches in the same staged run
directory. Both launches use three MPI ranks in fixed MPMD order:

```text
rank_0 -> coupler
rank_1 -> ocean
rank_2 -> atmosphere
```

The order matters because each binary calls MITgcm's `setdir` helper
and then reads/writes inside `rank_<mpi_rank>/`.

### Phase 1: coupled spin-up

Default phase-1 settings:

| Setting | Value |
|---------|------:|
| Duration | 30 days |
| Atmosphere timestep | 450 s |
| Ocean timestep | 3600 s |
| Coupler exchange period | 3600 s |
| Diagnostics | disabled |
| Monitor output | daily |
| End-of-phase pickups | enabled |

The ocean starts from the pre-spun `global_ocean.cs32x15` pickup pair
staged under `inputs/ocn/`:

```text
pickup.0000072000
pickup.0000072000.meta
```

The atmosphere starts from the upstream `tRef` profile with a small
seeded potential-temperature perturbation written to
`rank_2/theta_pert.bin` and activated through:

```text
hydrogThetaFile='theta_pert.bin'
```

The perturbation is smooth on each cs32 face, stored as big-endian
float64 to match `readBinaryPrec=64`, and has default RMS amplitude
0.1 K. It is only used in phase 1; phase 2 restarts from the phase-1
pickup.

With the default 30-day spin-up, the final pickup suffixes are:

```text
atmosphere: 0000005760
ocean:      0000072720
```

The ocean suffix includes the parent pickup's initial iteration
72000 plus 720 additional one-hour spin-up steps.

### Phase 2: data collection

Default phase-2 settings:

| Setting | Value |
|---------|------:|
| Duration | 365 days |
| Restart | phase-1 atm and ocean pickups |
| Diagnostics | enabled |
| Snapshot cadence | 1 day |
| Further pickups | disabled |
| Final per-run artifact | `<out_dir>/run.zarr` |

Diagnostics are instantaneous snapshots. The namelist generator emits
one MDS stream per field so that `xmitgcm`'s cubed-sphere reader can
consume each file as a single-record stream.

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
            run_0000/
                mitgcm_run/
                    rank_0/         (coupler namelists, inputs, logs)
                    rank_1/         (ocean namelists, inputs, logs)
                    rank_2/         (atm namelists, inputs, logs)
                    phase1_std_outp
                    phase2_std_outp
                run.zarr            (raw cs32 zarr per run)
            run_0001/
            ...
            run_XXXX.FAILED         (only when scripts/run.py catches an error)
        train/
            run_0000.zarr -> ../runs/run_XXXX/run.zarr   (symlinks)
            ...
        val/
            ...
        test/
            ...
        splits.json                  random 80/10/10, seed=42
        stats.json                   per-channel z-score stats

## Stability and validation checks

The most important stability check is that rendered namelists preserve
AIM's internal units. Before running a long sweep, render one phase-1
namelist set and inspect the patched AIM fields:

```bash
uv run --project datagen python -c "
from datagen.cpl_aim_ocn.solver import RunConfig, SimulationParams, _phase_1_namelists
cfg = RunConfig()
sim = SimulationParams(280, 0.97, 500, 0)
aim = _phase_1_namelists(cfg, sim)['atm']['data.aimphys']
print('\n'.join(line for line in aim.splitlines()
                if 'aim_fixed_pCO2' in line or 'SOLC' in line))
"
```

Expected output for `run_0000`:

```text
aim_fixed_pCO2=0.00028,
SOLC=331.74,
```

For the reference-like control point `(320 ppm, solar_scale=1,
gm_kappa=800)`, the expected rendered values are:

```text
aim_fixed_pCO2=0.00032,
SOLC=342.,
GM_background_K=800.,
GM_isopycK=800.,
```

The targeted regression suite is:

```bash
uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_namelist.py -q
uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_solver.py -q
```

The broader package suite is:

```bash
uv run --project datagen pytest datagen/cpl_aim_ocn/tests -q
```

If a run still becomes unphysical after the unit checks pass, debug in
this order:

1. Compare `rank_2/data.aimphys` against the expected CO₂ and `SOLC`
   values above.
2. Check `rank_2/data` and `rank_1/data` for phase length, timestep,
   `pickupSuff`, and `pChkptFreq` values.
3. Run the 5-hour smoke test with `--source upstream`; if upstream
   succeeds and rendered namelists fail, diff the staged rank
   directories.
4. Temporarily remove the phase-1 `hydrogThetaFile` override so the
   atmosphere initializes only from upstream `tRef`. That separates
   atmospheric IC-layout problems from coupled-physics problems.
5. Inspect `rank_2/STDOUT.0000` monitor output before the crash; AIM
   temperature collapse usually appears there before post-processing
   sees invalid fields.

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
