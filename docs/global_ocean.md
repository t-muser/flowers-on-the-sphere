# MITgcm Global Ocean (cs32x15)

**One-line description.** A 243-member parametric ensemble of 100-year
MITgcm `global_ocean.cs32x15` integrations on the cubed sphere
(six 32×32 faces, 15 vertical levels), exported to a face-major Zarr
plus a ready-to-train lat/lon dataloader with vector rotation and
landmask-aware loss.

**Extended description.** Each trajectory runs the cs32x15 verification
case with realistic bathymetry, Levitus hydrography (3-D initial
condition and surface restoring), Trenberth wind stress, and shi/ncep
heat and freshwater fluxes. Runs warm-start from the
`pickup.0000072000` file shipped with the tutorial, which contains
roughly 200 model years of free spin-up. The Python driver stages the
tutorial code from the vendored MITgcm tree, patches only the runtime
settings needed for data generation (`nIter0`, `nTimeSteps`, output
cadence, checkpoint cadence, timestep values, restoring timescales, and
a small GM/Redi sweep hook), and reads the resulting MDS state dumps
into a face-major Zarr store. The production workflow is a
corner-based preflight followed by a 243-member sweep, both written
under a single `DATA_ROOT`. After the sweep, the dataset is finalised
with split + spinup-trim + landmask-aware z-score stats; the same root
becomes a HuggingFace-publishable directory and the source for the
fots dataloader.

## Associated resources

- **Tutorial.** MITgcm `verification/global_ocean.cs32x15`.
- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [MITgcm](../mitgcm) with the global-ocean
  customizations in
  [`datagen/mitgcm/global_ocean/`](../datagen/mitgcm/global_ocean).
- **Code.** Build, run, preflight, sweep, grid-extraction, split, and
  stats scripts live in the global-ocean package of this repository.
- **DataModule.** [`fots/data/global_ocean.py`](../fots/data/global_ocean.py).

## Physical framework

The benchmark is a primitive-equation ocean simulation on the cubed
sphere with non-linear free surface (`z*` coordinate, real fresh-water
flux) and vector-invariant momentum. The configuration includes:

- realistic cs32 bathymetry (`bathy_Hmin50.bin`)
- Levitus hydrographic initial conditions (`lev_T_cs_15k.bin`,
  `lev_S_cs_15k.bin`) and monthly surface restoring (`lev_surfT_cs_12m.bin`,
  `lev_surfS_cs_12m.bin`)
- Trenberth wind stress (`trenberth_taux.bin`, `trenberth_tauy.bin`)
- shi/ncep heat and freshwater fluxes (`shiQnet_cs32.bin`,
  `shiEmPR_cs32.bin`)
- GM/Redi mesoscale eddy parameterization
- GGL90 vertical mixing

The solver is configured through the tutorial namelist files in the
vendored MITgcm tree, with a small set of runtime overrides applied by
the Python driver.

## Data specifications

### Grid

- **Discretization.** Cubed-sphere grid with six 32×32 faces, supplied
  via `grid_cs32.face00{1..6}.bin` (symlinked from the
  `tutorial_held_suarez_cs` input directory at run time).
- **Compile-time domain decomposition.** `sNx=32, sNy=32, nSx=6, nSy=1,
  nPx=1, nPy=1` — single rank with one tile per cube face. Global
  output shape is `(Nr, 32, 192)` and reshapes trivially to
  `(Nr, 6, 32, 32)`.
- **Vertical coordinate.** 15 depth levels with layer thicknesses from
  `50 m` to `690 m`.

The exported benchmark is a horizontal slice through the 3-D ocean
state rather than the full 3-D volume. The selected levels are the same
ones used by the driver defaults.

### Temporal layout

- **Pickup.** Runs warm-start from `pickup.0000072000` (`nIter0=72000`),
  which provides roughly 200 model years of free spin-up.
- **Production run.** `36000` time steps on a 1-day model clock, i.e.
  `100` model years of post-pickup integration.
- **Snapshot cadence.** `30` simulated days by default for production
  runs.
- **Preflight.** `365` time steps with `30`-day snapshots (override via
  `PREFLIGHT_TIMESTEPS` and `PREFLIGHT_SNAPSHOT_DAYS` env vars on the
  sbatch line). The 365-step horizon is chosen to expose the slow
  vertical-CFL buildup that the original 30-step preflight could not
  see — see "Numerical-stability story" below.
- **Spinup trim.** During finalisation, the first `12` snapshots
  (≈1 model year of parameter-change adjustment transient) are dropped
  and the `time` coord is rebased so `time[0] == 0`. After trim each
  trajectory contains 1189 snapshots covering ~97.6 model years.

### Available fields

The exported Zarr stores the following 2-D (per-face) slices:

| Field name | Units  | Mask     | Description |
| ---        | ---    | ---      | --- |
| `theta_k1` | `degC` | `mask_k1`| Potential temperature at tracer level 1 |
| `salt_k1`  | `psu`  | `mask_k1`| Salinity at tracer level 1 |
| `u_k2`     | `m/s`  | `mask_k2`| Face-x velocity at velocity level 2 (rotated to east in the dataloader) |
| `v_k2`     | `m/s`  | `mask_k2`| Face-y velocity at velocity level 2 (rotated to north in the dataloader) |
| `eta`      | `m`    | `mask_eta`| Sea surface height |

All fields are stored as `float32`. Land cells are filled with `0.0`.
Per-cell longitudes (`xc`) and latitudes (`yc`) are stored as
`(face, y, x)` coordinate arrays alongside the data.

### Storage format and on-disk layout

Per-run Zarr stores are written at
`<DATA_ROOT>/sweep/run_NNNN/run.zarr` during generation; the native
layout is `(time, field, face, y, x)` with `face=6` and `y=x=32`.
After finalisation, the run zarrs are flattened and split into
`train/`, `val/`, `test/` subdirectories. The published-form root
looks like:

```
<DATA_ROOT>/global-ocean/
    manifest.json           243 runs × {run_id, params, hash}
    splits.json             random 80/10/10, seed=42 (counts: 194/24/25)
    stats.json              landmask-aware per-field z-score (see below)
    grid.zarr/              static grid info shared by all runs
    train/  run_NNNN.zarr   194 trajectories × 1189 snapshots
    val/    run_NNNN.zarr   24
    test/   run_NNNN.zarr   25
```

The default `DATA_ROOT` for production is
`/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean`
(group archive, feeds HuggingFace publication).

### grid.zarr (static)

A small companion Zarr at the dataset root holds the static grid info
that is identical across all 243 members:

| Variable    | Shape         | Dtype | Description |
| ---         | ---           | ---   | --- |
| `xc`, `yc`  | `(6, 32, 32)` | f32   | Cell-centre longitude / latitude (degrees) |
| `depth`     | `(6, 32, 32)` | f32   | Bathymetry, m, 0 = land |
| `hfac_c_k1` | `(6, 32, 32)` | f32   | Tracer-cell open fraction at k=1 |
| `hfac_w_k2`, `hfac_s_k2` | `(6, 32, 32)` | f32 | Velocity-cell open fractions at k=2 |
| `mask_k1`   | `(6, 32, 32)` | bool  | True ⇔ ocean (`hfac_c_k1 > 0`) |
| `mask_k2`   | `(6, 32, 32)` | bool  | True ⇔ ocean (`hfac_w_k2 > 0` and `hfac_s_k2 > 0`) |
| `mask_eta`  | `(6, 32, 32)` | bool  | True ⇔ column has any water at any level |
| `angle_cs`  | `(6, 32, 32)` | f32   | `cos(α)` of face → east/north rotation |
| `angle_sn`  | `(6, 32, 32)` | f32   | `sin(α)` of the same rotation |

Vector rotation convention: `u_east = angle_cs * u - angle_sn * v`,
`v_north = angle_sn * u + angle_cs * v`.

Why three masks: `theta_k1`/`salt_k1` live at tracer level 1 (use
`mask_k1`); `u_k2`/`v_k2` live at velocity level 2 on staggered W/S
faces (use `mask_k2`); `eta` lives wherever a column has any water
(`mask_eta`, slightly larger than `mask_k1` because some columns have
deep water but a dry surface partial cell). A single global mask is
wrong — applying `mask_k1` to `eta` drops valid SSH cells.

## Initial and boundary conditions

The cs32x15 verification case ships its own input files in
`mitgcm/verification/global_ocean.cs32x15/input/`. The Python driver
symlinks every binary file from that directory plus the six
`grid_cs32.face00?.bin` files into the run directory, then writes
patched `data`, `data.gmredi`, `data.pkg`, and the static `eedata`
namelist on top.

The default output levels are:

- tracer level `1` for `theta_k1` and `salt_k1`
- velocity level `2` for `u_k2` and `v_k2`

Those defaults are chosen to match the tutorial diagnostics and to give
near-surface fields with clear large-scale structure.

## Parameter space

The current sweep varies five physically meaningful ocean controls
around the tutorial defaults:

| Parameter | Symbol | Values | Count |
| --- | --- | --- | --- |
| GM/Redi background diffusivity | `gm_background_k` | `250, 1000, 2500` | 3 |
| Horizontal viscosity | `visc_ah` | `1.5e5, 3e5, 5e5` | 3 |
| Vertical diffusivity | `diff_kr` | `1e-5, 3e-5, 1e-4` | 3 |
| Surface temperature restoring | `tau_theta_relax_days` | `30, 60, 120` | 3 |
| Surface salinity restoring | `tau_salt_relax_days` | `90, 180, 360` | 3 |

Held fixed:

- `n_iter0 = 72000` (warm-start from the shipped pickup)
- `n_timesteps = 36000` for the regular run
- `delta_t_mom = 1800 s`, `delta_t_tracer = delta_t_clock = delta_t_freesurf = 86400 s`
- `tracer_level = 1`
- `velocity_level = 2`

The current sweep is a `3^5 = 243` run tensor product and is designed
to span weak, standard, and strong mixing / restoring regimes rather
than only tiny perturbations around the tutorial settings.

## Numerical-stability story

The original `visc_ah` grid was `(2e5, 5e5, 1e6)`. At `visc_ah = 1e6`
the model crosses CFL=1 around model month 8: vertical CFL accumulates
slowly during the parameter-change adjustment to >1 by timestep ~240,
then the run blows up via `S/R ALL_PROC_DIE` within a few more
timesteps. 16/32 preflight corners and 81 of the first 243 production
runs failed this way before we caught it.

Two facts mattered:

- The original 30-step preflight was blind to the failure (CFL stays
  under 0.07 through timestep ~50). The preflight now defaults to
  365 timesteps, which puts the divergence point comfortably inside
  the test horizon.
- `visc_ah = 1e6` was the only common factor: every failed corner had
  it set, none of the 16 corners with smaller viscosity blew up.
  The grid was tightened to `(1.5e5, 3e5, 5e5)`, all below the
  unstable boundary; the 365-step preflight on the new grid showed
  `max advcfl_wvel ≈ 0.094`, and the rerun production sweep finished
  243/243 cleanly.

This is documented in case anyone is tempted to widen the viscosity
range later. If you do, run the preflight at >365 timesteps first.

## Computational details

- **Per-run wall.** ~25–30 minutes of single-CPU wall on scicore for
  a 100-year production run (36 000 timesteps × ~0.04 s/step plus I/O
  for 1200 snapshots). 32 preflight corners at 365 timesteps land in
  ~5 minutes total when the scheduler doesn't throttle.
- **Total compute.** With no array throttle, scicore parallelises the
  243-member sweep onto the available pool; full sweep wall-time was
  ~30 min on a quiet day. Storage: ~93 MB per run.zarr × 243 ≈ 22 GB.
  The per-run solver-artifact dirs (`<DATA_ROOT>/sweep/run_NNNN/`) hold
  another ~3 GB each (raw MDS state); these are kept around for
  debugging and grid extraction but are *not* part of the published
  dataset.
- **Solver precision.** MITgcm runs in double precision internally;
  exported Zarr stores are `float32`.
- **Cluster setup.** The provided scripts assume the `foss/2024a` and
  `HDF5/1.14.5-gompi-2024a` module stack on sciCORE.

## End-to-end data flow

All commands assume the repo root and a SLURM cluster (sciCORE).
`DATA_ROOT` should point at a writable shared location; the production
default is `/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean`.

```bash
# Login node: sync Python deps for both projects
uv sync --project datagen
uv sync

# 1. Compile the MITgcm executable for the cs32x15 case (~1 min wall).
sbatch datagen/mitgcm/global_ocean/slurm/build.sbatch

# 2. Emit the 243 sweep configs and the 32 preflight corner configs.
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.generate_sweep \
    --out datagen/mitgcm/global_ocean/configs
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.preflight generate \
    --out datagen/mitgcm/global_ocean/configs/preflight

# 3. 32-corner preflight (365 timesteps each).
sbatch --export=ALL,DATA_ROOT=$DATA_ROOT \
    datagen/mitgcm/global_ocean/slurm/preflight_array.sbatch

# 4. 243-member production sweep.
sbatch --export=ALL,DATA_ROOT=$DATA_ROOT \
    datagen/mitgcm/global_ocean/slurm/sweep_array.sbatch

# 5. Extract the static grid (landmask, depth, AngleCS/SN) once.
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.extract_grid \
    --run-dir $DATA_ROOT/sweep/run_0000/global_ocean_run \
    --out $DATA_ROOT/grid.zarr

# 6. Random 80/10/10 split + bake spinup-trim into the train/val/test zarrs.
cp datagen/mitgcm/global_ocean/configs/manifest.json $DATA_ROOT/manifest.json
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.generate_split \
    --manifest $DATA_ROOT/manifest.json --out $DATA_ROOT/splits.json \
    --strategy random --seed 42
# Move the per-run zarrs into a flat 'processed/' dir, then split-with-trim.
# (When fresh from the sweep, use --src-dir sweep --nested instead, the
# trim flag still applies.)
uv run --project datagen python datagen/scripts/reorganize_splits.py \
    --root $DATA_ROOT --src-dir processed --trim-first 12

# 7. Per-field, ocean-only z-score stats.
uv run --project datagen python datagen/scripts/compute_stats.py \
    --root $DATA_ROOT --var-name data \
    --fields theta_k1 salt_k1 u_k2 v_k2 eta \
    --mask-zarr $DATA_ROOT/grid.zarr \
    --field-mask theta_k1=mask_k1 --field-mask salt_k1=mask_k1 \
    --field-mask u_k2=mask_k2  --field-mask v_k2=mask_k2 \
    --field-mask eta=mask_eta
```

For interactive inspection, the notebook
[`notebooks/mitgcm_global_ocean_visualize.ipynb`](../notebooks/mitgcm_global_ocean_visualize.ipynb)
opens any `run.zarr` (override the path with the `GLOBAL_OCEAN_ZARR`
env var) and shows the six-face snapshot, cube-unfolded `eta`, and
basin-mean diagnostics.

### Why the masked stats matter

Without the landmask, salinity stats absorb the 28% land cells stored
as `0.0` and report `salt_k1.mean ≈ 25 psu` (real ocean is ~34.7 psu);
the standard deviation is similarly inflated. Z-score normalisation on
those biased numbers leaves real ocean salinities at non-zero
"normalised" values and pulls land cells to large negative values,
turning every coastline into a step gradient that the model has to
fit. The masked pass restores physically meaningful per-field
statistics:

| Field | Masked mean | Masked std |
| --- | --- | --- |
| `theta_k1` | 18.04 °C | 9.54 |
| `salt_k1`  | 34.70 psu | 1.51 |
| `u_k2`     | -7.2e-04 m/s | 3.20e-02 |
| `v_k2`     | 2.3e-05 m/s | 3.43e-02 |
| `eta`      | 0.012 m | 0.575 |

## Training

Drop-in via Hydra:

```bash
uv run python -m fots.train data=global_ocean model=zinnia_v5
```

The config at [`configs/data/global_ocean.yaml`](../configs/data/global_ocean.yaml)
points `DATA_ROOT` at the dataset root and instantiates
`fots.data.global_ocean.GlobalOceanDataModule`.

### What the dataloader does

For each window it samples (1189 valid start positions per
trajectory, after the spinup trim):

1. Slice `(T_in + T_out, 5, 6, 32, 32)` from the run zarr.
2. Rotate `(u_k2, v_k2)` from face-aligned to geographic
   `(east, north)` using `angle_cs/sn` from `grid.zarr`. This is a
   per-cell linear combine on the native cs grid; doing it before
   the scalar regrid is correct in the limit where the source
   neighbourhood spans a single face.
3. Build (once at module init) a cs32 → `(64, 128)` lat/lon
   regrid via `datagen.cpl_aim_ocn.regrid.build_weights` — k-NN /
   IDW with `k=4` neighbours by default. The same weights apply to
   every dynamic channel and to the static depth + masks.
4. Apply z-score normalisation per field, using the
   landmask-aware `stats.json`. Land cells are zeroed both before
   and after normalisation so the model sees a clean
   signal-on-mask layout (no residual offsets from non-zero
   mean values).
5. Concatenate the static depth channel
   (`log10(max(depth_ll, 1)) / 4`, broadcast across timesteps) and
   per-timestep `sin/cos(2π · day-of-year / 365)`.
6. Return a dict per sample:

   ```python
   {
       "input_fields":  Tensor(T_in,  8, 64, 128),  # 5 dyn + 1 depth + 2 doy
       "output_fields": Tensor(T_out, 5, 64, 128),  # dyn only
       "valid_mask":    Tensor(5, 64, 128),         # per-channel ocean mask
   }
   ```

### Day-of-year and seasonality

The ocean is non-autonomous: `lev_surfT_cs_12m.bin`,
`trenberth_taux/tauy.bin`, and the heat/freshwater fluxes carry an
annual cycle, so the same instantaneous state can map to *different*
30-day futures depending on hidden seasonal phase. With only a 4-step
30-day input history the model cannot infer phase from the dynamic
fields alone; the `(sin, cos)` doy channels supply it explicitly.

The trim rebases `time` to start at 0 inside each trajectory; the
absolute pickup epoch is not exposed by MITgcm output, so the doy
channels treat the trajectory start as Jan 1. All 243 trajectories
share that pickup, so the assumption is consistent — if it turns out
to be off, the fix is one constant offset.

### Loss masking

The trainer plumbs `valid_mask` through to
[`fots.metrics.LatitudeWeightedMSELoss`](../fots/metrics.py) and
`compute_loss_metrics`. With the mask present, the loss is

```
loss = ((pred - target)^2 * cos(lat) * mask).sum()
       / (cos(lat) * mask).sum()
```

per channel, then averaged over channels. The same per-channel mask
weighting is applied to all spherically-weighted metrics
(`mse_sphere`, `vrmse`, `nrmse`, `pearson`, ...) so reported numbers
reflect ocean-only error. `linf` and `rel_l2` are invariant to
land=0 (residuals are zero on land cells) and need no extra masking.

The mask kwarg is **optional** at the loss/metrics layer — datasets
that don't supply `valid_mask` (e.g. `cpl_aim_ocn`, `galewsky-sw`)
keep their existing unweighted-mean behaviour.

### Channel layout

Input (8 channels):

| Idx | Source | Notes |
| --- | --- | --- |
| 0–4 | `theta_k1`, `salt_k1`, `u_east`, `v_north`, `eta` | z-scored, land-zeroed |
| 5   | `depth` | static, `log10(max(d, 1)) / 4` ≈ [0, 1] |
| 6   | `sin(2π·doy/365)` | constant in space, varies per timestep |
| 7   | `cos(2π·doy/365)` | constant in space, varies per timestep |

Output (5 channels): the dynamic fields only, in the same z-scored
units. The trainer's `denormalize_fn` undoes the z-score (it does
*not* unscale depth or doy — those aren't predicted).

## Provenance

cs32x15 verification files come from MITgcm
(commit recorded in `<DATA_ROOT>/sweep/run_0000/global_ocean_run/`'s
`build_info.json`). The cs32 grid binaries
`grid_cs32.face00{1..6}.bin` are pulled from the
`tutorial_held_suarez_cs/input/` sibling directory and shared with the
held-suarez dataset; do not delete them.
