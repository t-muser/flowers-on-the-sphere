# MITgcm Global Ocean 3-D (cs32x15)

**One-line description.** A 243-member parametric ensemble of 100-year
MITgcm `global_ocean.cs32x15` integrations on the cubed sphere
(six 32×32 faces, **all 15 vertical levels**), exported to a
per-variable Zarr plus a ready-to-train lat/lon dataloader that
folds depth into the channel axis.

**Relation to the 2-D dataset.** This is the volumetric companion to
[`docs/global_ocean.md`](global_ocean.md). The MITgcm runs are
identical (same 3⁵ = 243 sweep, same pickup, same timestepping, same
duration); the difference is purely in what gets exported. The 2-D
variant keeps a single tracer level (`k=1`) and a single velocity
level (`k=2`); the 3-D variant keeps every level. **Refer to the 2-D
doc for the physical framework, parameter space, and numerical
stability story** — those are unchanged.

## Associated resources

- **Data generator.** Till Muser (University of Basel), 2026.
- **Generation software.** [MITgcm](../mitgcm) with the global-ocean
  customizations in
  [`datagen/mitgcm/global_ocean/`](../datagen/mitgcm/global_ocean).
- **Solver entry points.** `solver.py` (3-D extract via `cfg.levels`),
  `scripts/run.py` and `scripts/preflight.py` (`--levels all` flag),
  `scripts/extract_grid.py` (per-level masks).
- **DataModule.** [`fots/data/global_ocean_3d.py`](../fots/data/global_ocean_3d.py).
- **Hydra config.** [`configs/data/global_ocean_3d.yaml`](../configs/data/global_ocean_3d.yaml).
- **Tests.** `datagen/mitgcm/global_ocean/tests/test_solver.py`
  (`extract_global_ocean_fields_3d`, `write_cubed_sphere_zarr_3d`),
  `fots/data/tests/test_global_ocean.py`
  (`apply_dynamic_3d` shape + rotation linearity).

## Data specifications

### Storage format and on-disk layout

Per-run Zarr stores at `<DATA_ROOT>/sweep/run_NNNN/run.zarr`. Unlike
the 2-D variant (single stacked `data` variable), the 3-D layout is
**per-variable**:

| Variable | Dims                              | Dtype | Description                          |
| ---      | ---                               | ---   | ---                                  |
| `theta`  | `(time, level, face, y, x)`       | f32   | Potential temperature, °C            |
| `salt`   | `(time, level, face, y, x)`       | f32   | Salinity, psu                        |
| `u`      | `(time, level, face, y, x)`       | f32   | Face-x velocity, m/s                 |
| `v`      | `(time, level, face, y, x)`       | f32   | Face-y velocity, m/s                 |
| `eta`    | `(time, face, y, x)`              | f32   | Sea-surface height, m                |

Coordinates: `time` (s), `level` (1..15, 1-indexed), `depth(level)`
(cell-centre depth in m), `xc(face,y,x)` / `yc(face,y,x)` (deg).
Land cells are filled with `0.0`.

After finalisation:

```
<DATA_ROOT>/global-ocean-3D/
    manifest.json           243 runs × {run_id, params, hash}
    splits.json             random 80/10/10, seed=42 (counts: 194/24/25)
    stats.json              per-channel landmask-aware z-score (61 entries)
    grid.zarr/              static grid info, 2-D + 3-D masks
    train/  run_NNNN.zarr   194 trajectories × 1189 snapshots × 15 levels
    val/    run_NNNN.zarr   24
    test/   run_NNNN.zarr   25
```

Storage footprint: each run.zarr is ≈350 MB on disk. Full dataset is
≈85 GB after the 12-snapshot spin-up trim.

The default `DATA_ROOT/global-ocean-3D/` lives in the group archive
at `/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/`.

### grid.zarr (static)

In addition to the surface-only fields shared with the 2-D dataset
(`xc`, `yc`, `depth`, `mask_k1`, `mask_k2`, `mask_eta`, `angle_cs`,
`angle_sn`), the 3-D `grid.zarr` adds per-level masks:

| Variable    | Shape                | Dtype | Description |
| ---         | ---                  | ---   | --- |
| `mask_c_3d` | `(15, 6, 32, 32)`    | bool  | Tracer-cell wet mask at every level (`hfac_c > 0`) |
| `mask_w_3d` | `(15, 6, 32, 32)`    | bool  | Velocity-cell wet mask at every level (`hfac_w > 0` AND `hfac_s > 0`) |
| `hfac_c`    | `(15, 6, 32, 32)`    | f32   | Tracer-cell open fraction at every level |

Ocean fraction by level (from a representative production grid):

```
k= 1:  72%        k= 6: 66%        k=11: 60%
k= 2:  70%        k= 7: 65%        k=12: 58%
k= 3:  69%        k= 8: 64%        k=13: 52%
k= 4:  67%        k= 9: 63%        k=14: 42%
k= 5:  67%        k=10: 62%        k=15: 26%
```

The decline reflects realistic bathymetry: deep basins occupy a
shrinking fraction of the global area as you go down.

## End-to-end data flow

The pipeline reuses every step from the 2-D version with only the
solver-output stage diverging.

```bash
# 1. Build the MITgcm binary (one-shot, identical to 2-D).
sbatch datagen/mitgcm/global_ocean/slurm/build.sbatch

# 2. Cheap 32-corner preflight on the 3-D path.
sbatch datagen/mitgcm/global_ocean/slurm/preflight_array_3d.sbatch

# 3. Production sweep (243 jobs, --mem=16G is critical for the 3-D
#    Python read/write phase; 4G default OOMs).
sbatch datagen/mitgcm/global_ocean/slurm/sweep_array_3d.sbatch

# 4. Extract the static grid (now also writes per-level masks).
DATA_ROOT=/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean-3D
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.extract_grid \
    --run-dir $DATA_ROOT/sweep/run_0000/global_ocean_run \
    --out $DATA_ROOT/grid.zarr

# 5. Random 80/10/10 split, same seed as 2-D for cross-comparability.
cp datagen/mitgcm/global_ocean/configs/manifest.json $DATA_ROOT/manifest.json
uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.generate_split \
    --manifest $DATA_ROOT/manifest.json --out $DATA_ROOT/splits.json \
    --strategy random --seed 42

# 6. Reorganize zarrs with spinup trim. Use --workers > 1 because
#    each move + trim is ~1 GB of I/O; sequential takes hours.
sbatch datagen/mitgcm/global_ocean/slurm/reorganize_3d.sbatch

# 7. Per-channel ocean-only z-score stats (theta_k01..theta_k15,
#    salt_k01..salt_k15, u_k01..u_k15, v_k01..v_k15, eta).
sbatch datagen/mitgcm/global_ocean/slurm/stats_3d.sbatch
```

## DataModule and channel layout

The `GlobalOcean3DDataModule` mirrors the 2-D module's surface but
**folds the level axis into the channel axis** at regrid time so
downstream models see a flat `(T, C, H, W)` stack.

**Default level subset (6 log-spaced levels):**

| k  | Depth   | What lives there                  |
| -- | ---     | ---                               |
| 1  | 25 m    | SST/SSS, mixed layer, eddy KE peak |
| 3  | 170 m   | Mixed-layer base / upper thermocline |
| 5  | 455 m   | Subtropical mode water            |
| 7  | 935 m   | AAIW core / intermediate water    |
| 10 | 2030 m  | NADW upper                        |
| 13 | 3575 m  | Bottom-water signal               |

Each captures roughly half the remaining vertical variance — no
near-redundant entries, no near-empty entries. Channels with this
default:

```
dynamic channels (25):
    theta_k01, theta_k03, theta_k05, theta_k07, theta_k10, theta_k13
    salt_k01, ...                                              (k01..k13)
    u_k01, ...     (geographic east, rotated on cs grid)
    v_k01, ...     (geographic north)
    eta            (sea-surface height)
static / time conditioning (3):
    depth                  (1)    log10-scaled, 0 over land
    sin(doy), cos(doy)     (2)    day-of-year phase
total input channels: 28
total output channels: 25
```

Override `data.levels` in the Hydra config to change the subset:
`null` → all 15 levels (61 dyn channels); `[1, 4, 8, 13]` → minimal
4-level (17 dyn); `[1, 2, 3, 5, 7, 9, 11, 14]` → surface-dense
8-level (33 dyn); `[1]` → surface only (≈ 2-D dataset).

Each batch dict is identical in shape to the 2-D module:

```python
{
    "input_fields":  Tensor(B, T_in,  C_in,  H=64, W=128),
    "output_fields": Tensor(B, T_out, C_out, H=64, W=128),
    "valid_mask":    Tensor(B,        C_out, H=64, W=128),
}
```

with `C_in = 4 * Nlevel + 1 + N_STATIC_INPUT + N_TIME_INPUT` and
`C_out = 4 * Nlevel + 1`.

Per-channel masks: `theta_*` and `salt_*` use `mask_c_3d` at the
corresponding level; `u_*` and `v_*` use `mask_w_3d`; `eta` uses
`mask_eta` (column has any water).

## Training

Use the dedicated data config:

```bash
uv run python -m fots.train_lit data=global_ocean_3d \
    model=zinnia_v5_small \
    optim.learning_rate=5e-4 \
    +trainer.max_epochs=100
```

**Models that hardcode `dim_in=8` / `dim_out=5` (the 2-D defaults) need
to be reconfigured for `dim_in=64` / `dim_out=61` when training on the
3-D dataset.** This is the only invasive consumer-side change. Common
ways to handle 61 channels:

- Treat all 61 as independent channels (simplest; works with any 2-D
  network — the 64×128 spatial size is unchanged).
- Reshape the 4 × 15 = 60 fluid channels back to a `(level, ...)`
  tensor inside the model and apply 1-D depth convolutions or
  attention over level (this is where the channel-stacked layout pays
  for itself: the model decides the level treatment, not the loader).

The valid_mask broadcast applies to the loss as in the 2-D pipeline:

```python
loss = LatitudeWeightedMSELoss(...)
out = loss(y_pred, y, mask=batch["valid_mask"])
```

## Land-bleed handling

`stats.json` is computed from the cs32 wet-cell values, where deep-ocean
salinity is very uniform (`salt_k15`: 34.72 ± 0.042 psu). A naive
cs→lat/lon IDW regrid mixes the 0.0 land-fill into wet cells across
continental margins, picking up a ~0.5 psu bias. With the deep-level
variance two orders of magnitude tighter than the surface, this bleed
becomes a 100× z-score blow-up at the deepest tracer levels — a single
batch's normalised `salt_k15` slot would otherwise show mean≈-68,
std≈100 instead of ≈N(0,1).

The DataModule defaults to `impute_land=True`, which replaces every
land cell with the per-variable per-level wet-cell mean *before* the
IDW regrid. The kernel then mixes wet values with a neutral wet-mean
instead of 0; the bool mask zeros land cells back out post-regrid.
Smoke test with imputation on:

```
salt_k15:  μ=-0.08  σ=0.87   ← was -68 / 108 without imputation
theta_k15: μ=-0.11  σ=0.94
eta:       μ=-0.33  σ=1.07
```

Set `data.impute_land=false` in the Hydra config to disable (e.g. when
A/B-testing the artifact). The 2-D dataset is unaffected either way:
its surface-level variance is wide enough that the bleed normalises to
small values.

## Subsetting depth levels

The on-disk dataset keeps all 15 levels; the DataModule selects a
subset at load time via `data.levels`. The deepest levels are nearly
homogeneous (`salt_k15` σ=0.04 psu, `theta_k15` σ=0.48 K) — they
contribute parameter count without much information. Default to the
6-level log-spaced subset described above; drop further or add more
depending on the model.

The variance scaling that motivates the default:

```
theta std:  k1=9.5  k3=7.0  k5=4.0  k7=2.6  k10=1.0  k13=0.6
salt  std:  k1=1.5  k3=0.89 k5=0.74 k7=0.69 k10=0.30 k13=0.06
```

`field_names`, `valid_mask`, `dim_out`, and the regrid shape all adjust
automatically to the chosen subset; `stats.json` is keyed by channel
name so the right normalisation parameters are picked up regardless.
Recommended subsets:

| Levels                       | Channels (dyn) | Notes                              |
| ---                          | ---            | ---                                |
| `[1, 3, 5, 7, 10, 13]`       | 25             | **default** — log-spaced, full column |
| `[1, 4, 8, 13]`              | 17             | minimal, fast iteration            |
| `[1, 2, 3, 5, 7, 9, 11, 14]` | 33             | surface-dense, more vertical detail|
| `null` (all 15)              | 61             | full vertical resolution           |
| `[1]`                        | 5              | surface only (≈ 2-D dataset)       |

## Provenance and version

- **MITgcm:** vendored at `mitgcm/` (commit recorded in repo).
- **Solver code:** `datagen/mitgcm/global_ocean/solver.py` (the 3-D
  branch is gated on `cfg.levels` being a non-empty tuple of
  1-indexed level integers).
- **Channel-stacked regrid:** `datagen/mitgcm/global_ocean/regrid.py`
  → `apply_dynamic_3d` and `field_masks_3d_ll`.
- **Production memory floor:** `--mem=16G`. The 4 GB per-CPU default
  on scicore OOM-kills the Python MDS-read phase; bump if you ever
  re-run on another cluster with similar defaults.
