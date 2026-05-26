# Held-Suarez (ClimaAtmos.jl) producer

This is the active path for the Held-Suarez ensemble. The cubed-sphere
MITgcm path (`datagen/mitgcm/held_suarez_cs/`) is superseded — see
`docs/held_suarez_cs.md` and that directory's `README.md` for context.

## The 30-second pitch

- **Solver:** ClimaAtmos.jl spectral-element method on the cubed sphere.
    The CliMA team maintains the HS config in their CI; we are not on
    our own.
- **Producer surface area:** small Julia driver (`run_hs.jl`, ~150
    lines) plus a Python post-processor that writes the same Zarr schema
    the MITgcm HS-3D dataset uses. The training-side `HeldSuarezDataModule`
    needs zero code changes.
- **Sweep:** 27 regimes (3 ClimaParams-native physical axes, 3 levels
    each) × 6 IC seeds = **162 trajectories**. Multiple seeds per
    regime give the spherical neural PDE solver benchmark enough
    independent samples per regime to estimate variance under the
    standard random 80/10/10 split used across the suite. The three
    physical axes hit ClimaParams TOML keys directly; the IC seed needs
    a minimal one-function patch to `DecayingProfile.jl`'s perturbation
    (stock HS uses a deterministic `sind(long)` bump). See "Parameter
    sweep" below.
- **Compute target:** `he24ze31` on a single GPU (L40s / A100 / RTX
    4090). The GPU smoke run gives ~5 h wall per 565-d corner; the
    full 162-corner sweep finishes in ~32 h wall with ~25 concurrent
    GPUs across partitions.

## Layout

```
datagen/clima_atmos/held_suarez/
├── run_hs.jl              ← Julia driver
├── postprocess.py         ← ClimaAtmos NetCDF → HS-3D Zarr
├── env/Project.toml       ← Pinned Julia env
├── configs/resolution/    ← he24ze31.yml
├── scripts/               ← Python orchestration (mirrors mitgcm/.../scripts/)
├── slurm/                 ← Install env, phase 0 smoke + scoping,
│                            preflight, production
└── tests/                 ← Schema + consumer compat + integration smoke
```

See `datagen/clima_atmos/held_suarez/README.md` for the full how-to.

## Parameter sweep

Three physical axes plus a seed axis. The physical axes are the
strongest diversity drivers in HS — each one takes the flow through
qualitatively distinct states rather than rescaling a single state.
The seed axis provides multiple independent realizations per regime
so that benchmark metrics get proper variance estimates.

| Axis | ClimaParams key (or transport) | Levels | Role |
|------|-----------------|--------|------|
| Ω       | `angular_velocity_planet_rotation`      | 3: 0.5, 1.0, 2.0 × Ω_Earth | Sets Rossby radius. Drives qualitatively distinct regimes: wide single jet (slow) → Earth-like single midlat jet → multiple narrower jets (fast). The strongest single diversity driver in dry-dycore land (Williams 1988; Mitchell & Vallis 2010). |
| ΔT_y    | `equator_pole_temperature_gradient_dry` | 3: 40, 60, 80 K            | Baroclinic forcing strength → jet speed, EKE. |
| Δθ_z    | `potential_temp_vertical_gradient`      | 3: 5, 10, 20 K             | Static stability → vertical mode structure, EKE/MKE partition. Wider than the HS-canonical 10 K → mirrors the MITgcm sweep's validated range. |
| seed    | `HS_IC_SEED` env var (patched `DecayingProfile`) | 6                | Independent realizations of each regime. |

3 × 3 × 3 × 6 = 162 trajectories.

**Design notes.**

- **Why not damping timescales.** Hardcoded in
    `src/parameterized_tendencies/radiation/held_suarez.jl` (the `k_a`,
    `k_s`, `k_f` constants) in v0.39 — varying them requires a source
    patch. They also mostly control *how fast* the flow is pulled
    toward equilibrium rather than *what* the equilibrium is.
- **Why the IC seed needs a patch.** ClimaAtmos v0.39's `DecayingProfile`
    setup applies a deterministic temperature bump
    (`0.1·sind(long)·mask(z<5km)`) — there is no RNG to seed. To get N
    independent realizations per regime we replace that bump with
    per-cell Gaussian noise of the same amplitude, seeded by hashing
    `(seed, lat, long, z)`. The patch lives at
    `datagen/clima_atmos/held_suarez/env/source_patch.diff`; it is
    bit-identical to stock when the `HS_IC_SEED` env var is unset.

## Trajectory length and sampling

Each of the 162 trajectories is a 565-day ClimaAtmos run, of which the
first **200 days are discarded as spin-up** and the remaining **365
days are emitted to Zarr at a 6-hourly cadence** (1460 snapshots per
trajectory, ~237k snapshots in the full dataset).

**Spin-up budget.** HS thermal equilibration from a perturbed
isothermal state takes ~50–100 days; the slower drift to a
statistically steady baroclinic state adds another ~50 days. 200 days
has comfortable margin and matches the conventional choice in the HS
literature. Phase 0.3 should confirm this on at least one corner by
plotting global-mean KE and EKE vs. time — both should plateau well
before day 200. If they plateau much earlier we can revisit, but the
200-day commit is what production runs against.

**Sampling cadence.** 6-hourly is chosen against the relevant
dynamical timescales:

| Timescale | At Earth Ω | Samples per timescale at 6h |
|-----------|-----------:|-----------------------------:|
| Baroclinic eddy lifetime | 5–7 d | 20–28 |
| Inertial period (midlat) | ~1 d  | 4 |
| Boundary-layer drag      | 1 d   | 4 |

This is the conventional weather/climate ML cadence (ERA5,
FourCastNet, GraphCast, SFNO benchmarks all use 6h native). It gives
enough samples per eddy lifetime for a solver to learn the dynamics
rather than just the climatology, while keeping snapshots decorrelated
enough to not be redundant.

**Ω-axis caveat.** At `2 × Ω_Earth` the midlatitude inertial period
halves to ~12h, putting 6h sampling at exactly the Nyquist rate for
inertial oscillations. In practice this is fine: HS at the
hydrostatic-ish ClimaAtmos configuration has weak large-scale
inertia–gravity wave activity, and the dynamics the solver should
learn are the geostrophically-balanced eddies, which evolve on
multi-day timescales regardless of Ω. We don't tighten cadence in the
fast-Ω corners — uniform 6h across all 162 trajectories.

**Storage envelope** (he24ze31, regridded to lat-lon, rough
order-of-magnitude — see note below on regrid resolution):

| Rate | Snapshots / trajectory | Total snapshots | Compressed (180×90 regrid) |
|------|-----------------------:|----------------:|---------------------------:|
| 3h   | 2920 | 473k | ~1.2 TB |
| **6h** | **1460** | **237k** | **~600 GB** |
| 12h  | 730  | 118k | ~300 GB |
| 24h  | 365  | 59k  | ~150 GB |

**Regrid target needs a decision.** The MITgcm cs32 schema implied
~180×90 lat-lon (~2° at the equator). At he24 the native spacing is
~36 km ≈ 0.33°, so a 180×90 regrid discards most of the resolution
gain. Candidates: 360×180 (~1°, ~4× the per-snapshot storage of the
table above) or 720×360 (~0.5°, ~16×). The 1° target is probably the
defensible compromise — matches the native resolution to within ~3×
and keeps total storage in the low-TB range at 6h. Pin during Phase 0.

**Cadence is a postprocess parameter, not a simulation parameter.**
The Julia driver writes ClimaAtmos's native diagnostic output;
`postprocess.py` decides what cadence to emit into the Zarr. If we
later want to revisit the cadence (denser for some study, sparser for
storage), we can regenerate from the raw ClimaAtmos output without
rerunning any simulations — provided the raw output is retained.
**Retention policy is therefore worth deciding at the same time as
cadence**: keep raw ClimaAtmos NetCDF for at least one full sweep, and
only delete after the final Zarr cadence is locked.

## Train/val/test split

Standard random seeded 80/10/10 split across all 162 trajectories,
matching the convention used by every other dataset in this benchmark
suite (130 / 16 / 16). Regimes and seeds are not stratified — with 6
seeds per regime the split will naturally include a mix of all 27
dynamical regimes in each partition.

## Phase 0 — go/no-go gate

| # | Check | Pass criterion |
|---|-------|----------------|
| 0.1 | Env install | `julia --project=env -e 'using ClimaAtmos'` succeeds |
| 0.2 | Stock smoke | `phase0_smoke.sbatch` completes 2 sim-days; `ta` ∈ [180, 320] K |
| 0.3 | Scoping     | `he24ze31` gives ≤ 8 h wall per 565 d on 1 L40s/A100/RTX4090 |
| 0.4 | Param override | Varying `equator_pole_temperature_gradient_dry` (40 vs 80) **and** planet Ω (0.5× vs 2× Earth) yields visibly different `ta` and zonal-wind fields |
| 0.5 | **Seed reproducibility** | Two runs with the same params and the same seed produce identical output; two runs with the same params and different seeds diverge into statistically distinct trajectories by t = 50 days |
| 0.6 | Postprocess | `HeldSuarezDataModule` consumes the Zarr; schema test passes |

Check 0.5 is load-bearing — the patched `DecayingProfile` hashes
`(seed, lat, long, z)` per cell, which gives bitwise-identical IC for
the same seed and statistically distinct IC across seeds. Phase 0.5
verifies that this property survives spin-up: same-seed runs stay
identical for one timestep, different-seed runs decorrelate by t≈50d.

Pure-Python checks are green today:

```
uv run --project datagen pytest \
    datagen/clima_atmos/held_suarez/tests/test_postprocess.py \
    datagen/clima_atmos/held_suarez/tests/test_param_override.py::TestCornerSchema \
    -v
```

The integration tests (Julia required) live in the same files and are
gated by `RUN_CLIMA_INTEGRATION=1`. Run them on a compute node after
`install_env.sbatch` finishes.

## What carries over from the MITgcm pipeline

Reused as-is:

- `fots/data/held_suarez.py` — the consumer DataModule.
- `datagen/resample.py::write_latlon_zarr_3d()` — canonical Zarr writer.
- `datagen/mitgcm/held_suarez/scripts/finalize_3d.py` — splits + stats
    (grid-agnostic; we wrap it via `scripts/finalize_dataset.py`).

Regenerated for ClimaAtmos:

- `generate_sweep.py` — different shape: 3 physical axes
    (Ω, ΔT_y, Δθ_z) + 6 IC seeds rather than the MITgcm grid's
    5-physical-axis Cartesian product. MITgcm version stays at
    `datagen/mitgcm/held_suarez/scripts/generate_sweep.py` for the
    superseded sweep; ClimaAtmos version at
    `datagen/clima_atmos/held_suarez/scripts/generate_sweep.py`.

Replaced:

- The MITgcm subprocess invocation; everything under
    `datagen/mitgcm/held_suarez_cs/` is the old producer and now lives
    for reference only.

## Known risks (open at time of writing)

1. **Seed perturbation amplitude is set by analogy to the stock
   `sind(long)` bump (0.1 K).** That choice is convention, not a
   measured optimum; if Phase 0.5 sees same-params/different-seed
   trajectories that fail to decorrelate by t≈50 d, bumping the
   amplitude to ~1 K is the first knob to try.
2. **Fast-Ω jet sharpness at he24.** At he24 the grid spacing is
   ~36 km, comfortable for the Rossby radius at `2 × Ω_Earth` (~half
   Earth's), so eddy resolution is fine. The remaining concern is jet
   sharpness — fast-Ω corners can develop tight jets that may need
   inspection of the zonal-wind field. Validate during Phase 0.4 by
   comparing the kinetic-energy spectrum across the Ω levels.
3. **Production wall-time at `he24ze31` × 565 d is not yet validated
   across the full Ω range.** A single he24 corner has run without
   failure; the 8 h preflight budget covers the slow- and fast-Ω
   corners (which can have different effective Courant numbers). The
   8-corner preflight at he24 × 60 sim-d is the load-bearing check;
   we won't submit the 162-run array until it lands. Budget at
   ~5 h/run × 162 runs ≈ 810 GPU-hours.
4. **Regrid target not yet pinned.** he24 native output is ~144×288
   (~1.25°). The MITgcm schema defaulted to 180×90 (~2°); 360×180
   (~1°) is the recommended compromise. Resolve during Phase 0
   alongside cadence and retention policy.
5. **HuggingFace publish path** is not wired up yet. The Zarr schema
   matches the MITgcm HS-3D dataset, so the existing HF upload recipe
   will work once we settle the bundle layout.

The earlier "TOML override keys may drift" and "CFL across rotation
rates" risks are retired: the three physical axes use ClimaParams-native
keys and rotation rate doesn't enter CFL (we dropped the one axis that
did, planetary radius `a`).
