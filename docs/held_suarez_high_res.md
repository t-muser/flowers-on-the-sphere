# Held-Suarez at Higher Resolution — Feasibility & Validity

This note assesses what it would take to generate a higher-resolution version
of the Held-Suarez (HS) dataset — concretely doubling the native horizontal
grid from the current `Nlon × Nlat = 128 × 64` (≈ "C32 / T32") to `256 × 128`
(≈ "C64 / T64"). It covers both the engineering cost and the **scientific**
question of whether such an upscaling is meaningful, including the contrast
with the ocean case where naive resolution doubling is problematic.

## Current generation pipeline (summary)

The HS pipeline lives under
[`datagen/mitgcm/held_suarez/`](../datagen/mitgcm/held_suarez) and is documented
in [`held_suarez.md`](held_suarez.md) (2-D variant) and
[`held_suarez_3d.md`](held_suarez_3d.md) (3-D variant). The relevant facts for
this note:

- **Native solver grid** is fixed at compile time in
  [`code/SIZE.h`](../datagen/mitgcm/held_suarez/code/SIZE.h):
  `sNx=128, sNy=16, nPy=4` → `Nlon × Nlat × Nr = 128 × 64 × 20`.
- **Forcing** is the Held & Suarez (1994) Newtonian thermostat + Rayleigh drag,
  implemented in
  [`code/apply_forcing.F`](../datagen/mitgcm/held_suarez/code/apply_forcing.F)
  and mirrored in pure Python in
  [`_physics.py`](../datagen/mitgcm/held_suarez/_physics.py). All forcing
  parameters are pointwise functions of `(φ, p, σ)` only.
- **Numerical dissipation** is provided exclusively by an 8th-order Shapiro
  filter (`Shap_Trtau = Shap_uvtau = 1200 s`, `nShapT = nShapUV = 4` in
  [`namelist.py:write_data_shap`](../datagen/mitgcm/held_suarez/namelist.py));
  explicit viscosity and diffusivity are zero (`viscAh = 0.0`,
  `diffKhT = 0.0`).
- **Timestep** is `Δt = 45 s` for the 2-D variant and `Δt = 30 s` for the 3-D
  variant. The reduction was needed because the 162-run sweep at `Δt = 45 s`
  had a **26 % CFL-failure rate** concentrated in the weak-drag /
  strong-meridional-forcing corner of the parameter box (see
  [`scripts/generate_sweep_safe.py`](../datagen/mitgcm/held_suarez/scripts/generate_sweep_safe.py)).
- **Ensemble** is a 5-axis tensor product over
  `tau_drag_days × delta_T_y × delta_theta_z × tau_surf_days × seed`, with a
  32-corner preflight sweep used to validate stability before the full
  production array.

## Engineering cost of a 256×128 run

Most of the pipeline is already parameterised on `Nlon`/`Nlat`/`Nr`.

| Change | Where | Effort |
| --- | --- | --- |
| Recompile with new grid | `code/SIZE.h` (`sNx`, `sNy`, `nPy`); the header comment in `SIZE.h:7-8` explicitly documents this knob | Trivial — edit + `sbatch slurm/build.sbatch` |
| Pass new dimensions to the Python driver | `RunConfig.Nlon`/`Nlat` in `solver.py`. Currently no CLI override; needs a small patch in `scripts/run.py` and `scripts/preflight.py` | Small Python patch |
| Reduce Δt | `RunConfig.delta_t` — already a `--delta-t` CLI flag | None |
| Generate IC at new shape | `ic.py:write_temperature_ic` already takes `Nlon`/`Nlat`/`Nr` | None |
| Bathymetry at new shape | `ic.py:write_bathymetry` already parameterised; regenerate via `scripts/build.py:generate_static_inputs` | None |
| MPI layout | `nPy = Nlat / sNy`; e.g. `Nlat=128, sNy=16 → nPy=8`. Bump `--ntasks` in the sbatch wrapper | Trivial |
| Optional: Shapiro retune | `namelist.py:write_data_shap`. Shapiro acts at Nyquist, so a fixed timescale roughly auto-scales — but a polar retune may be needed (see below) | Optional |

A 32-corner preflight at the new resolution is **mandatory** to identify the
new CFL ceiling.

### Compute cost

Doubling the horizontal resolution scales the per-trajectory cost by ~8×:

- 4× more horizontal grid points
- ~2× more timesteps (CFL ⇒ `Δt` halves with `Δx`)

Concretely: the 72-run 3-D sweep at `Δt = 30 s` finishes within a 12-hour SLURM
QoS per run. At 256×128 expect roughly a week of array wall time per run-bundle,
with `Δt` likely needing to drop to **~12–15 s** (not the naive 15 s) to clear
the same fraction of the parameter box that 30 s clears at 128×64. The current
1-day QoS used by
[`production_array_3d.sbatch`](../datagen/mitgcm/held_suarez/slurm/production_array_3d.sbatch)
is therefore insufficient — either move to a longer-QoS partition, shorten the
data-collection window, or split each trajectory into restartable chunks.

**Disk** is not an issue: 17 GB for the 72-run 3-D dataset at 128×64 → ~68 GB
at 256×128.

## Scientific validity

### Why this is **not** the ocean problem

The concern about naive upscaling — eddies appearing alongside the
parameterisation that was originally added to compensate for unresolved eddies
— is a real issue in **ocean** GCMs (Gent–McWilliams / Redi schemes for
isopycnal eddy transport, Smagorinsky-style closures, etc.). Held-Suarez is
specifically designed to avoid this. The benchmark contains:

- **No cumulus convection scheme** (the atmosphere is dry).
- **No gravity-wave-drag parameterisation.**
- **No boundary-layer turbulence closure** — the σ > σ_b Rayleigh drag is a
  fixed-rate proxy, not a turbulence closure.
- **No eddy parameterisation of any kind** — no atmospheric analog of
  Gent–McWilliams.

The code reflects this: `_physics.py:equilibrium_temperature`,
`rayleigh_friction_rate`, and `newtonian_cooling_rate` are pointwise functions
of `(φ, p, σ)` only, and the Fortran in `apply_forcing.F` simply multiplies a
local relaxation rate by `(θ − θ_eq)` per cell. There is **no flux-divergence
eddy term** to rescale or switch off when resolution increases.

Held & Suarez (1994) chose the forcing constants
(`HS_KF`, `HS_KA`, `HS_KS`, `HS_DELTA_T_Y`, `HS_DELTA_T_Z` in
[`_constants.py`](../datagen/mitgcm/held_suarez/_constants.py)) explicitly to
be scale-invariant. That is the whole point of the benchmark — it is a clean
target for dynamical-core intercomparison precisely because the physics does
not need to be retuned across resolutions.

### Subtleties that still apply

1. **Climatological convergence, not equivalence.** The HS climatology
   (zonal-mean `U`, `T`, eddy fluxes, jet latitude, storm-track width) is
   known to be modestly resolution-dependent in the T30–T85 range — jets
   sharpen and storm tracks narrow somewhat at higher resolution. See Wan et
   al. (2008) and the Polvani-line literature on the topic. The dataset
   distribution **will shift** between 32-class and 64-class. That is not a
   bug, but it is a fact worth flagging for any ML model that assumes
   distributional invariance under coarsening.

2. **Effective dissipation range moves.** The Shapiro filter damps the 2Δx
   mode on a fixed timescale at the chosen Δx. Doubling the resolution gives a
   strictly finer effective dissipation range and therefore **more resolved
   eddy activity** at scales that were filtered out at 128×64. Coarsening the
   64-class output back to 32-class will **not** reproduce the 32-class
   climatology one-to-one.

3. **Vertical resolution becomes the bottleneck.** `Nr=20` is already coarse;
   `pressure_thicknesses` in
   [`ic.py`](../datagen/mitgcm/held_suarez/ic.py) puts the top layer at
   ~150 hPa thickness centred near 75 hPa, which is why the 3-D extract
   resolves both 50 hPa and 100 hPa to the same model level (see
   [`held_suarez_3d.md`](held_suarez_3d.md)). Quadrupling horizontal resolution
   without addressing vertical resolution introduces an aspect-ratio mismatch.
   For scientific defensibility, a 256×128 upgrade should be paired with at
   least `Nr=30` (finer stratosphere), at additional compute cost.

4. **Polar singularity worsens.** Spherical-polar grids develop a singular
   zonal CFL at the poles that the Shapiro filter regularises. At 256×128 the
   filter has to work harder; some polar noise in `u`/`v` near 89° is plausible
   even when the run does not formally blow up. Mitigations (selectVortScheme,
   stronger filter at high latitude) are well-trodden in MITgcm.

5. **Failure rate is concentrated, not uniform.** The 162-run sweep's CFL
   failures clustered in `delta_T_y = 80 K`, `delta_theta_z = 5 K`. At higher
   resolution this corner gets worse, so the preflight at 256×128 should be
   prepared either to drop more corner values (as
   [`generate_sweep_safe.py`](../datagen/mitgcm/held_suarez/scripts/generate_sweep_safe.py)
   already does) or to tighten `Δt` further than naive scaling suggests.

### What does **not** apply

- No GM/Redi-style eddy parameterisation to switch off.
- No convection-permitting threshold to cross.
- No mixed-layer or sub-mesoscale scheme to retune.
- No restoring boundary conditions to revisit.

## Bottom line

- **Engineering cost: low.** ~1 engineer-day to wire the `Nlon`/`Nlat` CLI
  override, regenerate ICs/bathymetry, and rebuild. The `SIZE.h` header
  already documents this upgrade path.
- **Compute cost: ~8× the current 3-D sweep.** Tractable on the existing
  scicore array setup, but the per-run wall time exceeds the current 1-day
  QoS; either move QoS or split trajectories.
- **Scientific validity: clean.** Held-Suarez has no parameterisation /
  resolved-scale double-count to worry about — the forcing is pointwise and
  scale-invariant by construction. This is in deliberate contrast to ocean
  GCMs and is the entire reason HS exists as a dynamical-core benchmark.
- **Caveats worth stating in any paper:**
  1. Climatology shifts modestly with resolution (Wan et al. 2008 et seq.).
  2. The Shapiro filter's effective dissipation range tightens, so the
     high-wavenumber spectrum genuinely differs.
  3. `Nr=20` becomes the dominant resolution bottleneck — consider `Nr≥30`.
  4. The polar CFL is more aggressive; the 32-corner preflight should be
     re-run at the target Δt.

## Concrete next steps (if pursued)

1. Add a `--nlon`/`--nlat`/`--nr` set of CLI flags to `scripts/run.py` and
   `scripts/preflight.py`, threaded into `RunConfig` and (separately) into a
   templated `SIZE.h`.
2. Add `slurm/build_256x128.sbatch` that compiles against a `SIZE_256x128.h`
   variant. Keep the 128×64 build alongside.
3. Add `slurm/preflight_array_256x128.sbatch` that runs the 32 corners at
   `Δt = 15 s` first, then at `Δt = 12 s` for any failures.
4. Decide on `Nr` (20 vs 30) before launching the production sweep — the
   compute cost difference is linear and significant.
5. Produce a short companion note documenting the climatology delta against
   the 128×64 dataset (zonal-mean `U`, `T`, eddy heat flux, jet latitude) so
   downstream ML users can see the distribution shift quantitatively.
