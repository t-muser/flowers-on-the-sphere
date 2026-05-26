#!/bin/bash
#SBATCH --job-name=clima-phase0-smoke
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=datagen/clima_atmos/held_suarez/slurm/logs/phase0_smoke-%j.out
#SBATCH --error=datagen/clima_atmos/held_suarez/slurm/logs/phase0_smoke-%j.err
#
# Phase 0.2: run the stock ClimaAtmos.jl Held-Suarez example at he6ze31,
# t_end=2days, but invoked through our run_hs.jl driver (not CliMA's
# buildkite ci_driver, which depends on a long list of post-proc
# packages — PrettyTables, CairoMakie, ProfileCanvas, etc.).
#
# CliMA's own CI uses TWO --config_file args: the numerics config and
# the HS config. Passing only the HS config triggers cloud-microphysics
# defaults that crash the dry solver — we hit this once. The smoke uses
# the same two-config combination CI does.

set -euo pipefail

module purge
module load Julia/1.10.8-linux-x86_64

export JULIA_DEPOT_PATH="${PWD}/datagen/clima_atmos/held_suarez/env/.julia_depot"

REPO_ROOT="${PWD}"
ENV_DIR="${REPO_ROOT}/datagen/clima_atmos/held_suarez/env"
CLIMA_DIR="${REPO_ROOT}/external/ClimaAtmos.jl"
DRIVER_JL="${REPO_ROOT}/datagen/clima_atmos/held_suarez/run_hs.jl"
OUT_DIR="${REPO_ROOT}/datagen/clima_atmos/held_suarez/slurm/logs/phase0_smoke_out"

mkdir -p "${OUT_DIR}" datagen/clima_atmos/held_suarez/slurm/logs

if [ ! -d "${CLIMA_DIR}" ]; then
    echo "ERROR: ${CLIMA_DIR} not present."
    exit 2
fi

# Generate a stock-defaults corner JSON so run_hs.jl has a uniform input.
CORNER_JSON="${OUT_DIR}/stock_corner.json"
cat > "${CORNER_JSON}" <<'EOF'
{
  "run_id": 0,
  "run_name": "phase0_smoke",
  "params": {
    "omega_factor":  1.0,
    "delta_T_y":     60.0,
    "delta_theta_z": 10.0,
    "seed":          0
  }
}
EOF

# Smoke-specific config: HS forcing settings + diagnostic blocks so the
# run writes NetCDF for ua/va/ta + ps. We re-emit the held_suarez.yml
# keys here rather than layering with a third --config_file because
# run_hs.jl currently exposes only two slots.
SMOKE_HS="${OUT_DIR}/held_suarez_with_diagnostics.yml"
cat > "${SMOKE_HS}" <<'EOF'
z_max: 55000.0
viscous_sponge: false
implicit_diffusion: false
dt_save_state_to_disk: "2days"
t_end: "2days"
disable_surface_flux_tendency: true
forcing: "held_suarez"
toml: ["toml/sphere_held_suarez.toml"]
netcdf_output_at_levels: false
diagnostics:
  # Primary 3D fields on pressure coords (ClimaAtmos's RealPressureLevels
  # writer; ERA5 levels are a subset of the 37-level default axis).
  - short_name: [ua, va, ta]
    period: "6hours"
    reduction_time: "inst"
    pressure_coordinates: true
  # pfull on the model's native height axis — used by postprocess.py to
  # derive surface pressure as pfull[z=0]. Same 6h cadence so the time
  # axes align with ua/va/ta after postprocess.
  - short_name: [pfull]
    period: "6hours"
    reduction_time: "inst"
EOF

echo "=== Phase 0.2 smoke"
echo "    ClimaAtmos: $(cd "${CLIMA_DIR}" && git describe --tags --always 2>/dev/null || echo unknown)"
echo "    env:        ${ENV_DIR}"
echo "    out:        ${OUT_DIR}"
echo "    corner:     ${CORNER_JSON}"
echo "    depot:      ${JULIA_DEPOT_PATH}"

julia --project="${ENV_DIR}" "${DRIVER_JL}" \
    --corner "${CORNER_JSON}" \
    --base-config "${CLIMA_DIR}/config/common_configs/numerics_sphere_he6ze31.yml" \
    --resolution-config "${SMOKE_HS}" \
    --out-dir "${OUT_DIR}/run" \
    --job-id phase0_smoke

echo "=== Phase 0.2 smoke completed"
ls -la "${OUT_DIR}/run/" 2>&1
find "${OUT_DIR}/run" -name "*.nc" | head -5
