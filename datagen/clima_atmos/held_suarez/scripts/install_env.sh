#!/usr/bin/env bash
# One-time Julia environment install.
#
# Run on a compute node (see slurm/install_env.sbatch) — instantiate +
# precompile of ClimaAtmos.jl is heavy enough that running it on the
# login node is rude.
#
# After the first run, commit env/Manifest.toml so subsequent installs are
# reproducible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "${SCRIPT_DIR}/../env" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CLIMA_DIR="${REPO_ROOT}/external/ClimaAtmos.jl"

echo "=== Julia env at ${ENV_DIR}"
julia --version
echo "    JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-<default>}"
echo "    CLIMA_DIR=${CLIMA_DIR}"

if [ ! -d "${CLIMA_DIR}" ]; then
    echo "ERROR: ${CLIMA_DIR} not present. Clone it first:"
    echo "    git clone --depth 1 --branch v0.39.0 \\"
    echo "        https://github.com/CliMA/ClimaAtmos.jl ${CLIMA_DIR}"
    exit 2
fi

# Dev-link the locally patched ClimaAtmos clone. The HS forcing patch
# (env/source_patch.diff) lives in that clone so we must use it, not a
# registry release. ``Pkg.develop`` registers the local path as the
# resolved source of the ClimaAtmos package.
#
# Split:
#   Pkg.develop + Pkg.instantiate  → MUST run on the login node (scicore
#                                    compute nodes are blocked from
#                                    pkg.julialang.org / github.com).
#   Pkg.precompile                  → runs anywhere; expensive enough to
#                                    push onto a compute node.
#
# Trigger which step by setting INSTALL_PHASE=instantiate or =precompile.
PHASE="${INSTALL_PHASE:-instantiate}"

if [ "${PHASE}" = "instantiate" ]; then
    julia --project="${ENV_DIR}" -e "
        using Pkg
        Pkg.develop(PackageSpec(path = \"${CLIMA_DIR}\"))
        Pkg.instantiate()
    "
elif [ "${PHASE}" = "precompile" ]; then
    julia --project="${ENV_DIR}" -e "
        using Pkg
        Pkg.precompile()
    "
elif [ "${PHASE}" = "all" ]; then
    julia --project="${ENV_DIR}" -e "
        using Pkg
        Pkg.develop(PackageSpec(path = \"${CLIMA_DIR}\"))
        Pkg.instantiate()
        Pkg.precompile()
    "
else
    echo "ERROR: unknown INSTALL_PHASE=${PHASE} (want instantiate|precompile|all)"
    exit 2
fi

if [ "${PHASE}" = "precompile" ] || [ "${PHASE}" = "all" ]; then
    # Smoke test: imports succeed. Skipped on instantiate-only because
    # precompile hasn't run yet (it would trigger the same expensive
    # compile cascade we wanted to push to a compute node).
    julia --project="${ENV_DIR}" -e '
        using ClimaAtmos
        using ClimaParams
        using ClimaComms
        using NCDatasets
        println("OK: ClimaAtmos + ClimaParams + ClimaComms + NCDatasets loaded")
    '
fi
