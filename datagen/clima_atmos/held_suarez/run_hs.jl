#!/usr/bin/env julia
# Single-run Held-Suarez driver on top of ClimaAtmos.jl.
#
# CLI shape:
#     julia --project=env run_hs.jl --corner CORNER.json \
#         --base-config /path/to/held_suarez.yml \
#         --resolution-config /path/to/he12ze31_overrides.yml \
#         --out-dir OUT_DIR --job-id JOB_ID
#
# What it does:
#   1. Reads the corner JSON emitted by generate_sweep.py (or preflight.py).
#   2. Constructs a TOML override dict that maps the 3 physical sweep
#      axes onto ClimaParams keys read by the Held-Suarez forcing:
#        omega_factor   → angular_velocity_planet_rotation  (Ω)
#        delta_T_y      → equator_pole_temperature_gradient_dry  (ΔT_y)
#        delta_theta_z  → potential_temp_vertical_gradient        (Δθ_z)
#      and writes it to <out_dir>/overrides.toml. The IC-seed axis is
#      threaded through the HS_IC_SEED env var, read by the patched
#      DecayingProfile setup.
#   3. Launches the standard ClimaAtmos ci_driver pipeline with the base HS
#      config, the resolution override, and our parameter override.
#
# We deliberately keep the Julia surface area small. Anything we can do in
# Python (orchestration, postprocess, finalize) lives there.

using TOML
using ArgParse
using JSON
using Logging

# Load ClimaAtmos at top-level — loading it inside ``main()`` triggers a
# world-age error: methods defined by ``using`` inside a function are
# invisible to the same call frame (``method too new to be called from
# this world context``). Top-level ``import`` is the canonical fix.
#
# Note: import CUDA *before* ClimaComms so the ClimaCommsCUDAExt extension
# activates (it provides `ClimaComms.array_type(::CUDADevice)` which the
# topology constructor needs). The `device: "CUDADevice"` YAML key drives
# the choice.
import CUDA
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

# Earth's angular velocity in rad/s — ClimaParams default. We multiply by
# the corner's omega_factor to get the absolute Ω handed to ClimaAtmos.
const OMEGA_EARTH = 7.2921159e-5

function parse_cli()
    s = ArgParseSettings(description = "Held-Suarez run driver (ClimaAtmos)")
    @add_arg_table! s begin
        "--corner"
            help = "Path to corner/run JSON config (with .params dict)"
            required = true
        "--base-config"
            help = "Path to ClimaAtmos held_suarez.yml (or compatible)"
            required = true
        "--resolution-config"
            help = "Optional YAML overriding h_elem/z_elem/dt/t_end"
            default = ""
        "--out-dir"
            help = "Output directory (will be created)"
            required = true
        "--job-id"
            help = "ClimaAtmos job_id (used in output filenames)"
            required = true
        "--t-end"
            help = "Override t_end (e.g. \"565days\"); empty = take from configs"
            default = ""
    end
    return parse_args(s)
end

# Convert one corner's physical params → ClimaParams TOML override dict.
#
# All three physical sweep axes hit ClimaParams keys directly, so there is
# no source patch dependency:
#   angular_velocity_planet_rotation       → Ω
#   equator_pole_temperature_gradient_dry  → ΔT_y_dry (used in dry HS)
#   potential_temp_vertical_gradient       → Δθ_z
#
# The HS damping timescales (k_a / k_s / k_f) remain hardcoded in
# src/parameterized_tendencies/radiation/held_suarez.jl. We deliberately
# do not vary them — they control how fast the flow relaxes to
# equilibrium rather than what the equilibrium is.
function build_param_override(params::Dict)
    omega = params["omega_factor"] * OMEGA_EARTH
    return Dict(
        "angular_velocity_planet_rotation" =>
            Dict("value" => omega),
        "equator_pole_temperature_gradient_dry" =>
            Dict("value" => params["delta_T_y"]),
        "potential_temp_vertical_gradient"      =>
            Dict("value" => params["delta_theta_z"]),
    )
end

function main()
    args = parse_cli()
    out_dir = abspath(args["out-dir"])
    mkpath(out_dir)

    @info "Loading corner config" path=args["corner"]
    corner = open(JSON.parse, args["corner"])
    params = corner["params"]
    @info "Corner params" params

    # 1. Emit the parameter TOML override.
    overrides = build_param_override(params)
    toml_path = joinpath(out_dir, "overrides.toml")
    open(toml_path, "w") do io
        TOML.print(io, overrides)
    end
    @info "Wrote parameter override TOML" path=toml_path

    # 2. Build the YAML override stub for this run: out path, job id,
    #    optional t_end, and the TOML override path.
    run_override_path = joinpath(out_dir, "run_override.yml")
    open(run_override_path, "w") do io
        # ClimaAtmos reads ``toml`` as a list of files that get layered onto
        # the default ClimaParams TOML dict in order.
        println(io, "toml: [\"$(toml_path)\"]")
        println(io, "output_dir: \"$(out_dir)\"")
        println(io, "job_id: \"$(args["job-id"])\"")
        if args["t-end"] != ""
            println(io, "t_end: \"$(args["t-end"])\"")
        end
    end
    @info "Wrote run override YAML" path=run_override_path

    # 3. Plumb the IC seed through to the (patched) DecayingProfile setup
    #    via env var. Stock CliMA ignores the var; on our patched
    #    checkout, _temperature_perturbation() reads it and draws
    #    per-cell Gaussian noise seeded by hash(seed, lat, long, z).
    ENV["HS_IC_SEED"] = string(params["seed"])
    @info "HS IC seed set" seed=params["seed"]

    # 4. Hand off to ClimaAtmos. We replicate the buildkite ci_driver
    #    pattern (get_simulation + solve_atmos!) without pulling in the
    #    diagnostic post-proc deps (PrettyTables, CairoMakie, etc.).
    config_files = [args["base-config"]]
    if args["resolution-config"] != ""
        push!(config_files, args["resolution-config"])
    end
    push!(config_files, run_override_path)

    @info "Building AtmosConfig" config_files=config_files
    atmos_config = CA.AtmosConfig(config_files; job_id=args["job-id"])

    @info "get_simulation"
    simulation = CA.get_simulation(atmos_config)

    @info "solve_atmos!"
    sol_res = CA.solve_atmos!(simulation)
    if sol_res.ret_code == :simulation_crashed
        error("ClimaAtmos simulation crashed; see stacktrace above.")
    end

    @info "Run complete" out_dir=out_dir
    return 0
end

exit(main())
