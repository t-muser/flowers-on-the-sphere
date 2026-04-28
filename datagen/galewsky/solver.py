"""Dedalus v3 shallow-water solver on the rotating sphere.

Builds the problem, sets Galewsky initial conditions, runs the time loop with
CFL-adaptive ``dt``, and writes native-grid HDF5 snapshots via Dedalus's
``FileHandler``. Callers should pass parameter dicts emitted by
``datagen/galewsky/scripts/generate_sweep.py``.

The equations are the shallow-water system on the sphere with added
biharmonic hyperviscosity::

    dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = -u@grad(u)
    dt(h) + nu*lap(lap(h)) + H*div(u)                     = -div(h*u)

where ``h`` is the perturbation depth, total depth is ``H + h``, and
``zcross(u) = sin(lat) * k̂×u`` (so ``2*Omega*zcross(u) = f*k̂×u``).

All user-facing quantities — CLI flags, JSON configs, HDF5 ``sim_time``
metadata interpreted through ``datagen.resample`` — are physical SI units.
The solver itself runs in scaled units (``R_earth = 1``, 1 time unit =
1 hour), matching the canonical Dedalus shallow-water example. See
``datagen.galewsky._units`` for the conversion factors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import dedalus.public as d3
from dedalus.core.basis import SphereBasis
from dedalus.core.coords import S2Coordinates
from dedalus.core.distributor import Distributor
from dedalus.core.field import Field, VectorField
from dedalus.core.problems import InitialValueProblem
from dedalus.extras import flow_tools

from datagen.galewsky._units import METER, SECOND
from datagen.galewsky.ic import set_initial_conditions

# Fixed physics constants, expressed in the solver's sim units. The
# physical SI values are ``R_earth = 6.37122e6 m``, ``Omega = 7.292e-5 rad/s``,
# ``g = 9.80616 m/s²``.
R_EARTH = 6.37122e6 * METER  # = 1.0
OMEGA = 7.292e-5 / SECOND  # ≈ 0.2625 rad/sim-hour
G = 9.80616 * METER / SECOND ** 2  # ≈ 0.01992 sim-length/sim-time²

# Hyperviscosity coefficient for the biharmonic operator ``ν·∇⁴``.
# We match the damping rate of a physical Laplacian viscosity ν_lap = 1e5 m²/s
# at spherical harmonic degree ``ELL_MATCH = 32`` (the Dedalus shallow-water
# example's reference). Matching
#     ν_lap · ℓ(ℓ+1)/R²  =  ν_bi · [ℓ(ℓ+1)/R²]²
# gives  ν_bi = ν_lap · R²/ℓ²  at the match degree, in physical m⁴/s.
# Converting to sim units and absorbing R = 1:
#     NU_BASE_SIM = ν_lap · METER²/SECOND / ELL_MATCH²
#
# Resolution scaling is ``∝ 1/Ntheta⁴`` because the biharmonic's eigenvalue
# grows as ℓ⁴ — that keeps the grid-scale damping time invariant as Ntheta
# changes. At ``Ntheta = NU_REF_NTHETA`` the multiplier is 1.
NU0_PHYS = 1.0e5  # [m²/s] — reference Laplacian viscosity
ELL_MATCH = 32
NU_BASE_SIM = (NU0_PHYS * METER ** 2 / SECOND) / (ELL_MATCH ** 2)
NU_REF_NTHETA = 256

_LOG_CADENCE = 100
_MONITOR_CADENCE = 50

logger = logging.getLogger(__name__)


def _hyperviscosity(Ntheta: int) -> float:
    """Hyperviscosity (sim units) for the biharmonic operator."""
    return NU_BASE_SIM * (NU_REF_NTHETA / Ntheta) ** 4


def _equivalent_lap_viscosity_phys(nu_bi_sim: float) -> float:
    """Physical Laplacian viscosity (m²/s) matched to ν_bi at ELL_MATCH."""
    return nu_bi_sim * ELL_MATCH ** 2 * SECOND / METER ** 2


def _to_sim_params(params: dict) -> dict:
    """Convert a physical-SI parameter dict into a fresh sim-unit dict."""
    return {
        "H": float(params["H"]) * METER,
        "u_max": float(params["u_max"]) * METER / SECOND,
        "h_hat": float(params["h_hat"]) * METER,
        "lat_center": float(params["lat_center"]),
    }


@dataclass(frozen=True)
class RunConfig:
    """Run configuration. Time units are in physical seconds."""
    snapshot_det: float = 3600.0
    stop_sim_time: float = 16 * 86400.0
    Nphi: int = 512
    Ntheta: int = 256
    initial_dt: float = 120.0
    max_dt: float = 600.0
    cfl_safety: float = 0.3
    max_writes_per_file: int = 300


@dataclass(frozen=True)
class SimulationParams:
    """Parameters of the Galewski simulation in sim units, built from a physical-SI dict."""
    H: float  # Mean depth [sim length]
    u_max: float  # Jet peak speed [sim length/sim time]
    h_hat: float  # Perturbation amplitude [sim length]
    lat_center: float  # Jet center latitude [sim rad]

    @classmethod
    def from_physical(cls, params: dict[str, Any]) -> "SimulationParams":
        return cls(
            H=float(params["H"]) * METER,
            u_max=float(params["u_max"]) * METER / SECOND,
            h_hat=float(params["h_hat"]) * METER,
            lat_center=float(params["lat_center"]),
        )


@dataclass(frozen=True)
class SpectralContext:
    """Spectral discretization objects (shared across helpers)."""
    coords: S2Coordinates
    dist: Distributor
    basis: SphereBasis


@dataclass(frozen=True)
class ProblemBundle:
    """A configured Dedalus IVP, plus its primary fields and discretization."""
    problem: InitialValueProblem
    u: VectorField
    h: Field
    ctx: SpectralContext


def _build_problem(
        cfg: RunConfig, sim_params: SimulationParams, nu: float
) -> ProblemBundle:
    """Construct the Dedalus IVP. Returns (problem, u, h, coords, dist, basis)."""
    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.SphereBasis(coords, (cfg.Nphi, cfg.Ntheta), radius=R_EARTH)

    u = dist.VectorField(coords, name="u", bases=basis)
    h = dist.Field(name="h", bases=basis)

    def zcross(A):
        return d3.MulCosine(d3.skew(A))

    H = sim_params.H
    g = G
    Omega = OMEGA
    _ = (zcross, H, g, Omega, nu)

    problem = d3.IVP(variables=[u, h], namespace=locals())
    problem.add_equation(
        "dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = -u@grad(u)"
    )
    problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = -div(h*u)")

    bundle = ProblemBundle(problem, u, h, SpectralContext(coords, dist, basis))
    return bundle


def _attach_outputs(
        solver, bundle: ProblemBundle, out_dir: Path, snapshot_dt_sim: float, max_writes: int,
):
    """Register HDF5 snapshot outputs (velocity, height, vorticity)."""
    handler = solver.evaluator.add_file_handler(
        str(out_dir), sim_dt=snapshot_dt_sim, max_writes=max_writes,
    )
    handler.add_task(bundle.u, name="u", layout="g")
    handler.add_task(bundle.h, name="h", layout="g")
    handler.add_task(-d3.div(d3.skew(bundle.u)), name="vorticity", layout="g")
    return handler


def _build_cfl(
        solver, u: VectorField, cfg: RunConfig,
        initial_dt_sim: float, max_dt_sim: float,
):
    cfl = flow_tools.CFL(
        solver,
        initial_dt=initial_dt_sim,
        cadence=10,
        safety=cfg.cfl_safety,
        threshold=0.05,
        max_change=1.5,
        min_change=0.5,
        max_dt=max_dt_sim,
    )
    cfl.add_velocity(u)
    return cfl


def _log_run_header(
        cfg: RunConfig, params: dict[str, Any], nu: float, stop_sim_time: float,
) -> None:
    """Log resolution, hyperviscosity and Galewsky parameters in physical units."""
    logger.info(
        "Starting run: Nphi=%d Ntheta=%d nu_biharm=%.3e [sim] "
        "(≡ %.3e m²/s Laplacian at ℓ=%d) "
        "u_max=%g [m/s] lat_center=%g h_hat=%g [m] H=%g [m] "
        "stop_sim_time=%g [s]",
        cfg.Nphi, cfg.Ntheta, nu,
        _equivalent_lap_viscosity_phys(nu), ELL_MATCH,
        float(params["u_max"]), float(params["lat_center"]),
        float(params["h_hat"]), float(params["H"]),
        stop_sim_time,
    )


def run_simulation(
        params: dict,
        out_dir: Path,
        snapshot_dt: float = 3600.0,
        stop_sim_time: float = 16 * 86400.0,
        Nphi: int = 512,
        Ntheta: int = 256,
        initial_dt: float = 120.0,
        max_dt: float = 600.0,
        cfl_safety: float = 0.3,
        max_writes_per_file: int = 300,
) -> None:
    """Run one Galewsky shallow-water simulation and write HDF5 snapshots.

    All time arguments (``snapshot_dt``, ``stop_sim_time``, ``initial_dt``,
    ``max_dt``) are in physical seconds; all ``params`` values are in
    physical SI units. Conversion to the solver's sim units happens here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Physical → sim conversion ------------------------------------
    p = _to_sim_params(params)
    H = p["H"]
    nu = _hyperviscosity(Ntheta)

    snapshot_dt_sim = snapshot_dt * SECOND
    stop_sim_time_sim = stop_sim_time * SECOND
    initial_dt_sim = initial_dt * SECOND
    max_dt_sim = max_dt * SECOND

    # ---- Dedalus problem setup ----------------------------------------
    dtype = np.float64
    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(
        coords, (Nphi, Ntheta), radius=R_EARTH, dealias=3 / 2, dtype=dtype
    )

    u = dist.VectorField(coords, name="u", bases=basis)
    h = dist.Field(name="h", bases=basis)

    g = G
    Omega = OMEGA
    zcross = lambda A: d3.MulCosine(d3.skew(A))  # noqa: E731

    problem = d3.IVP([u, h], namespace=locals())
    problem.add_equation(
        "dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = -u@grad(u)"
    )
    problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = -div(h*u)")

    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time_sim

    set_initial_conditions(
        u, h, p, coords, dist, basis, g=g, R=R_EARTH, Omega=Omega
    )

    # ---- Output handler ----------------------------------------------
    snapshots = solver.evaluator.add_file_handler(
        str(out_dir),
        sim_dt=snapshot_dt_sim,
        max_writes=max_writes_per_file,
    )
    snapshots.add_task(u, name="u", layout="g")
    snapshots.add_task(h, name="h", layout="g")
    snapshots.add_task(-d3.div(d3.skew(u)), name="vorticity", layout="g")

    # ---- CFL + monitors -----------------------------------------------
    cfl = flow_tools.CFL(
        solver,
        initial_dt=initial_dt_sim,
        cadence=10,
        safety=cfl_safety,
        threshold=0.05,
        max_change=1.5,
        min_change=0.5,
        max_dt=max_dt_sim,
    )
    cfl.add_velocity(u)

    flow = flow_tools.GlobalFlowProperty(solver, cadence=50)
    flow.add_property(np.sqrt(u @ u), name="speed")

    # Log in physical units for readability. The "nu equiv" is the
    # physical Laplacian viscosity (m²/s) whose damping at ℓ=ELL_MATCH
    # matches this biharmonic coefficient.
    nu_equiv_lap_phys = nu * (ELL_MATCH ** 2) * SECOND / (METER ** 2)
    logger.info(
        "Starting run: Nphi=%d Ntheta=%d nu_biharm=%.3e [sim] "
        "(≡ %.3e m²/s Laplacian at ℓ=%d)  "
        "u_max=%g [m/s] lat_center=%g h_hat=%g [m] H=%g [m] "
        "stop_sim_time=%g [s]",
        Nphi, Ntheta, nu, nu_equiv_lap_phys, ELL_MATCH,
        float(params["u_max"]), float(params["lat_center"]),
        float(params["h_hat"]), float(params["H"]),
        stop_sim_time,
    )
    wallclock_start = time.time()
    try:
        while solver.proceed:
            dt = cfl.compute_timestep()
            solver.step(dt)
            if solver.iteration % 100 == 0:
                max_speed_sim = flow.max("speed")
                max_speed_phys = max_speed_sim / (METER / SECOND)
                sim_time_days = solver.sim_time / SECOND / 86400.0
                dt_phys = dt / SECOND
                logger.info(
                    "it=%d t=%.3g d dt=%.3g s max|u|=%.3g m/s",
                    solver.iteration, sim_time_days, dt_phys, max_speed_phys,
                )
                if not np.isfinite(max_speed_sim):
                    raise RuntimeError(
                        f"Non-finite velocity at iteration {solver.iteration}"
                    )
    finally:
        solver.log_stats()
        logger.info("Wallclock: %.1f s", time.time() - wallclock_start)
