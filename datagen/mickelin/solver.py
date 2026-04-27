"""Dedalus v3 solver for the Mickelin GNS equation on the sphere.

Vorticity–stream form on a sphere of radius ``R``::

    Δψ = −ω
    ∂_t ω + J(ψ, ω) = f(Δ + 4K)(Δ + 2K) ω
    f(x) = Γ₀ − Γ₂·x + Γ₄·x²,     K = 1/R²

The right-hand-side linear operator expands as

    L ω = Γ₀·(Δ+2K)·ω − Γ₂·(Δ+4K)(Δ+2K)·ω + Γ₄·(Δ+4K)²(Δ+2K)·ω

and is treated implicitly (its highest derivatives are grid-scale-limiting
without an extra hyperviscosity). The nonlinear advection ``J(ψ, ω)`` is
written as ``u · ∇ω`` with divergence-free velocity
``u = skew(∇ψ)`` and kept on the explicit side of the IMEX split.

Callers should pass parameter dicts emitted by
``datagen.mickelin.scripts.generate_sweep``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import dedalus.public as d3
from dedalus.extras import flow_tools

from datagen.mickelin.coeffs import (
    coefficients_from_RLkT,
    set_initial_conditions,
    unstable_band_ell_max,
)

logger = logging.getLogger(__name__)


def run_simulation(
    params: dict,
    out_dir: Path,
    snapshot_dt: float = 0.2,
    stop_sim_time: float = 130.0,
    Nphi: int = 256,
    Ntheta: int = 128,
    initial_dt: float = 5.0e-3,
    max_dt: float = 5.0e-2,
    cfl_safety: float = 0.3,
    max_writes_per_file: int = 250,
    ell_init: int | None = None,
    epsilon: float = 1.0e-3,
) -> None:
    """Run one Mickelin GNS simulation and write HDF5 snapshots.

    ``params`` must contain ``R``, ``Lambda``, ``kappa``, ``tau``, ``seed``.
    Snapshot cadence ``snapshot_dt`` and ``stop_sim_time`` are expressed in
    the same time units as ``tau``. Pass ``ell_init=None`` to derive it from
    the unstable band.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    R = float(params["R"])
    Lambda = float(params["Lambda"])
    kappa = float(params["kappa"])
    tau = float(params["tau"])
    seed = int(params["seed"])

    Gamma_0, Gamma_2, Gamma_4 = coefficients_from_RLkT(R, Lambda, kappa, tau)
    K = 1.0 / R ** 2

    if ell_init is None:
        ell_init = unstable_band_ell_max(R, Lambda, kappa)

    dtype = np.float64
    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(
        coords, (Nphi, Ntheta), radius=R, dealias=3 / 2, dtype=dtype
    )

    omega = dist.Field(name="omega", bases=basis)
    psi = dist.Field(name="psi", bases=basis)
    # The sphere Laplacian has a 1-D null space (constants). Absorb the
    # ℓ=0 mode with a scalar tau + gauge condition ``integ(psi) = 0``,
    # which makes the LHS solve well-posed without over-determining the
    # system.
    tau_psi = dist.Field(name="tau_psi")

    tau_omega = dist.Field(name="tau_omega")

    # Build the expanded linear operator as a Dedalus expression tree.
    A = d3.lap(omega) + 2.0 * K * omega            # (Δ + 2K) ω
    B_A = d3.lap(A) + 4.0 * K * A                  # (Δ + 4K)(Δ + 2K) ω
    B2_A = d3.lap(B_A) + 4.0 * K * B_A             # (Δ + 4K)²(Δ + 2K) ω
    L_omega = Gamma_0 * A - Gamma_2 * B_A + Gamma_4 * B2_A

    u = d3.skew(d3.grad(psi))

    problem = d3.IVP([psi, tau_psi, omega, tau_omega], namespace=locals())
    problem.add_equation("lap(psi) + tau_psi = -omega")
    problem.add_equation("ave(psi) = 0")
    problem.add_equation(
        (d3.dt(omega) - L_omega + tau_omega, -u @ d3.grad(omega))
    )
    problem.add_equation("ave(omega) = 0")

    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    set_initial_conditions(
        omega, dist, basis, seed=seed, ell_init=ell_init, epsilon=epsilon
    )

    snapshots = solver.evaluator.add_file_handler(
        str(out_dir),
        sim_dt=snapshot_dt,
        max_writes=max_writes_per_file,
    )
    snapshots.add_task(omega, name="vorticity", layout="g")

    cfl = flow_tools.CFL(
        solver,
        initial_dt=initial_dt,
        cadence=10,
        safety=cfl_safety,
        threshold=0.05,
        max_change=1.5,
        min_change=0.5,
        max_dt=max_dt,
    )
    cfl.add_velocity(u)

    flow = flow_tools.GlobalFlowProperty(solver, cadence=50)
    flow.add_property(np.sqrt(u @ u), name="speed")
    flow.add_property(omega, name="omega")

    logger.info(
        "Starting run: Nphi=%d Ntheta=%d R=%g Lambda=%g kappa=%g tau=%g seed=%d "
        "Gamma_0=%.3e Gamma_2=%.3e Gamma_4=%.3e stop_sim_time=%g",
        Nphi, Ntheta, R, Lambda, kappa, tau, seed,
        Gamma_0, Gamma_2, Gamma_4, stop_sim_time,
    )
    wallclock_start = time.time()
    try:
        while solver.proceed:
            dt = cfl.compute_timestep()
            solver.step(dt)
            if solver.iteration % 100 == 0:
                max_speed = flow.max("speed")
                max_omega = flow.max("omega")
                logger.info(
                    "it=%d t=%.3f τ dt=%.3g max|u|=%.3g max|ω|=%.3g",
                    solver.iteration,
                    solver.sim_time / tau,
                    dt,
                    max_speed,
                    max_omega,
                )
                if not np.isfinite(max_speed):
                    raise RuntimeError(
                        f"Non-finite velocity at iteration {solver.iteration}"
                    )
    finally:
        solver.log_stats()
        logger.info("Wallclock: %.1f s", time.time() - wallclock_start)
