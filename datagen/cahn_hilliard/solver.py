"""Dedalus v3 Cahn-Hilliard solver on the sphere with Krekhov-style forcing.

Equation, following Krekhov / Weith-Krekhov-Zimmermann (2009)::

    ∂_t ψ = ∇² [ −ε·ψ + ψ³ − ξ²·∇²ψ + a·F(x̂, t) ]
          = −ε·∇²ψ − ξ²·∇⁴ψ + ∇²(ψ³) + a·∇²F

ψ ∈ [-1, 1] is the order parameter. ``ε`` is the bulk control parameter
(positive ⇒ ordered phase); ``ξ`` is the interface width. Linear
stability about ``ψ=0`` is unstable for ``k² < ε/ξ²``, so on a sphere
of radius ``R`` patterns require ``ε/ξ² > 2/R²``. The forcing field
``F(x̂, t) = Re[Y_ℓ^m(R_ê(ω·t)·x̂)]`` is a single ``Y_ℓ^m`` whose pattern
rotates rigidly about a per-trajectory axis ``ê`` drawn from ``seed``;
because ``∇² Y_ℓ^m = −ℓ(ℓ+1)/R² · Y_ℓ^m``, ``∇²F`` is exact in the
spectral basis and there is no spectral leakage from the forcing.

IMEX split: the linear ``−ε·∇²ψ − ξ²·∇⁴ψ`` is implicit; ``∇²(ψ³)`` and
``a·∇²F`` are explicit. The constant null mode of ``∇²`` is absorbed by a
scalar tau plus the gauge condition ``ave(psi) = psi_mean``.

Output is the same Dedalus ``FileHandler`` HDF5 layout as
Mickelin/Galewsky, so ``datagen.resample.resample_run`` can postprocess
it directly with the standard structured path.

Two-phase run: a silent ``burn_in_time`` lets the spinodal transient
finish before snapshots start, then the production window runs with the
``FileHandler`` attached.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import dedalus.public as d3
from mpi4py import MPI

from datagen.cahn_hilliard.forcing import axis_from_seed, update_forcing_field
from datagen.cahn_hilliard.ic import set_initial_conditions

logger = logging.getLogger(__name__)


def _global_psi_stats(psi_local: np.ndarray, comm) -> tuple[bool, float, float]:
    """Reduce the rank-local psi grid data to global (finite, min, max).

    Each rank only owns a colatitude slice; rank 0's slice sits near a
    pole where high-``m`` ``Y_ℓ^m`` ∝ ``sin^m(θ)`` is essentially zero,
    so a rank-0-only ``min/max`` log is misleading. This collective
    reduce gives the true global magnitude.
    """
    local_finite = bool(np.all(np.isfinite(psi_local)))
    local_min = float(np.min(psi_local)) if psi_local.size else float("inf")
    local_max = float(np.max(psi_local)) if psi_local.size else float("-inf")
    g_finite = bool(comm.allreduce(local_finite, op=MPI.LAND))
    g_min = float(comm.allreduce(local_min, op=MPI.MIN))
    g_max = float(comm.allreduce(local_max, op=MPI.MAX))
    return g_finite, g_min, g_max


def _select_dt(max_dt: float, forcing_period: float | None) -> float:
    """Conservative explicit time-step: small enough to resolve the
    forcing rotation to ~12 substeps per period, capped by ``max_dt``.
    """
    if forcing_period is None or not math.isfinite(forcing_period) or forcing_period <= 0.0:
        return float(max_dt)
    return float(min(max_dt, forcing_period / 12.0))


def run_simulation(
    params: dict,
    out_dir: Path,
    snapshot_dt: float = 10.0,
    burn_in_time: float = 200.0,
    stop_sim_time: float = 2000.0,
    Nphi: int = 256,
    Ntheta: int = 128,
    initial_dt: float = 1.0e-3,
    max_dt: float = 0.5,
    max_writes_per_file: int = 250,
) -> None:
    """Run one Cahn-Hilliard simulation and write Dedalus HDF5 snapshots.

    ``params`` keys:
      * ``epsilon``  — Krekhov control parameter (positive ⇒ ordered phase).
      * ``ell``, ``m`` — forcing harmonic indices (``m = ell`` ⇒ sectoral).
      * ``amplitude`` — forcing amplitude ``a`` in front of ``∇²F``.
      * ``omega``    — forcing rigid-rotation rate (rad / time-unit).
      * ``mean_init``, ``variance``, ``ell_init`` — IC parameters.
      * ``psi_mean`` — conserved spatial mean of ψ (gauge).
      * ``R``        — sphere radius.
      * ``seed``     — RNG seed for IC and forcing axis.

    Snapshot cadence and time arguments are in solver time units (the
    equation is dimensionless). Snapshots cover only the *production*
    window, i.e. the interval ``[burn_in_time, burn_in_time + stop_sim_time]``;
    HDF5 ``sim_time`` values for those snapshots are rebased to start at 0.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = float(params["epsilon"])
    xi = float(params["xi"])
    xi_sq = xi * xi
    ell = int(params["ell"])
    m = int(params.get("m", params["ell"]))
    a_force = float(params["amplitude"])
    omega = float(params["omega"])
    mean_init = float(params["mean_init"])
    variance = float(params["variance"])
    ell_init = int(params["ell_init"])
    psi_mean = float(params["psi_mean"])
    R = float(params["R"])
    seed = int(params["seed"])

    dtype = np.float64
    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(
        coords, (Nphi, Ntheta), radius=R, dealias=3 / 2, dtype=dtype
    )

    psi = dist.Field(name="psi", bases=basis)
    forcing = dist.Field(name="forcing", bases=basis)
    tau_psi = dist.Field(name="tau_psi")

    problem = d3.IVP([psi, tau_psi], namespace=locals())
    # CH: dt(ψ) = ∇²(−εψ + ψ³ − ξ²∇²ψ + a·F). IMEX split puts the
    # linear stiff terms (+ε·∇²ψ, +ξ²·∇⁴ψ) on the LHS so the unstable
    # band 0 < k < √ε/ξ is treated implicitly; the cubic and forcing
    # are explicit on the RHS.
    problem.add_equation(
        "dt(psi) + eps*lap(psi) + xi_sq*lap(lap(psi)) + tau_psi"
        " = lap(psi**3) + a_force*lap(forcing)"
    )
    problem.add_equation("ave(psi) = psi_mean")

    # RK443 (4-stage, 3rd-order IMEX) has a larger explicit stability
    # region than RK222 — important here because the explicit ∇²(ψ³)
    # term has eigenvalue ~3ε·k² that grows rapidly during saturation.
    solver = problem.build_solver(d3.RK443)

    set_initial_conditions(
        psi, dist, basis,
        seed=seed, mean_init=mean_init, variance=variance, ell_init=ell_init,
    )

    axis = axis_from_seed(seed)
    forcing_period = (2.0 * math.pi / (abs(omega) * max(int(m), 1))) if omega != 0.0 else None

    # Explicit CFL: the cubic ∇²(ψ³) on the RHS, at saturation
    # (ψ²~ε), acts like extra diffusion 3ε·∇² on the unstable band
    # (k²_max ~ ε/ξ²). The bound is dt ≲ ξ²/(3ε); we use a 0.2
    # safety factor (with RK443 the explicit region is generous,
    # but the cubic interfacial corrections can spike the local
    # eigenvalue well above 3ε so we stay well inside it).
    cfl_dt = 0.2 * xi_sq / (3.0 * eps)
    max_dt = float(min(max_dt, cfl_dt))

    dt = _select_dt(max_dt, forcing_period)
    initial_dt = float(min(initial_dt, dt))

    logger.info(
        "Starting CH run: Nphi=%d Ntheta=%d eps=%g xi=%g ell=%d m=%d a=%g omega=%g "
        "psi_mean=%g R=%g seed=%d burn_in=%g production=%g axis=%s dt=%g",
        Nphi, Ntheta, eps, xi, ell, m, a_force, omega,
        psi_mean, R, seed, burn_in_time, stop_sim_time,
        axis.tolist(), dt,
    )

    wallclock_start = time.time()

    # Phase 1: silent burn-in. No FileHandler attached; the spinodal
    # transient runs through with no snapshots. The forcing keeps updating
    # so the initial coarsening already feels the right pattern.
    if burn_in_time > 0.0:
        solver.stop_sim_time = burn_in_time
        update_forcing_field(forcing, dist, basis, t=solver.sim_time,
                             ell=ell, m=m, axis=axis, omega=omega)
        cur_dt = initial_dt
        try:
            while solver.proceed:
                solver.step(cur_dt)
                update_forcing_field(forcing, dist, basis, t=solver.sim_time,
                                     ell=ell, m=m, axis=axis, omega=omega)
                cur_dt = _select_dt(max_dt, forcing_period)
                if solver.iteration % 200 == 0:
                    finite, g_min, g_max = _global_psi_stats(
                        psi["g"], dist.comm_cart
                    )
                    if not finite:
                        raise RuntimeError(
                            f"Non-finite psi during burn-in at it={solver.iteration}"
                        )
                    logger.info(
                        "[burn-in] it=%d t=%.3g dt=%.3g min/max(psi)=%.3g/%.3g",
                        solver.iteration, solver.sim_time, cur_dt, g_min, g_max,
                    )
        except Exception:
            logger.exception("Burn-in failed")
            raise

    # Phase 2: production. Attach FileHandler and rebase sim_time to 0 so
    # downstream consumers see a clean ``[0, stop_sim_time]`` axis.
    solver.sim_time = 0.0
    solver.iteration = 0
    solver.stop_sim_time = stop_sim_time

    snapshots = solver.evaluator.add_file_handler(
        str(out_dir),
        sim_dt=snapshot_dt,
        max_writes=max_writes_per_file,
    )
    snapshots.add_task(psi, name="psi", layout="g")
    snapshots.add_task(forcing, name="forcing", layout="g")

    # Make sure t=0 is captured with a fresh forcing snapshot.
    update_forcing_field(forcing, dist, basis, t=solver.sim_time,
                         ell=ell, m=m, axis=axis, omega=omega)
    cur_dt = _select_dt(max_dt, forcing_period)
    try:
        while solver.proceed:
            solver.step(cur_dt)
            update_forcing_field(forcing, dist, basis, t=solver.sim_time,
                                 ell=ell, m=m, axis=axis, omega=omega)
            cur_dt = _select_dt(max_dt, forcing_period)
            if solver.iteration % 200 == 0:
                finite, g_min, g_max = _global_psi_stats(
                    psi["g"], dist.comm_cart
                )
                if not finite:
                    raise RuntimeError(
                        f"Non-finite psi at it={solver.iteration}"
                    )
                logger.info(
                    "it=%d t=%.3g dt=%.3g min/max(psi)=%.3g/%.3g",
                    solver.iteration, solver.sim_time, cur_dt, g_min, g_max,
                )
    finally:
        solver.log_stats()
        logger.info("Wallclock: %.1f s", time.time() - wallclock_start)
