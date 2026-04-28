"""Cahn-Hilliard phase separation on a thin spherical shell with FiPy.

Follows the NIST FiPy ``examples/cahnHilliard/sphere.py`` recipe: a Gmsh
``Gmsh2DIn3DSpace`` mesh of the sphere, extruded radially to a thin shell so
that face-based fluxes have a well-defined orientation, with the mobility-form
Cahn-Hilliard equation

    dt(phi) = ∇·[D · a² · (1 - 6φ(1-φ)) ∇φ]  -  ∇·[D · ε² ∇(∇²φ)]

solved by an exponentially growing ``dt`` schedule.

Output is a single HDF5 file per run with a Dedalus-flavoured layout
(``/scales/sim_time``, ``/tasks/<name>``) so the downstream resampler shares
its mental model with the existing Galewsky/Mickelin pipelines, even though
``phi`` is stored as ``(Nt, Ncells)`` over the unstructured mesh rather than
``(Nt, Nphi, Ntheta)``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from fipy import (
    CellVariable,
    DiffusionTerm,
    Gmsh2DIn3DSpace,
    TransientTerm,
)

from datagen.cahn_hilliard.ic import gaussian_noise_field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """Run configuration."""

    snapshot_dt: float = 10.0
    stop_sim_time: float = 500.0
    cell_size: float = 0.3
    mesh_overlap: int = 2
    initial_dexp: float = -5.0
    max_dt: float = 100.0


@dataclass(frozen=True)
class SimulationParams:
    """Parameters of the Cahn-Hilliard simulation."""

    epsilon: float
    D: float
    a: float
    mean_init: float
    variance: float
    seed: int
    radius: float

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> SimulationParams:
        return cls(
            epsilon=float(params["epsilon"]),
            D=float(params["D"]),
            a=float(params["a"]),
            mean_init=float(params["mean_init"]),
            variance=float(params["variance"]),
            seed=int(params["seed"]),
            radius=float(params["radius"]),
        )


@dataclass(frozen=True)
class ProblemBundle:
    """A configured FiPy equation, its primary field, and the mesh."""

    eq: TransientTerm
    phi: CellVariable
    mesh: Gmsh2DIn3DSpace


# ---------------------------------------------------------------------------
# Problem assembly
# ---------------------------------------------------------------------------
def _sphere_geo(radius: float, cell_size: float) -> str:
    """Gmsh geometry string for a unit-radius spherical surface mesh.

    Mirrors the ``.geo`` body in the NIST FiPy Cahn-Hilliard sphere example.
    Six surface patches stitched together so the meshing is well-conditioned
    over the whole sphere.
    """
    return f"""
radius = {radius:g};
cellSize = {cell_size:g};

// create inner 1/8 shell
Point(1) = {{0, 0, 0, cellSize}};
Point(2) = {{-radius, 0, 0, cellSize}};
Point(3) = {{0, radius, 0, cellSize}};
Point(4) = {{0, 0, radius, cellSize}};
Circle(1) = {{2, 1, 3}};
Circle(2) = {{3, 1, 4}};
Circle(3) = {{4, 1, 2}};
Line Loop(1) = {{1, 2, 3}};
Ruled Surface(1) = {{1}};

// create remaining 7/8 inner shells
t1[] = Rotate {{{{0,0,1}},{{0,0,0}},Pi/2}}    {{Duplicata{{Surface{{1}};}}}};
t2[] = Rotate {{{{0,0,1}},{{0,0,0}},Pi}}      {{Duplicata{{Surface{{1}};}}}};
t3[] = Rotate {{{{0,0,1}},{{0,0,0}},Pi*3/2}}  {{Duplicata{{Surface{{1}};}}}};
t4[] = Rotate {{{{0,1,0}},{{0,0,0}},-Pi/2}}   {{Duplicata{{Surface{{1}};}}}};
t5[] = Rotate {{{{0,0,1}},{{0,0,0}},Pi/2}}    {{Duplicata{{Surface{{t4[0]}};}}}};
t6[] = Rotate {{{{0,0,1}},{{0,0,0}},Pi}}      {{Duplicata{{Surface{{t4[0]}};}}}};
t7[] = Rotate {{{{0,0,1}},{{0,0,0}},Pi*3/2}}  {{Duplicata{{Surface{{t4[0]}};}}}};

// create entire inner and outer shell
Surface Loop(100)={{1,t1[0],t2[0],t3[0],t7[0],t4[0],t5[0],t6[0]}};
"""


def _build_mesh(radius: float, cell_size: float, overlap: int):
    """Build the radially-extruded thin-shell mesh used by the CH example."""
    geo = _sphere_geo(radius=radius, cell_size=cell_size)
    surface = Gmsh2DIn3DSpace(geo, overlap=overlap)
    # Thin radial extrusion: 10% outward bump turns a 2D-in-3D surface into
    # a one-cell-thick shell, which is what the FiPy CH example uses so that
    # face normals are unambiguous.
    return surface.extrude(extrudeFunc=lambda r: 1.1 * r)


def run_simulation(
    params: dict,
    out_path: Path,
    snapshot_dt: float = 10.0,
    stop_sim_time: float = 500.0,
    cell_size: float = 0.3,
    initial_dexp: float = -5.0,
    max_dt: float = 100.0,
) -> None:
    """Run one Cahn-Hilliard simulation on the sphere and write a single HDF5 file.

    ``params`` keys: ``epsilon``, ``D``, ``a``, ``mean_init``, ``variance``,
    ``seed``, ``radius``. All quantities are dimensionless (the NIST example
    fixes ``D = a = epsilon = 1``); the Zarr ``time`` coordinate downstream
    is in solver time units.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon = float(params["epsilon"])
    D = float(params["D"])
    a = float(params["a"])
    mean_init = float(params["mean_init"])
    variance = float(params["variance"])
    seed = int(params["seed"])
    radius = float(params["radius"])

    logger.info(
        "Building mesh: radius=%g cell_size=%g",
        radius,
        cell_size,
    )
    mesh = _build_mesh(radius=radius, cell_size=cell_size)
    n_cells = mesh.numberOfCells
    cell_xyz = np.asarray(mesh.cellCenters.value).T  # (Ncells, 3)
    logger.info("Mesh built: %d cells", n_cells)

    phi = gaussian_noise_field(
        mesh,
        mean=mean_init,
        variance=variance,
        seed=seed,
        name="phi",
    )

    PHI = phi.arithmeticFaceValue
    eq = TransientTerm() == DiffusionTerm(
        coeff=D * a**2 * (1.0 - 6.0 * PHI * (1.0 - PHI))
    ) - DiffusionTerm(coeff=(D, epsilon**2))

    logger.info(
        "Starting CH solve: epsilon=%g D=%g a=%g mean_init=%g variance=%g "
        "seed=%d snapshot_dt=%g stop_sim_time=%g max_dt=%g",
        epsilon,
        D,
        a,
        mean_init,
        variance,
        seed,
        snapshot_dt,
        stop_sim_time,
        max_dt,
    )

    with h5py.File(out_path, mode="w") as f:
        scales = f.create_group("scales")
        tasks = f.create_group("tasks")

        scales.create_dataset("cell_xyz", data=cell_xyz)
        scales.attrs["radius"] = radius

        time_ds = scales.create_dataset(
            "sim_time",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",
        )
        phi_ds = tasks.create_dataset(
            "phi",
            shape=(0, n_cells),
            maxshape=(None, n_cells),
            chunks=(1, n_cells),
            dtype="float64",
        )

        def _append_snapshot(t: float) -> None:
            i = time_ds.shape[0]
            time_ds.resize((i + 1,))
            phi_ds.resize((i + 1, n_cells))
            time_ds[i] = t
            phi_ds[i, :] = np.asarray(phi.value)

        wallclock_start = time.time()
        elapsed = 0.0
        dexp = float(initial_dexp)
        next_snapshot_t = 0.0
        step = 0

        # Always snapshot t=0 so downstream consumers see the initial state.
        _append_snapshot(elapsed)
        next_snapshot_t += snapshot_dt

        try:
            while elapsed < stop_sim_time:
                dt = min(max_dt, math.exp(dexp))
                # Don't overshoot the stop time.
                dt = min(dt, stop_sim_time - elapsed)
                eq.solve(var=phi, dt=dt)
                elapsed += dt
                dexp += 0.01
                step += 1

                phi_arr = np.asarray(phi.value)
                if not np.all(np.isfinite(phi_arr)):
                    raise RuntimeError(
                        f"Non-finite phi at step {step} (t={elapsed:.4g})"
                    )

                if elapsed + 1.0e-12 >= next_snapshot_t:
                    _append_snapshot(elapsed)
                    next_snapshot_t += snapshot_dt

                if step % 50 == 0:
                    logger.info(
                        "step=%d t=%.4g dt=%.4g min(phi)=%.4g max(phi)=%.4g",
                        step,
                        elapsed,
                        dt,
                        float(phi_arr.min()),
                        float(phi_arr.max()),
                    )
        finally:
            logger.info(
                "Wrote %d snapshots; wallclock %.1f s",
                time_ds.shape[0],
                time.time() - wallclock_start,
            )
