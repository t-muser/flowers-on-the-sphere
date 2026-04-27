"""Reproducible Gaussian-noise initial condition for Cahn-Hilliard on the sphere.

FiPy ships ``GaussianNoiseVariable``, but it draws from the global NumPy RNG
without exposing a seed parameter. We need bit-reproducible initial fields per
run, so this module fills a fresh ``CellVariable`` from a seeded ``Generator``
instead.
"""

from __future__ import annotations

import math

import numpy as np
from fipy import CellVariable


def gaussian_noise_field(
    mesh,
    mean: float,
    variance: float,
    seed: int,
    name: str = "phi",
) -> CellVariable:
    """Return a ``CellVariable`` filled with seeded Gaussian noise."""
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=mean, scale=math.sqrt(variance), size=mesh.numberOfCells)
    var = CellVariable(name=name, mesh=mesh)
    var.setValue(values)
    return var
