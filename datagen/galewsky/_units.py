"""Sim-unit conversion factors.

Inside the Dedalus solver we run in a scaled unit system where the Earth
radius is 1 and one time unit equals one hour. This matches the canonical
Dedalus shallow-water example and keeps the numerical values entering the
sparse matrices O(1), which matters for linear-solve conditioning:
``R = 6.37e6`` metres, ``Omega = 7.3e-5 rad/s`` and ``g = 9.8 m/s²`` differ
by eleven orders of magnitude in SI, and mixing them in the same matrix
eats most of the float64 dynamic range.

Physical → sim: multiply lengths by ``METER`` and divide times by
``SECOND`` (equivalently, multiply times by ``SECOND⁻¹``). Velocities are
``METER / SECOND``, accelerations ``METER / SECOND²``, and so on.

All user-facing interfaces (CLI flags, JSON configs, Zarr ``time`` coord)
stay in physical SI units. These factors are used only inside the solver
and the HDF5 → Zarr resampler.
"""

from __future__ import annotations

# Reference scales
R_EARTH_PHYS = 6.37122e6   # metres
HOUR_PHYS = 3600.0         # seconds per hour

# Conversion factors: multiply a physical SI quantity by these to get the
# corresponding sim-unit number.
METER = 1.0 / R_EARTH_PHYS
SECOND = 1.0 / HOUR_PHYS
