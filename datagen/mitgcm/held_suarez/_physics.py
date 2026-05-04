"""Pure-Python implementation of the Held-Suarez (1994) atmospheric forcing.

This module implements the three components of the idealized GCM forcing:

1. Equilibrium potential temperature ``θ_eq(φ, p)``
2. Newtonian cooling rate ``k_T(φ, σ)``
3. Rayleigh friction rate ``k_v(σ)``

These Python functions serve two roles:
- They are the reference against which the Fortran implementation in
  ``datagen/mitgcm/held_suarez/code/apply_forcing.F`` is validated.
- They are directly tested in
  ``datagen/mitgcm/held_suarez/tests/test_held_suarez.py``.

All arguments use physical SI units (Pa for pressure, radians for latitude,
seconds for timescales). The forcing follows Held & Suarez (1994), J. Atmos.
Sci., 51, 1825–1835.
"""

from __future__ import annotations

import numpy as np

from datagen.mitgcm.held_suarez._constants import (
    KAPPA,
    P0,
    HS_KA,
    HS_KF,
    HS_KS,
    HS_DELTA_T_Y,
    HS_DELTA_T_Z,
    HS_SIGMAB,
    HS_T0,
    HS_T_MIN,
)


def equilibrium_temperature(
    lat: np.ndarray,
    p: np.ndarray,
    *,
    T0: float = HS_T0,
    delta_T_y: float = HS_DELTA_T_Y,
    delta_theta_z: float = HS_DELTA_T_Z,
    kappa: float = KAPPA,
    p0: float = P0,
    T_min: float = HS_T_MIN,
) -> np.ndarray:
    """Held-Suarez equilibrium POTENTIAL temperature θ_eq(φ, p).

    From Held & Suarez (1994) Eq. (3)::

        θ_eq = max(T_min·(p0/p)^κ,
                   T0 − ΔTy·sin²φ − Δθz·ln(p/p0)·cos²φ)

    The max with ``T_min·(p0/p)^κ`` prevents unphysically cold temperatures
    in the stratosphere. At the equatorial surface both branches agree when
    ``T0 − Δθz·ln(1) = T0``.

    Args:
        lat: Latitude in radians, any broadcastable shape.
        p:   Pressure in Pa, any broadcastable shape.

    Returns:
        θ_eq in Kelvin, same broadcast shape as ``lat`` and ``p``.
    """
    lat = np.asarray(lat, dtype=float)
    p = np.asarray(p, dtype=float)

    sin2 = np.sin(lat) ** 2
    cos2 = np.cos(lat) ** 2
    log_p = np.log(p / p0)

    theta_eq_uncapped = T0 - delta_T_y * sin2 - delta_theta_z * log_p * cos2
    theta_min = T_min * (p0 / p) ** kappa
    return np.maximum(theta_min, theta_eq_uncapped)


def equilibrium_temperature_K(
    lat: np.ndarray,
    p: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Equilibrium TEMPERATURE T_eq = θ_eq · (p/p0)^κ in Kelvin."""
    theta_eq = equilibrium_temperature(lat, p, **kwargs)
    p = np.asarray(p, dtype=float)
    kappa = kwargs.get("kappa", KAPPA)
    p0 = kwargs.get("p0", P0)
    return theta_eq * (p / p0) ** kappa


def rayleigh_friction_rate(
    sigma: np.ndarray,
    *,
    kf: float = HS_KF,
    sigmab: float = HS_SIGMAB,
) -> np.ndarray:
    """Held-Suarez Rayleigh surface-friction rate k_v(σ) [1/s].

    From Held & Suarez (1994) Eq. (1)::

        k_v(σ) = k_f · max(0, (σ − σ_b) / (1 − σ_b))

    This is non-zero only within the boundary layer σ > σ_b.

    Args:
        sigma: Normalised pressure σ = p/p_s, any shape.

    Returns:
        k_v in 1/s, same shape as ``sigma``.
    """
    sigma = np.asarray(sigma, dtype=float)
    return kf * np.maximum(0.0, (sigma - sigmab) / (1.0 - sigmab))


def newtonian_cooling_rate(
    lat: np.ndarray,
    sigma: np.ndarray,
    *,
    ka: float = HS_KA,
    ks: float = HS_KS,
    sigmab: float = HS_SIGMAB,
) -> np.ndarray:
    """Held-Suarez Newtonian cooling rate k_T(φ, σ) [1/s].

    From Held & Suarez (1994) Eq. (4)::

        k_T(φ, σ) = k_a + (k_s − k_a) · max(0, (σ − σ_b)/(1 − σ_b)) · cos⁴φ

    This blends from the free-atmosphere rate k_a aloft to the fast surface
    rate k_s in the warm subtropical boundary layer.

    Args:
        lat:   Latitude in radians, any broadcastable shape.
        sigma: Normalised pressure σ = p/p_s, broadcastable with ``lat``.

    Returns:
        k_T in 1/s, broadcast shape of ``lat`` and ``sigma``.
    """
    lat = np.asarray(lat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    boundary_fraction = np.maximum(0.0, (sigma - sigmab) / (1.0 - sigmab))
    return ka + (ks - ka) * boundary_fraction * np.cos(lat) ** 4


def reference_temperature_profile(
    p_centers: np.ndarray,
    *,
    T0: float = HS_T0,
    delta_theta_z: float = HS_DELTA_T_Z,
    kappa: float = KAPPA,
    p0: float = P0,
    T_min: float = HS_T_MIN,
) -> np.ndarray:
    """Reference temperature T_ref(p) [K] used for MITgcm's tRef array.

    Evaluates the Held-Suarez equilibrium temperature at the equator (φ=0),
    which gives the warmest possible profile. Used to initialise MITgcm's
    hydrostatic reference state (tRef in &PARM01).

    Args:
        p_centers: Pressure at cell centres [Pa], shape (Nr,).

    Returns:
        T_ref in Kelvin, shape (Nr,).
    """
    lat_eq = np.zeros_like(p_centers)
    return equilibrium_temperature_K(lat_eq, p_centers, T0=T0,
                                     delta_theta_z=delta_theta_z,
                                     kappa=kappa, p0=p0, T_min=T_min)
