"""Initial conditions for the Galewsky et al. (2004) barotropic-instability test.

Two public entry points:

- ``jet_zonal_profile(lat, u_max, lat_center_deg, width_deg)`` — compact
  analytic jet profile between ``lat_center - width/2`` and
  ``lat_center + width/2``, reaching peak ``u_max`` at jet center.
- ``set_initial_conditions(u, h, params, coords, dist, basis, g, R, Omega)`` —
  build a *bi-hemispheric* zonal flow (the canonical northern jet plus a
  mirrored southern jet at ``-lat_center``), solve for the geostrophically
  and cyclostrophically balanced height ``h``, then add the Galewsky
  bi-Gaussian height bump *only on the northern jet*. The asymmetric
  perturbation makes cross-equatorial Rossby propagation the dominant
  signal that distinguishes trajectories.

The simulation always runs in the canonical frame (rotation axis = polar
axis); the per-trajectory SO(3) tilt is applied at postprocess time by
``datagen.galewsky.scripts.postprocess`` so the Coriolis term in the solver
stays valid.

Conventions: Dedalus ``S2Coordinates('phi', 'theta')`` uses colatitude
``theta ∈ (0, π)`` (0 at the north pole) and longitude ``phi ∈ [0, 2π)``.
Latitude is ``lat = π/2 - theta``. The ``u`` vector components are
``(u_phi, u_theta)`` = (eastward, southward).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def jet_zonal_profile(
    lat: np.ndarray,
    u_max: float,
    lat_center_deg: float,
    width_deg: float = 40.0,
) -> np.ndarray:
    """Galewsky compact-support zonal jet profile in latitude (radians).

    ``u(lat) = (u_max / e_n) * exp(1 / ((lat - lat0)(lat - lat1)))`` between
    ``lat0`` and ``lat1``, zero outside. ``e_n`` is chosen so the peak value
    equals ``u_max``.
    """
    lat_center = np.deg2rad(lat_center_deg)
    half_width = np.deg2rad(width_deg) / 2.0
    lat0 = lat_center - half_width
    lat1 = lat_center + half_width

    u = np.zeros_like(lat)
    inside = (lat > lat0) & (lat < lat1)
    en = np.exp(-4.0 / (lat1 - lat0) ** 2)
    arg = 1.0 / ((lat[inside] - lat0) * (lat[inside] - lat1))
    u[inside] = (u_max / en) * np.exp(arg)
    return u


def two_jet_zonal_profile(
    lat: np.ndarray,
    u_max: float,
    lat_center_deg: float,
    width_deg: float = 40.0,
) -> np.ndarray:
    """Sum of the canonical northern jet at ``+lat_center`` and a mirrored
    southern jet at ``-lat_center``.

    Both compact-support jets use ``jet_zonal_profile``; their supports are
    disjoint as long as ``lat_center > width/2`` (true for every grid point
    in the sweep).
    """
    return (
        jet_zonal_profile(lat, u_max, +lat_center_deg, width_deg=width_deg)
        + jet_zonal_profile(lat, u_max, -lat_center_deg, width_deg=width_deg)
    )


def height_perturbation(
    phi: np.ndarray,
    lat: np.ndarray,
    h_hat: float,
    lon_c_deg: float,
    lat_c_deg: float,
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 15.0,
) -> np.ndarray:
    """Galewsky bi-Gaussian height perturbation.

    ``h'(lat, lon) = h_hat * cos(lat) * exp(-((lon - lon_c)/beta)^2)
                                      * exp(-((lat_c - lat)/alpha)^2)``

    ``phi`` and ``lat`` are broadcastable grids of longitude and latitude
    (radians). ``beta`` sets the zonal width, ``alpha`` the meridional width.
    """
    lam_c = np.deg2rad(lon_c_deg)
    lat_c = np.deg2rad(lat_c_deg)
    dlon = (phi - lam_c + np.pi) % (2.0 * np.pi) - np.pi
    return (
        h_hat
        * np.cos(lat)
        * np.exp(-((dlon / beta) ** 2))
        * np.exp(-(((lat_c - lat) / alpha) ** 2))
    )


def balanced_height_profile(
    u_max: float,
    lat_center_deg: float,
    g: float,
    R: float,
    Omega: float,
    width_deg: float = 40.0,
    n_samples: int = 4096,
):
    """Zonally-balanced height profile ``h(lat)`` as a 1-D interpolant.

    Integrates the zonal-flow geostrophic + cyclostrophic balance

        dh/dlat = -(1/g) * u(lat) * (2·Ω·R·sin(lat) + u(lat)·tan(lat))

    from the south pole upward and subtracts the spherical area-mean so the
    result has ``<h> = 0``. The returned ``interp1d`` is cheap to evaluate at
    any set of target latitudes (radians).

    This follows the canonical Galewsky (2004) initial-condition recipe used
    in the Dedalus shallow-water example: integrate the 1-D ODE on a fine
    uniform latitude grid with scipy, then evaluate at the native solver
    grid. Avoids the sphere-Laplacian nullspace issue that makes a direct
    LBVP formulation singular.
    """
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_samples)
    u = two_jet_zonal_profile(lat, u_max, lat_center_deg, width_deg=width_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = -(1.0 / g) * u * (2.0 * Omega * R * np.sin(lat) + u * np.tan(lat))
    # u vanishes at the poles, so `u * tan(lat)` is 0/0 there; patch NaNs.
    integrand = np.where(np.isfinite(integrand), integrand, 0.0)
    h = np.concatenate([[0.0], cumulative_trapezoid(integrand, lat)])
    # Subtract spherical area-mean so <h> = 0 (weight by cos(lat)).
    w = np.cos(lat)
    mean_h = np.trapezoid(h * w, lat) / np.trapezoid(w, lat)
    h -= mean_h
    return interp1d(
        lat, h, kind="cubic",
        bounds_error=False, fill_value=(float(h[0]), float(h[-1])),
    )


def set_initial_conditions(
    u,
    h,
    params: dict,
    coords,
    dist,
    basis,
    g: float,
    R: float,
    Omega: float,
) -> None:
    """Populate ``u`` (VectorField) and ``h`` (Field) with the Galewsky IC.

    The balanced height is computed by 1-D numerical integration of the
    zonal-balance ODE in latitude (see ``balanced_height_profile``) and
    normalized so ``<h> = 0``, so the total depth ``H + h`` has area-mean
    equal to the ``H`` parameter.
    """
    u_max = float(params["u_max"])
    lat_center = float(params["lat_center"])
    h_hat = float(params["h_hat"])
    # Asymmetric perturbation: pin the bi-Gaussian bump at lon_c=0 in the
    # canonical frame. The per-trajectory SO(3) rotation applied at
    # postprocess time supplies the rotational diversity that lon_c used to
    # provide in the previous (single-jet) sweep.
    lon_c = 0.0

    phi, theta = dist.local_grids(basis)
    lat = np.pi / 2.0 - theta

    u["g"][0] = two_jet_zonal_profile(lat, u_max, lat_center)
    u["g"][1] = 0.0

    h_profile = balanced_height_profile(u_max, lat_center, g, R, Omega)
    h["g"][...] = h_profile(lat)
    h["g"] += height_perturbation(phi, lat, h_hat, lon_c, lat_center)
