"""Generate MITgcm MDS binary initial condition files.

MITgcm reads initial fields from "MDS" binary files: a raw big-endian
float32 array (the ``.data`` file) paired with a text header (``.meta``).
For a 3-D field of shape ``(Nr, Ny, Nx)`` the data file contains
``Nr × Ny × Nx`` big-endian float32 values in C-contiguous order, which
corresponds to the Fortran column-major layout ``A(Nx, Ny, Nr)`` (x
varies fastest). No Fortran record-length separators are written.

The MDS meta format:
    nDims = [  3 ];
    dimList = [
         Nx,  1,  Nx,
         Ny,  1,  Ny,
         Nr,  1,  Nr
    ];
    dataprec = [ 'float32' ];
    nrecords = [  1 ];
    timeStepNumber = [  0 ];

The ``dimList`` entries follow Fortran dimension order (fastest-varying
index first), so Nx is listed first even though the field is written in
C order.

Two IC files are written per run:
  - ``T.init.data`` / ``.meta`` : initial potential temperature [K]
  - ``bathyFile.bin``            : ocean depth mask (all zero = aqua-planet)

The temperature field is the latitude-dependent Held-Suarez equilibrium
potential temperature at each pressure level plus small bandlimited random
perturbations controlled by ``seed``. The perturbations are smoothed with a
Gaussian kernel to avoid seeding grid-scale noise into the spin-up, which
could alias into the resolved baroclinic eddies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from datagen.mitgcm.held_suarez._constants import KAPPA, P0
from datagen.mitgcm.held_suarez._physics import equilibrium_temperature


def pressure_thicknesses(Nr: int, p0: float = P0) -> np.ndarray:
    """MITgCM pressure-layer thicknesses [Pa], surface to top."""
    if Nr < 2:
        return np.asarray([p0], dtype=float)
    top_thickness = min(15000.0, 0.5 * p0)
    lower = (p0 - top_thickness) / (Nr - 1)
    return np.concatenate([np.full(Nr - 1, lower), [top_thickness]])


def _pressure_centers(Nr: int, p0: float = P0) -> np.ndarray:
    """Pressure at level centres [Pa] ordered as MITgCM k=1 surface to top."""
    delR = pressure_thicknesses(Nr, p0)
    upper_edges = p0 - np.concatenate([[0.0], np.cumsum(delR[:-1])])
    return upper_edges - 0.5 * delR


def _write_mds_meta(path: Path, shape: tuple[int, ...]) -> None:
    """Write an MDS ``.meta`` header file for a field of shape ``(Nr, Ny, Nx)``
    or ``(Ny, Nx)`` (2-D). ``shape`` follows C order; ``dimList`` is reversed
    to Fortran convention.
    """
    # dimList follows Fortran order: fastest-varying dimension first.
    fortran_dims = list(reversed(shape))
    dim_str = ",\n         ".join(
        f"{d:6d}, {1:6d}, {d:6d}" for d in fortran_dims
    )
    meta = f"""\
 nDims = [{len(shape):5d} ];
 dimList = [
         {dim_str}
 ];
 dataprec = [ 'float32' ];
 nrecords = [{1:5d} ];
 timeStepNumber = [{0:10d} ];
"""
    Path(path).write_text(meta)


def write_temperature_ic(
    path: Path,
    *,
    Nlon: int,
    Nlat: int,
    Nr: int,
    seed: int,
    amplitude: float = 0.1,
    smooth_sigma: float = 5.0,
    p0: float = P0,
    kappa: float = KAPPA,
) -> None:
    """Write the initial potential temperature field as an MDS binary.

    The field is a 3-D array of shape ``(Nr, Nlat, Nlon)`` [K] consisting
    of the Held-Suarez θ_eq(φ,p) field plus small bandlimited random
    perturbations.

    The reference state matches MITgCM's Held-Suarez verification setup and
    avoids starting from unrealistically warm polar columns. The perturbations
    are smoothed with a Gaussian kernel of width ``smooth_sigma`` grid points
    to avoid grid-scale seeding that could excite unresolved modes.

    Args:
        path:          Destination path (e.g., ``run_dir / "T.init.data"``).
                       The companion ``.meta`` file is written alongside.
        Nlon:          Number of longitude points.
        Nlat:          Number of latitude points.
        Nr:            Number of vertical levels (MITgCM k=1 surface to top).
        seed:          RNG seed for the perturbation (controls IC diversity).
        amplitude:     RMS amplitude of perturbation [K] before smoothing.
        smooth_sigma:  Gaussian smoothing width [grid points], applied
                       independently per level.
        p0:            Reference pressure [Pa].
        kappa:         Poisson exponent R_dry/Cp.
    """
    # Reference potential temperature field on MITgCM cell centers.
    p_centers = _pressure_centers(Nr, p0)
    del_lat = np.pi / Nlat
    lat = -0.5 * np.pi + (np.arange(Nlat) + 0.5) * del_lat
    theta_ref = equilibrium_temperature(
        lat[None, :],
        p_centers[:, None],
    )  # (Nr, Nlat)

    # Broadcast to (Nr, Nlat, Nlon).
    theta_ic = np.broadcast_to(theta_ref[:, :, None], (Nr, Nlat, Nlon)).copy()

    # Add seeded, bandlimited perturbations.
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((Nr, Nlat, Nlon)).astype(np.float64) * amplitude
    for k in range(Nr):
        # Wrap in longitude for periodic smoothing; latitude uses reflect.
        noise[k] = gaussian_filter(noise[k], sigma=smooth_sigma,
                                   mode=["reflect", "wrap"])
    theta_ic += noise

    # Write MDS binary: big-endian float32, C-contiguous = Fortran (x-fastest).
    path = Path(path)
    theta_ic.astype(">f4").tofile(str(path))
    _write_mds_meta(path.with_suffix(".meta"), shape=(Nr, Nlat, Nlon))


def write_bathymetry(path: Path, *, Nlon: int, Nlat: int) -> None:
    """Write the bathymetry file as an MDS binary (all zeros = aqua-planet).

    MITgcm uses negative depth values for ocean; zero means land/atmosphere.
    For the Held-Suarez dry atmospheric configuration, zero everywhere is
    the correct flat-surface aqua-planet setup.

    Args:
        path: Destination path (e.g., ``run_dir / "bathyFile.bin"``).
    """
    path = Path(path)
    bathy = np.zeros((Nlat, Nlon), dtype=">f4")
    bathy.tofile(str(path))
    _write_mds_meta(path.with_suffix(".meta"), shape=(Nlat, Nlon))
