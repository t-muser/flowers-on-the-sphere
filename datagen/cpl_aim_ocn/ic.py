"""Initial-condition file generation for the coupled AIM atmosphere.

The atmospheric IC is the only knob driven by the per-ensemble-member
``seed``: a small Gaussian-smoothed perturbation around the upstream
reference theta profile, written as an MDS binary that the atm reads
via the ``hydrogThetaFile`` namelist parameter.

MDS layout for cs32
-------------------
The cs32 cubed-sphere has six 32×32 panels. MITgcm's ``exch2`` package
unfolds them along the X axis at I/O time, so the on-disk layout for a
3-D atmospheric field is ``(Nr=5, Ny=32, Nx=192)`` in C-major. The six
faces sit side-by-side along Nx: panel ``f`` occupies columns
``f*32 … (f+1)*32 − 1``. The companion ``.meta`` file lists the
dimensions in Fortran order (fastest-varying first), so it reads
``Nx, Ny, Nr``.

Precision
---------
**64-bit big-endian floats** (``>f8``). The upstream verification's
``input_atm/data`` PARM01 sets ``readBinaryPrec=64,
writeBinaryPrec=64``, and every shipped ``.bin`` forcing file
(``topo.cpl_FM.bin``, etc.) is double precision. Writing the IC as
float32 silently mismatches the read precision and crashes at startup
with no clean diagnostic — the precision must match.

Reference profile
-----------------
Built into a constant ``THETA_REF_K`` here, taken verbatim from the
upstream ``verification/cpl_aim+ocn/input_atm/data`` ``tRef`` line
(289.6, 298.1, 314.5, 335.8, 437.4 K). The MITgcm atmospheric pressure-
coord convention orders levels from k=1 at the bottom (highest pressure,
lowest θ) to k=Nr at the top (lowest pressure, highest θ); the upstream
values describe a stably stratified atmosphere with surface θ ≈ 290 K
and stratospheric θ ≈ 437 K.

Perturbation
------------
A bandlimited zero-mean noise field of RMS amplitude ``amplitude`` K
(default 0.1 K), generated independently per face with a 2-D Gaussian
filter of width ``smooth_sigma`` grid points (default 3) — small enough
to give every cs cell its own perturbation, large enough that the
noise is smooth on the within-face grid. The filter is applied per
face only, never across face boundaries, to avoid spurious gradients
at the cube edges where the local x/y axes of neighbouring faces
disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

# ─── Reference theta profile (upstream verbatim) ─────────────────────────────

#: Upstream cs32 atm reference potential-temperature profile [K], k=1
#: (bottom, ~925 hPa) → k=5 (top, ~50 hPa). Source: verification/
#: ``cpl_aim+ocn/input_atm/data`` line ``tRef = 289.6, 298.1, 314.5, 335.8, 437.4,``.
THETA_REF_K: tuple[float, ...] = (289.6, 298.1, 314.5, 335.8, 437.4)

#: cs32 grid size constants (per-face and unfolded global dimensions).
N_FACE: int = 6
FACE_NX: int = 32
FACE_NY: int = 32
NX_GLOBAL: int = N_FACE * FACE_NX   # = 192
NY_GLOBAL: int = FACE_NY            # = 32
NR_ATM: int = len(THETA_REF_K)      # = 5


# ─── Internal helpers ────────────────────────────────────────────────────────

def _gen_per_face_noise(
    seed: int, *,
    n_levels: int = NR_ATM,
    amplitude: float = 0.1,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    """Build the (Nr, n_face, face_ny, face_nx) noise tensor.

    Each (level, face) panel gets its own noise field, smoothed in-place
    with a 2-D Gaussian filter so that the on-grid result is bandlimited
    but still has spatial structure. The post-smoothing field is then
    rescaled to the requested RMS ``amplitude`` to make ``amplitude``
    interpretable as the on-disk noise level (the smoothing reduces
    variance, so without rescaling the actual amplitude is below
    nominal).
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_levels, N_FACE, FACE_NY, FACE_NX))
    smooth = np.empty_like(raw)
    for k in range(n_levels):
        for f in range(N_FACE):
            smooth[k, f] = gaussian_filter(
                raw[k, f], sigma=smooth_sigma, mode="reflect",
            )
    # Rescale per-(level, face) to give RMS = amplitude. Per-panel
    # rescaling means each face contributes equally to the IC variance,
    # which is what we want for a uniformly-distributed seed.
    eps = 1e-12
    rms_per_panel = np.sqrt(np.mean(smooth ** 2, axis=(2, 3),
                                    keepdims=True)) + eps
    return (smooth / rms_per_panel * amplitude).astype(np.float64)


def _faces_to_unfolded(panels: np.ndarray) -> np.ndarray:
    """Reshape ``(Nr, n_face, face_ny, face_nx)`` → ``(Nr, Ny, Nx)``.

    cs32 unfolds along Nx (six panels concatenated side-by-side), so
    panel ``f`` occupies columns ``f*32 : (f+1)*32`` in the result.
    """
    nr, nf, ny, nx = panels.shape
    if nf != N_FACE or ny != FACE_NY or nx != FACE_NX:
        raise ValueError(
            f"Expected shape (Nr, {N_FACE}, {FACE_NY}, {FACE_NX}); got {panels.shape}"
        )
    # (Nr, n_face, face_ny, face_nx) → (Nr, face_ny, n_face, face_nx) → (Nr, Ny, Nx)
    return panels.swapaxes(1, 2).reshape(nr, ny, nf * nx)


def _write_mds_meta(path: Path, shape: Sequence[int]) -> None:
    """Write the MDS ``.meta`` companion for a field of C-order ``shape``.

    ``dimList`` is emitted in Fortran (fastest-varying first) order, so
    a 3-D field of C-shape ``(Nr, Ny, Nx)`` produces ``Nx, Ny, Nr``.
    """
    fortran_dims = list(reversed(shape))
    dim_str = ",\n         ".join(
        f"{d:6d}, {1:6d}, {d:6d}" for d in fortran_dims
    )
    meta = f"""\
 nDims = [{len(shape):5d} ];
 dimList = [
         {dim_str}
 ];
 dataprec = [ 'float64' ];
 nrecords = [{1:5d} ];
 timeStepNumber = [{0:10d} ];
"""
    Path(path).write_text(meta)


# ─── Public API ──────────────────────────────────────────────────────────────

def write_atm_theta_ic(
    path: Path,
    *,
    seed: int,
    amplitude: float = 0.1,
    smooth_sigma: float = 3.0,
    theta_ref: Sequence[float] = THETA_REF_K,
) -> None:
    """Write the seeded atmospheric θ initial-condition file.

    Output layout
    -------------
    On disk the field has shape ``(Nr=5, Ny=32, Nx=192)`` in C-major,
    big-endian float32. The companion ``.meta`` file is written next to
    ``path`` (replacing any ``.bin``/``.data`` extension).

    Args:
        path:          Destination ``.data`` (or ``.bin``) path.
        seed:          RNG seed — fully determines the perturbation field.
                       Two distinct seeds give different but
                       statistically-equivalent IC perturbations.
        amplitude:     Per-(level, face) RMS amplitude of the
                       perturbation [K]. Default 0.1 K — small enough
                       that the model is in dynamical balance but large
                       enough to grow under baroclinic instability over
                       a multi-week spin-up.
        smooth_sigma:  Gaussian smoothing width [grid points], applied
                       per-face. Smaller = more grid-scale variance;
                       larger = smoother fields. Default 3.
        theta_ref:     Reference θ profile [K], length ``Nr``. Defaults
                       to the upstream cs32 atm template's ``tRef``.
    """
    if len(theta_ref) != NR_ATM:
        raise ValueError(
            f"theta_ref must have length {NR_ATM}; got {len(theta_ref)}"
        )

    # Per-face noise on the (Nr, 6, 32, 32) grid.
    noise = _gen_per_face_noise(
        seed=seed, n_levels=NR_ATM,
        amplitude=amplitude, smooth_sigma=smooth_sigma,
    )

    # Reference profile broadcast to per-face shape, then add noise.
    ref = np.asarray(theta_ref, dtype=np.float64).reshape(NR_ATM, 1, 1, 1)
    field = (np.broadcast_to(ref, (NR_ATM, N_FACE, FACE_NY, FACE_NX))
             + noise).astype(np.float64)

    # Unfold faces → (Nr, Ny, Nx).
    unfolded = _faces_to_unfolded(field)

    # Write big-endian float64 (matches readBinaryPrec=64 set in the
    # upstream atm `data` namelist), then the .meta companion.
    path = Path(path)
    unfolded.astype(">f8").tofile(str(path))
    meta_path = path.with_suffix(".meta")
    _write_mds_meta(meta_path, shape=unfolded.shape)
