"""Resample one run's Dedalus HDF5 output onto a regular (lat, lon) Zarr store.

Two passes:

1. ``resample_run`` reads the native-grid Dedalus HDF5 snapshots and writes a
   ``(time, field, lat, lon)`` Zarr in the canonical (polar-aligned) frame.
2. ``apply_so3_to_zarr`` rotates the saved fields by a per-trajectory SO(3)
   tilt drawn from ``seed``. Scalars (``h``, ``vorticity``) are sampled at
   the back-rotated grid; vectors (``u_phi``, ``u_theta``) additionally get
   the local-frame Jacobian applied so components are expressed in the
   output local east/south basis.

Invocation (from the repo root so ``datagen.*`` imports resolve)::

    uv run --project datagen python -m datagen.galewsky.scripts.postprocess \\
        --raw $DATA_ROOT/raw/run_0000/ \\
        --out $DATA_ROOT/processed/run_0000.zarr \\
        --config datagen/galewsky/configs/run_0000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from datagen.galewsky.so3 import (
    back_rotated_thetaphi,
    rotation_from_seed,
    vector_jacobian,
)
from datagen.resample import resample_run


def _interp_scalar(
    field_t: np.ndarray,
    lat_in: np.ndarray,
    lon_in: np.ndarray,
    theta_target: np.ndarray,
    phi_target: np.ndarray,
) -> np.ndarray:
    """Bilinear (well, bicubic) interpolation of a scalar ``(Nlat, Nlon)``
    snapshot onto the back-rotated query grid, with periodic-in-lon wrap.

    ``lat_in`` is increasing radians; ``lon_in`` is in ``[0, 2π)``. To
    handle the ``φ = 2π`` discontinuity we pad one column at each end with
    the wrapped values.
    """
    # Pad along longitude axis for periodic wrap.
    lon_pad = np.concatenate([lon_in[-1:] - 2.0 * np.pi, lon_in, lon_in[:1] + 2.0 * np.pi])
    field_pad = np.concatenate(
        [field_t[:, -1:], field_t, field_t[:, :1]], axis=1
    )
    spline = RectBivariateSpline(lat_in, lon_pad, field_pad, kx=3, ky=3, s=0)
    lat_query = (np.pi / 2.0) - theta_target
    return spline(lat_query.ravel(), phi_target.ravel(), grid=False).reshape(
        theta_target.shape
    )


def apply_so3_to_zarr(zarr_path: Path, axis: np.ndarray, angle: float) -> None:
    """Rewrite ``zarr_path`` in place with the SO(3) tilt applied.

    Reads the existing ``(time, field, lat, lon)`` array, builds the
    back-rotated query grid once, runs cubic-spline scalar interpolation
    per snapshot, and applies the local Jacobian to the velocity pair.
    """
    with xr.open_zarr(str(zarr_path)) as src:
        ds = src.load()
    field_names = list(map(str, ds["field"].values))
    lat_deg = ds["lat"].values
    lon_deg = ds["lon"].values
    lat_in = np.deg2rad(lat_deg)
    lon_in = np.deg2rad(lon_deg)

    # Output grid in colat/lon (the same equispaced grid we just wrote).
    lat_target = lat_in
    lon_target = lon_in
    theta_in_query, phi_in_query = back_rotated_thetaphi(
        lat_target, lon_target, axis, angle,
    )

    # Output local frame uses the *output* (theta, phi); we already know it.
    lat_g, lon_g = np.meshgrid(lat_target, lon_target, indexing="ij")
    theta_out = (np.pi / 2.0) - lat_g
    phi_out = lon_g

    has_vec = ("u_phi" in field_names) and ("u_theta" in field_names)
    if has_vec:
        M = vector_jacobian(theta_out, phi_out, theta_in_query, phi_in_query,
                            axis, angle)  # (Nlat, Nlon, 2, 2)

    fields = ds["fields"].values  # (Nt, Nfield, Nlat, Nlon)
    Nt, Nf, Nlat, Nlon = fields.shape
    out = np.empty_like(fields)

    name_to_idx = {name: i for i, name in enumerate(field_names)}

    for t in range(Nt):
        # Scalar interpolation at the back-rotated points, per field.
        sampled: dict[str, np.ndarray] = {}
        for name in field_names:
            sampled[name] = _interp_scalar(
                fields[t, name_to_idx[name]],
                lat_in, lon_in, theta_in_query, phi_in_query,
            ).astype(np.float32)

        if has_vec:
            up_in = sampled["u_phi"]
            ut_in = sampled["u_theta"]
            up_out = M[..., 0, 0] * up_in + M[..., 0, 1] * ut_in
            ut_out = M[..., 1, 0] * up_in + M[..., 1, 1] * ut_in
            sampled["u_phi"] = up_out.astype(np.float32)
            sampled["u_theta"] = ut_out.astype(np.float32)

        for name in field_names:
            out[t, name_to_idx[name]] = sampled[name]

    # Overwrite the variable in the zarr store. Re-using xarray with mode='w'
    # is simplest and keeps coords/attrs intact.
    ds = ds.assign(
        fields=(("time", "field", "lat", "lon"), out),
        so3_axis_xyz=(("xyz",), np.asarray(axis, dtype=np.float64)),
        so3_angle_rad=((), np.float64(angle)),
    )

    encoding = {"fields": {"chunks": (1, Nf, Nlat, Nlon)}}
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True, encoding=encoding)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True,
                    help="Directory of Dedalus HDF5 snapshots (one run).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output Zarr store path.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional config JSON, attached as Zarr attrs.")
    ap.add_argument("--nlat", type=int, default=256)
    ap.add_argument("--nlon", type=int, default=512)
    ap.add_argument("--skip-seconds", type=float, default=4 * 86400.0,
                    help="Drop leading snapshots below this sim time and rebase "
                         "the remaining time axis to 0 (default: 4 days).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override the SO(3) seed (defaults to params['seed']).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    log = logging.getLogger("postprocess_latlon")

    # Skip silently if the raw run is missing (treated as a failed solve upstream).
    failed_marker = args.raw.with_suffix(".FAILED")
    if failed_marker.exists():
        log.warning("Run is marked FAILED (%s); skipping postprocess.", failed_marker)
        return 0

    params = None
    run_id = None
    if args.config is not None:
        with open(args.config) as f:
            entry = json.load(f)
        params = entry.get("params")
        run_id = entry.get("run_id")

    seed = args.seed
    if seed is None and params is not None:
        seed = int(params.get("seed", 0))
    elif seed is None:
        seed = 0

    log.info("Resampling %s -> %s (Nlat=%d, Nlon=%d, skip_seconds=%g)",
             args.raw, args.out, args.nlat, args.nlon, args.skip_seconds)
    resample_run(
        args.raw, args.out,
        Nlat=args.nlat, Nlon=args.nlon,
        time_offset_s=args.skip_seconds,
        params=params, run_id=run_id,
    )

    axis, angle = rotation_from_seed(seed)
    log.info("Applying SO(3) tilt: seed=%d axis=%s angle=%.6f rad",
             seed, axis.tolist(), angle)
    apply_so3_to_zarr(args.out, axis, angle)

    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
