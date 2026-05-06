"""Extract the static cubed-sphere grid (landmask, bathymetry, lon/lat) into
a single ``grid.zarr`` at the dataset root.

The grid is identical across all 243 sweep members, so this only needs to
be run once. The output companion file is intended to live next to
``train/``, ``val/``, ``test/`` and feed dataloaders that need a landmask
or depth field.

Usage::

    uv run --project datagen python -m datagen.mitgcm.global_ocean.scripts.extract_grid \\
        --run-dir /scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean/sweep/run_0000/global_ocean_run \\
        --out /scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/global-ocean/grid.zarr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from datagen.mitgcm.global_ocean.solver import _to_face_layout
from datagen.mitgcm.mds import mds_dtype, parse_mds_meta


def _read_mds_global(run_dir: Path, name: str) -> np.ndarray:
    """Read an MDS global-tile field and reshape to (..., face, y, x)."""
    meta = parse_mds_meta(run_dir / f"{name}.meta")
    dim_list = meta["dim_list"]
    shape = tuple(int(d[0]) for d in reversed(dim_list))  # Fortran → C
    raw = np.fromfile(run_dir / f"{name}.data", dtype=mds_dtype(meta["dataprec"]))
    arr = raw.astype(np.float32, copy=False).reshape(shape)
    return _to_face_layout(arr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Any sweep run's global_ocean_run/ subdir.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output grid.zarr path.")
    ap.add_argument("--tracer-level", type=int, default=1,
                    help="MITgcm tracer level (1-indexed) to derive mask_k1 from.")
    ap.add_argument("--velocity-level", type=int, default=2,
                    help="MITgcm velocity level (1-indexed) to derive mask_k2 from.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    print(f"Reading grid from {run_dir}")

    hfac_c = _read_mds_global(run_dir, "hFacC")  # (Nr, face, y, x)
    hfac_w = _read_mds_global(run_dir, "hFacW")
    hfac_s = _read_mds_global(run_dir, "hFacS")
    depth = _read_mds_global(run_dir, "Depth")    # (face, y, x)
    xc = _read_mds_global(run_dir, "XC")
    yc = _read_mds_global(run_dir, "YC")
    angle_cs = _read_mds_global(run_dir, "AngleCS")
    angle_sn = _read_mds_global(run_dir, "AngleSN")

    k_t = args.tracer_level - 1
    k_v = args.velocity_level - 1

    # eta lives at the surface of any wet column, so its valid mask is
    # "any level has water" — slightly bigger than mask_k1 (where surface k=1
    # may be dry due to free-surface partial cells but a deeper level is wet).
    column_has_water = (hfac_c > 0).any(axis=0)

    # Per-level masks (Nr, face, y, x) for the 3-D variant. Same convention
    # as the 2-D masks: ``True`` ⇔ ocean cell (hFac > 0). For staggered grids
    # (W/S) we mask the cell only when *both* face-edges are wet, matching
    # the 2-D ``mask_k2`` convention.
    mask_c_3d = (hfac_c > 0)
    mask_w_3d = (hfac_w > 0) & (hfac_s > 0)

    ds = xr.Dataset(
        data_vars={
            "depth":     (("face", "y", "x"), depth.astype(np.float32)),
            "hfac_c_k1": (("face", "y", "x"), hfac_c[k_t].astype(np.float32)),
            "hfac_w_k2": (("face", "y", "x"), hfac_w[k_v].astype(np.float32)),
            "hfac_s_k2": (("face", "y", "x"), hfac_s[k_v].astype(np.float32)),
            "mask_k1":   (("face", "y", "x"), (hfac_c[k_t] > 0)),
            "mask_k2":   (("face", "y", "x"), (hfac_w[k_v] > 0) & (hfac_s[k_v] > 0)),
            "mask_eta":  (("face", "y", "x"), column_has_water),
            "mask_c_3d": (("level", "face", "y", "x"), mask_c_3d),
            "mask_w_3d": (("level", "face", "y", "x"), mask_w_3d),
            "hfac_c":    (("level", "face", "y", "x"), hfac_c.astype(np.float32)),
            "angle_cs":  (("face", "y", "x"), angle_cs.astype(np.float32)),
            "angle_sn":  (("face", "y", "x"), angle_sn.astype(np.float32)),
        },
        coords={
            "level": ("level", np.arange(1, hfac_c.shape[0] + 1, dtype=np.int64)),
            "xc": (("face", "y", "x"), xc.astype(np.float32)),
            "yc": (("face", "y", "x"), yc.astype(np.float32)),
        },
        attrs={
            "description":
                "Static cubed-sphere grid for the MITgcm global-ocean cs32x15 dataset.",
            "tracer_level": args.tracer_level,
            "velocity_level": args.velocity_level,
            "mask_convention": "True = ocean cell (hFac > 0)",
            "depth_units": "m (positive down; 0 = land)",
            "vector_rotation":
                "u_east = angle_cs*u - angle_sn*v;  "
                "v_north = angle_sn*u + angle_cs*v",
        },
    )

    ds.to_zarr(args.out, mode="w", consolidated=True)
    n_ocean = int(ds.mask_k1.sum())
    n_total = int(ds.mask_k1.size)
    print(f"Wrote {args.out}")
    print(f"  ocean cells (k1): {n_ocean} / {n_total}  ({100*n_ocean/n_total:.1f}%)")
    print(f"  depth range: {float(depth[depth>0].min()):.0f} .. {float(depth.max()):.0f} m")


if __name__ == "__main__":
    main()
