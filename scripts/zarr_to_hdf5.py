"""Convert ``<src>/{train,val,test}/run_*.zarr`` to a Well-style HDF5 layout
tailored for spherical PDE data.

One ``run_*.zarr`` (a single trajectory) -> one ``run_*.h5`` with
``n_trajectories = 1``, written to ``<dst>/data/<split>/`` with the split
renamed ``val -> valid``.

Design (see DATASET_SPECS):
  * Horizontal velocity / momentum components are stored as a single
    multi-component VECTOR field (t1), not separate scalars -- on a sphere
    (u, v) is a tangent vector and must transform together under rotation.
  * Vertical levels are a THIRD spatial dim: spatial = (level, lat, lon).
    Surface-only fields (ps, eta) set ``dim_varying[level]=False`` so the
    loader broadcasts them across levels.
  * Everything else is a SCALAR field (t0).
  * No boundary_conditions group: a closed sphere has no boundary, and a BC
    across vertical levels is meaningless. The fots loader tolerates this.

Group layout per file:
  root attrs : dataset_name, grid_type, n_spatial_dims, n_trajectories,
               simulation_parameters (+ each param as an attr)
  dimensions/: time, [level,] lat, lon   (each: sample_varying=False)
  scalars/   : constant per-run parameters (field_names attr)
  t0_fields/ : scalar fields            (field_names attr)
  t1_fields/ : vector fields            (field_names attr; trailing comp axis)
  t2_fields/ : empty (field_names = [])

Postprocess-only (no PDE solve). Small 2-D datasets are fine on the login
node; run held-suarez (multi-GB/run) via sbatch.

    uv run scripts/zarr_to_hdf5.py --dataset shock-caps \
        --src-root /scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs \
        --dst-root /scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs \
        --limit-per-split 5 --workers 4
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}

# Per-dataset field schema. ``scalars``/``vectors`` name the *prognostic*
# fields to keep; ``drop`` lists diagnostic fields to skip. ``level_dim`` set
# means levels become a third spatial dim (held-suarez, ocean-3D). Vectors map
# a field name -> ordered list of its component fields.
DATASET_SPECS: dict[str, dict] = {
    "mickelin-gns": dict(
        name="mickelin_gns", scalars=["vorticity"], vectors={}, drop=[],
    ),
    "cahn-hilliard-sphere": dict(
        name="cahn_hilliard", scalars=["phi"], vectors={}, drop=[],
    ),
    "galewsky-sw": dict(
        name="galewsky_sw", scalars=["h"],
        vectors={"velocity": ["u_phi", "u_theta"]},
        drop=["vorticity"],  # diagnostic: recoverable from velocity
    ),
    "shock-caps": dict(
        name="shock_caps", scalars=["height"],
        vectors={"momentum": ["momentum_u", "momentum_v"]}, drop=[],
    ),
    "held-suarez-clima": dict(
        name="held_suarez_clima", level_dim="level",
        scalars=["T", "ps"],  # ps is surface-only -> broadcast across levels
        vectors={"velocity": ["u", "v"]}, drop=[],
    ),
}


def _get_field(ds: xr.Dataset, name: str) -> xr.DataArray:
    """Fetch a named field from either the stacked ``fields(time,field,...)``
    convention or a plain per-variable dataset."""
    if "fields" in ds.data_vars and "field" in ds["fields"].dims:
        return ds["fields"].sel(field=name)
    return ds[name]


def _spatial_dims(spec: dict) -> list[str]:
    return [spec["level_dim"], "lat", "lon"] if spec.get("level_dim") else ["lat", "lon"]


def _store_axes(da: xr.DataArray, spatial: list[str]) -> tuple[np.ndarray, list[bool]]:
    """Transpose ``da`` to (time, *spatial) and return the array plus a
    ``dim_varying`` flag per spatial dim (False where the field omits it).

    Spatial dims the field does not span are kept as a SIZE-1 placeholder axis,
    not dropped: the loader's ``_pad_axes`` tiles that axis up to full size
    (e.g. surface ``ps`` broadcast across vertical levels)."""
    present = [d for d in spatial if d in da.dims]
    arr = da.transpose("time", *present).values.astype(np.float32)
    dim_varying = [d in da.dims for d in spatial]
    for i, d in enumerate(spatial):
        if d not in da.dims:
            arr = np.expand_dims(arr, axis=1 + i)  # +1 for the leading time axis
    return arr, dim_varying


def convert_one(src: Path, dst: Path, spec: dict, grid_type: str) -> tuple[Path, str]:
    if dst.exists():
        return src, "skip(exists)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(str(src)).load()

    spatial = _spatial_dims(spec)
    coord_arrays = {d: ds[d].values for d in spatial}
    tvals = ds["time"].values.astype(np.float64)
    params = {k[len("param_"):]: v for k, v in ds.attrs.items() if k.startswith("param_")}

    tmp = dst.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["dataset_name"] = spec["name"]
        f.attrs["grid_type"] = grid_type
        f.attrs["n_spatial_dims"] = len(spatial)
        f.attrs["n_trajectories"] = 1
        f.attrs["simulation_parameters"] = list(params.keys())

        g = f.create_group("dimensions")
        g.attrs["spatial_dims"] = spatial
        for key, val in [("time", tvals)] + [(d, coord_arrays[d]) for d in spatial]:
            g.create_dataset(key, data=np.asarray(val))
            g[key].attrs["sample_varying"] = False

        g = f.create_group("scalars")
        g.attrs["field_names"] = list(params.keys())
        for key, val in params.items():
            g.create_dataset(key, data=np.asarray(val, dtype=np.float64))
            g[key].attrs["time_varying"] = False
            g[key].attrs["sample_varying"] = False

        # t0 scalar fields
        g = f.create_group("t0_fields")
        g.attrs["field_names"] = list(spec["scalars"])
        for name in spec["scalars"]:
            arr, dim_varying = _store_axes(_get_field(ds, name), spatial)
            dset = g.create_dataset(name, data=arr[None], compression="gzip", compression_opts=4)
            dset.attrs["dim_varying"] = dim_varying
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = True

        # t1 vector fields: stack components on a trailing axis
        g = f.create_group("t1_fields")
        g.attrs["field_names"] = list(spec["vectors"].keys())
        for vname, comps in spec["vectors"].items():
            stacked = [_store_axes(_get_field(ds, c), spatial) for c in comps]
            arr = np.stack([a for a, _ in stacked], axis=-1)  # (time, *spatial, C)
            dim_varying = stacked[0][1]
            dset = g.create_dataset(vname, data=arr[None], compression="gzip", compression_opts=4)
            dset.attrs["dim_varying"] = dim_varying
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = True
            # Per-component channel labels. The Well assumes a rank-1 field has
            # n_spatial_dims components; ours are genuine k-component tangent
            # vectors (k=2 even on a 3-D level/lat/lon grid), so we name the
            # components explicitly and the loader counts channels from this.
            dset.attrs["component_names"] = list(comps)

        f.create_group("t2_fields").attrs["field_names"] = []

    tmp.rename(dst)
    return src, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--dst-root", type=Path, required=True, help="<name>-hdf5/ is created under here.")
    ap.add_argument("--grid-type", default="spherical")
    ap.add_argument("--limit-per-split", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    spec = DATASET_SPECS[args.dataset]
    src = args.src_root / args.dataset
    dst = args.dst_root / f"{args.dataset}-hdf5"

    jobs: list[tuple[Path, Path]] = []
    for split, well_split in SPLIT_MAP.items():
        stores = sorted((src / split).glob("run_*.zarr"))
        if args.limit_per_split is not None:
            stores = stores[: args.limit_per_split]
        for store in stores:
            jobs.append((store, dst / "data" / well_split / store.name.replace(".zarr", ".h5")))

    print(f"converting {len(jobs)} stores: {src} -> {dst}/data  (fields: "
          f"scalars={spec['scalars']} vectors={list(spec['vectors'])} drop={spec.get('drop', [])})")
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_one, s, d, spec, args.grid_type): s for s, d in jobs}
        for i, fut in enumerate(as_completed(futures)):
            try:
                s, status = fut.result()
            except Exception as exc:
                s, status, n_err = futures[fut], f"ERR {exc!r}", n_err + 1
            else:
                n_ok += status == "ok"
                n_skip += status.startswith("skip")
                n_err += status.startswith("ERR")
            print(f"[{i + 1:4d}/{len(jobs)}] {s.name:18s} {status}")
    print(f"done: ok={n_ok} skip={n_skip} err={n_err} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
