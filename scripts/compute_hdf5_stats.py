"""Compute z-score normalization stats over the train split of a spherical
HDF5 dataset (written by ``scripts/zarr_to_hdf5.py``).

Stats are read straight from the published ``.h5`` files, so they match what
``SphericalHDF5Dataset`` actually serves. Output is *field-first* and keyed by
**core field name** (the format ``ZScoreNormalization`` expects): scalar fields
get a scalar stat; vector fields get one value per component, in stored order::

    velocity:
      mean: [<u>, <v>]
      std:  [<u>, <v>]
      ...

Per field we report ``mean, std, rms, mean_delta, std_delta``. ``*_delta`` are
over consecutive-timestep differences within a trajectory (stride 1).

For leveled datasets (a ``level`` spatial dim) stats are computed **per level**
for fields that span it: a leveled scalar -> a per-level list, a leveled vector
-> a per-level list of per-component values. Surface fields (ps) and all 2-D
datasets stay reduced to a single value (per component for vectors). The
per-level layout broadcasts over the level axis at normalize time
(see SphericalHDF5Dataset).

Parallel-Welford across files (Chan's algorithm) so peak memory is ~one field
of one run per worker.

    uv run scripts/compute_hdf5_stats.py \
        --root /scicore/.../SphericalPDEs/galewsky-sw-hdf5 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import yaml

try:
    # Register the LZ4/Blosc HDF5 filters so fields written by zarr_to_hdf5.py
    # with --compression lz4 (e.g. held-suarez-clima, the ocean stores) are
    # readable here. Each worker re-imports this module, so the filter is
    # registered in every process. gzip files need nothing extra.
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

# Welford accumulator = (n, mean, M2), all float64.
_ZERO = (0, 0.0, 0.0)


def _accumulate(flat: np.ndarray, acc):
    n_r = flat.size
    if n_r == 0:
        return acc
    mean_r = float(np.add.reduce(flat, dtype=np.float64) / n_r)
    M2_r = float(np.add.reduce((flat.astype(np.float64) - mean_r) ** 2))
    return _combine(*acc, n_r, mean_r, M2_r)


def _combine(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    n = n_a + n_b
    if n == 0:
        return _ZERO
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    M2 = M2_a + M2_b + delta * delta * n_a * n_b / n
    return n, mean, M2


def _finalise(acc) -> dict:
    n, mean, M2 = acc
    var = M2 / n if n > 1 else 0.0
    std = math.sqrt(var)
    rms = math.sqrt(var + mean * mean)
    return {"mean": mean, "std": std, "rms": rms}


def _field_accumulators(arr: np.ndarray):
    """(stat_acc, delta_acc) for one field array of shape (time, *spatial).

    Streams over time to bound memory; delta is x[t+1]-x[t]."""
    acc, dacc = _ZERO, _ZERO
    prev = None
    for t in range(arr.shape[0]):
        cur = arr[t]
        acc = _accumulate(cur.reshape(-1), acc)
        if prev is not None:
            dacc = _accumulate((cur - prev).reshape(-1), dacc)
        prev = cur
    return acc, dacc


def process_file(path: str):
    """Per-file partial accumulators.

    Returns ``(fields, scalars)``. Each field maps to::

        {"per_level": bool, "is_vec": bool, "nlev": int, "ncomp": int,
         "cells": {index_tuple: (acc, dacc)}}

    one accumulator per output stat, where the index tuple is ``()`` for a
    plain scalar, ``(c,)`` per component for a 2-D vector, ``(L,)`` per level
    for a leveled scalar, and ``(L, c)`` per (level, component) for a leveled
    vector. Leveled = the dataset has a ``level`` spatial dim and the field
    actually spans it (>1); surface fields (ps, stored with a size-1 level
    placeholder) stay reduced. ``scalars`` holds the per-run constant
    ``param_*`` values (one obs/run) that ZScoreNormalization needs stats for."""
    fields: dict[str, dict] = {}
    scalars: dict[str, tuple] = {}
    with h5py.File(path, "r") as f:
        spatial_dims = [str(d) for d in f["dimensions"].attrs["spatial_dims"]]
        level_pos = spatial_dims.index("level") if "level" in spatial_dims else None
        for grp in ("t0_fields", "t1_fields", "t2_fields"):
            if grp not in f:
                continue
            for name in f[grp].attrs.get("field_names", []):
                name = str(name)
                arr = f[grp][name][0]  # (time, *spatial[, comp]); traj axis dropped
                is_vec = grp != "t0_fields"
                ncomp = arr.shape[-1] if is_vec else 1
                lax = 1 + level_pos if level_pos is not None else None  # +1 for time
                nlev = arr.shape[lax] if lax is not None else 1
                per_level = nlev > 1  # leveled field that actually spans levels
                cells: dict[tuple, tuple] = {}
                for c in range(ncomp):
                    comp = arr[..., c] if is_vec else arr  # (time, *spatial)
                    if per_level:
                        for L in range(nlev):
                            sub = np.take(comp, L, axis=lax)  # drop the level axis
                            cells[(L, c) if is_vec else (L,)] = _field_accumulators(sub)
                    else:
                        cells[(c,) if is_vec else ()] = _field_accumulators(comp)
                fields[name] = {
                    "per_level": per_level, "is_vec": is_vec,
                    "nlev": nlev if per_level else 0, "ncomp": ncomp, "cells": cells,
                }
        if "scalars" in f:
            for name in f["scalars"].attrs.get("field_names", []):
                name = str(name)
                scalars[name] = (1, float(f["scalars"][name][()]), 0.0)  # one obs/run
    return fields, scalars


def _merge(into: dict, part: dict):
    for name, fd in part.items():
        if name not in into:
            into[name] = {k: fd[k] for k in ("per_level", "is_vec", "nlev", "ncomp")}
            into[name]["cells"] = {}
        cells = into[name]["cells"]
        for key, (acc, dacc) in fd["cells"].items():
            i_acc, i_dacc = cells.get(key, (_ZERO, _ZERO))
            cells[key] = (_combine(*i_acc, *acc), _combine(*i_dacc, *dacc))


def _merge_scalars(into: dict, part: dict):
    for name, acc in part.items():
        into[name] = _combine(*into.get(name, _ZERO), *acc)


def _scalar_as_stats(acc) -> dict:
    """Constant per-run scalar: mean/std/rms over run values. Deltas are 0
    (time-invariant) and unused by the normalizer for constant scalars."""
    f = _finalise(acc)
    return {
        "mean": round(f["mean"], 6), "std": round(f["std"], 6),
        "rms": round(f["rms"], 6), "mean_delta": 0.0, "std_delta": 0.0,
    }


def _as_stats(fd: dict) -> dict:
    """Field-first stat dict whose shape follows the field: scalar -> floats;
    2-D vector -> per-component lists; leveled scalar -> per-level lists;
    leveled vector -> per-level lists of per-component (level-major)."""
    cells = fd["cells"]
    per_level, is_vec, nlev, ncomp = (
        fd["per_level"], fd["is_vec"], fd["nlev"], fd["ncomp"],
    )

    def stat(key, which):
        acc, dacc = cells[key]
        return _finalise(dacc if which == "delta" else acc)

    def assemble(extract):
        if per_level and is_vec:
            return [[extract((L, c)) for c in range(ncomp)] for L in range(nlev)]
        if per_level:
            return [extract((L,)) for L in range(nlev)]
        if is_vec:
            return [extract((c,)) for c in range(ncomp)]
        return extract(())

    r = lambda x: round(x, 6)  # noqa: E731
    return {
        "mean":       assemble(lambda k: r(stat(k, "base")["mean"])),
        "std":        assemble(lambda k: r(stat(k, "base")["std"])),
        "rms":        assemble(lambda k: r(stat(k, "base")["rms"])),
        "mean_delta": assemble(lambda k: r(stat(k, "delta")["mean"])),
        "std_delta":  assemble(lambda k: r(stat(k, "delta")["std"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="<dataset>-hdf5 root.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    train_dir = args.root / "data" / "train"
    files = sorted(str(p) for p in train_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files under {train_dir}")
    print(f"computing stats over {len(files)} train files in {train_dir}")

    merged: dict[str, list] = {}
    merged_scalars: dict[str, tuple] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_file, p): p for p in files}
        for i, fut in enumerate(as_completed(futures)):
            fields, scalars = fut.result()
            _merge(merged, fields)
            _merge_scalars(merged_scalars, scalars)
            if (i + 1) % 25 == 0 or i == len(files) - 1:
                print(f"  [{i + 1}/{len(files)}]")

    stats = {name: _as_stats(fd) for name, fd in merged.items()}
    stats.update({name: _scalar_as_stats(acc) for name, acc in merged_scalars.items()})

    (args.root / "stats.yaml").write_text(yaml.safe_dump(stats, sort_keys=False))
    (args.root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.root}/stats.yaml and stats.json\n")
    print("--- inline-ready `stats:` block ---")
    print(yaml.safe_dump(stats, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
