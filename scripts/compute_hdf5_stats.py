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

Per field we report ``mean, std, rms, mean_delta, std_delta`` and ``log_mean /
log_std = NaN`` (placeholders; we don't log-transform). ``*_delta`` are over
consecutive-timestep differences within a trajectory (stride 1). For 3-D
(leveled) datasets the stats are reduced over the level axis too -- one value
per field, matching the single channel the field occupies.

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

    Returns ``(fields, scalars)``:
      * ``fields`` maps field name -> [(acc, dacc), ...] (one entry per
        component; scalars-as-fields have one, vectors one per channel);
      * ``scalars`` maps each per-run constant scalar (the ``param_*`` values)
        to a one-observation Welford acc. ``ZScoreNormalization`` needs a
        mean/std for every constant scalar the dataset exposes, even when it is
        not fed to the model (meta_scalars=[])."""
    fields: dict[str, list] = {}
    scalars: dict[str, tuple] = {}
    with h5py.File(path, "r") as f:
        for grp in ("t0_fields", "t1_fields", "t2_fields"):
            if grp not in f:
                continue
            for name in f[grp].attrs.get("field_names", []):
                name = str(name)
                arr = f[grp][name][0]  # drop the (size-1) trajectory axis
                if grp == "t0_fields":
                    fields[name] = [_field_accumulators(arr)]
                else:  # trailing axis enumerates components
                    fields[name] = [
                        _field_accumulators(arr[..., c])
                        for c in range(arr.shape[-1])
                    ]
        if "scalars" in f:
            for name in f["scalars"].attrs.get("field_names", []):
                name = str(name)
                scalars[name] = (1, float(f["scalars"][name][()]), 0.0)  # one obs/run
    return fields, scalars


def _merge(into: dict, part: dict):
    for name, comps in part.items():
        if name not in into:
            into[name] = [(_ZERO, _ZERO) for _ in comps]
        for c, (acc, dacc) in enumerate(comps):
            i_acc, i_dacc = into[name][c]
            into[name][c] = (_combine(*i_acc, *acc), _combine(*i_dacc, *dacc))


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


def _as_stats(comps: list) -> dict:
    """Field-first stat dict; scalar -> floats, vector -> per-component lists."""
    vec = len(comps) > 1
    base = [_finalise(acc) for acc, _ in comps]
    delta = [_finalise(dacc) for _, dacc in comps]

    def pick(vals):
        return vals if vec else vals[0]

    return {
        "mean": pick([round(b["mean"], 6) for b in base]),
        "std": pick([round(b["std"], 6) for b in base]),
        "rms": pick([round(b["rms"], 6) for b in base]),
        "log_mean": pick([float("nan")] * len(comps)),
        "log_std": pick([float("nan")] * len(comps)),
        "mean_delta": pick([round(d["mean"], 6) for d in delta]),
        "std_delta": pick([round(d["std"], 6) for d in delta]),
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

    stats = {name: _as_stats(comps) for name, comps in merged.items()}
    stats.update({name: _scalar_as_stats(acc) for name, acc in merged_scalars.items()})

    (args.root / "stats.yaml").write_text(yaml.safe_dump(stats, sort_keys=False))
    (args.root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.root}/stats.yaml and stats.json\n")
    print("--- inline-ready `stats:` block ---")
    print(yaml.safe_dump(stats, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
