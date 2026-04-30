"""Compute per-field normalization statistics from train-split zarr stores.

Reads all ``<root>/train/run_XXXX.zarr`` stores, each with a ``fields``
variable of shape ``(time, field, lat, lon)``, and writes ``<root>/stats.json``
in the field-first format expected by ``ZScoreNormalization``::

    {
      "<field_name>": {
        "mean": float,
        "std":  float,
        "mean_delta": float,
        "std_delta":  float
      },
      ...
    }

Statistics are computed via a single streaming pass (run by run) using
compensated sum aggregation, so peak memory is one run at a time.

Usage::

    uv run datagen/scripts/compute_stats.py \\
        --root /path/to/dataset \\
        --fields u_phi u_theta h vorticity
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import xarray as xr


_RUN_RE = re.compile(r"run_(\d+)\.zarr")


def _scan_zarrs(split_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in split_dir.iterdir() if _RUN_RE.fullmatch(p.name)
    )
    if not paths:
        raise FileNotFoundError(f"No run_XXXX.zarr files found in {split_dir}")
    return paths


def _update(n, mean, M2, batch: np.ndarray):
    """Welford online update for a flat array of new values."""
    for x in batch.ravel():
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    return n, mean, M2


def _combine(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    """Parallel Welford combine for two accumulators."""
    n = n_a + n_b
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
    return n, mean, M2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root containing train/ subdir.")
    ap.add_argument("--split", default="train",
                    help="Which split to compute stats over (default: train).")
    ap.add_argument("--fields", nargs="+", required=True,
                    help="Field names in channel order (e.g. u_phi u_theta h vorticity).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (defaults to <root>/stats.json).")
    args = ap.parse_args()

    split_dir = args.root / args.split
    zarr_paths = _scan_zarrs(split_dir)
    n_fields = len(args.fields)

    print(f"Computing stats over {len(zarr_paths)} runs in {split_dir}")
    print(f"Fields: {args.fields}")

    # Per-field Welford accumulators: (n, mean, M2)
    acc = [(0, 0.0, 0.0)] * n_fields
    acc_delta = [(0, 0.0, 0.0)] * n_fields

    for i, path in enumerate(zarr_paths):
        ds = xr.open_zarr(str(path), consolidated=True)
        data = ds["fields"].values  # (time, field, lat, lon)

        for f in range(n_fields):
            field_data = data[:, f, :, :]           # (time, lat, lon)
            delta_data = np.diff(field_data, axis=0) # (time-1, lat, lon)

            # Welford update using parallel combine with a per-run accumulator
            flat = field_data.ravel().astype(np.float64)
            n_r = flat.size
            mean_r = float(flat.mean())
            M2_r = float(((flat - mean_r) ** 2).sum())
            acc[f] = _combine(*acc[f], n_r, mean_r, M2_r)

            flat_d = delta_data.ravel().astype(np.float64)
            n_d = flat_d.size
            mean_d = float(flat_d.mean())
            M2_d = float(((flat_d - mean_d) ** 2).sum())
            acc_delta[f] = _combine(*acc_delta[f], n_d, mean_d, M2_d)

        if (i + 1) % 50 == 0 or (i + 1) == len(zarr_paths):
            print(f"  processed {i + 1}/{len(zarr_paths)}")

    stats: dict[str, dict] = {}
    for f, name in enumerate(args.fields):
        n, mean, M2 = acc[f]
        std = float(np.sqrt(M2 / n)) if n > 1 else 0.0
        n_d, mean_d, M2_d = acc_delta[f]
        std_d = float(np.sqrt(M2_d / n_d)) if n_d > 1 else 0.0
        stats[name] = {
            "mean":       round(float(mean), 6),
            "std":        round(float(std), 6),
            "mean_delta": round(float(mean_d), 6),
            "std_delta":  round(float(std_d), 6),
        }
        print(f"  {name:12s}  mean={mean:.4g}  std={std:.4g}"
              f"  mean_delta={mean_d:.4g}  std_delta={std_d:.4g}")

    out_path = args.out or (args.root / "stats.json")
    with open(out_path, "w") as fp:
        json.dump(stats, fp, indent=2)
        fp.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
