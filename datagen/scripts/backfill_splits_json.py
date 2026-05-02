"""Write a minimal ``splits.json`` reflecting on-disk train/val/test layout.

Some older datasets shipped with ``train/``, ``val/``, ``test/`` directories
populated but no ``splits.json``. The data loader (``ZarrDataModule``) prefers
``splits.json`` because directory scanning will pick up partially-written
shards from solver blow-ups (see shock-caps run_0340/run_0459). This script
writes an explicit ``splits.json`` mirroring whatever is currently on disk,
so the loader has a definitive list of valid runs.

Existing ``splits.json`` files are left untouched. Pass ``--force`` to
overwrite.

Usage::

    uv run datagen/scripts/backfill_splits_json.py --root /path/to/dataset
    uv run datagen/scripts/backfill_splits_json.py --all  # all datasets under SphericalPDEs/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_RUN_RE = re.compile(r"run_(\d+)\.zarr")
_DEFAULT_PARENT = Path("/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs")


def _scan(split_dir: Path) -> list[int]:
    if not split_dir.is_dir():
        return []
    return sorted(int(m.group(1)) for p in split_dir.iterdir()
                  if (m := _RUN_RE.fullmatch(p.name)))


def backfill(root: Path, force: bool = False) -> Path | None:
    out = root / "splits.json"
    if out.exists() and not force:
        print(f"  skip {out} (already exists)")
        return None
    splits = {s: _scan(root / s) for s in ("train", "val", "test")}
    counts = {s: len(v) for s, v in splits.items()}
    if sum(counts.values()) == 0:
        print(f"  skip {root} (no train/val/test runs on disk yet)")
        return None
    payload = {
        "strategy": "on-disk-backfill",
        "counts": counts,
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"  wrote {out}: counts={counts}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None,
                    help="Single dataset root containing train/val/test/.")
    ap.add_argument("--all", action="store_true",
                    help=f"Backfill every dataset under {_DEFAULT_PARENT}.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing splits.json.")
    args = ap.parse_args()

    if args.all == bool(args.root):
        ap.error("Pass exactly one of --root or --all.")

    roots = sorted(p for p in _DEFAULT_PARENT.iterdir() if p.is_dir()) if args.all else [args.root]
    for r in roots:
        print(f"== {r} ==")
        backfill(r, force=args.force)


if __name__ == "__main__":
    main()
