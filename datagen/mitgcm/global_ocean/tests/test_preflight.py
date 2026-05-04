"""Tests for the MITgcm global-ocean corner preflight helper."""

from __future__ import annotations

import json
import math

from datagen.mitgcm.global_ocean.scripts.generate_sweep import PARAM_GRID
from datagen.mitgcm.global_ocean.scripts.preflight import (
    _corner_params,
    cmd_generate,
)


def test_corner_count_and_values():
    corners = _corner_params()
    assert len(corners) == 2 ** len(PARAM_GRID)

    for corner in corners:
        for axis, values in PARAM_GRID.items():
            assert corner[axis] in {float(min(values)), float(max(values))}


def test_corners_are_unique():
    corners = _corner_params()
    keys = [tuple(sorted(corner.items())) for corner in corners]
    assert len(set(keys)) == len(corners)


def test_cmd_generate_writes_corner_configs(tmp_path):
    class Args:
        out = tmp_path / "preflight"

    rc = cmd_generate(Args())
    assert rc == 0

    configs = sorted(Args.out.glob("corner_*.json"))
    assert len(configs) == 2 ** len(PARAM_GRID)
    assert (Args.out / "corner_00.json").exists()
    assert (Args.out / f"corner_{len(configs) - 1:02d}.json").exists()

    with open(Args.out / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["n_runs"] == len(configs)
    assert len(manifest["corners"]) == len(configs)

    with open(Args.out / "corner_00.json", encoding="utf-8") as f:
        entry = json.load(f)
    assert entry["run_id"] == 0
    assert entry["run_name"] == "corner_00"
    assert "params" in entry
    assert "param_hash" in entry
