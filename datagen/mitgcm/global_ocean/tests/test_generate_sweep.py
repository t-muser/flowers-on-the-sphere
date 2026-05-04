"""Tests for the MITgcm global-ocean parameter sweep generator."""

from __future__ import annotations

import json
import math

from datagen.mitgcm.global_ocean.scripts.generate_sweep import (
    FIXED_PARAMS,
    PARAM_GRID,
    _hash_params,
    iter_grid,
    main,
)


def test_total_run_count():
    expected = math.prod(len(v) for v in PARAM_GRID.values())
    assert expected == 243
    assert len(iter_grid()) == expected


def test_all_params_present_and_float():
    required = set(PARAM_GRID) | set(FIXED_PARAMS)
    for run in iter_grid():
        assert required.issubset(run)
        for key in PARAM_GRID:
            assert isinstance(run[key], float)


def test_all_grid_values_are_covered():
    runs = iter_grid()
    for axis, values in PARAM_GRID.items():
        observed = {run[axis] for run in runs}
        assert observed == {float(v) for v in values}


def test_runs_are_unique_and_hashes_are_unique():
    runs = iter_grid()
    keys = [tuple(sorted(run.items())) for run in runs]
    hashes = [_hash_params(run) for run in runs]
    assert len(set(keys)) == len(runs)
    assert len(set(hashes)) == len(runs)


def test_row_major_order():
    runs = iter_grid()
    first_axis = list(PARAM_GRID.keys())[0]
    last_axis = list(PARAM_GRID.keys())[-1]
    n_inner = math.prod(len(v) for k, v in PARAM_GRID.items() if k != first_axis)

    assert all(run[first_axis] == runs[0][first_axis] for run in runs[:n_inner])
    assert [run[last_axis] for run in runs[:3]] == list(PARAM_GRID[last_axis])


def test_main_writes_configs_and_manifest(tmp_path, monkeypatch):
    out = tmp_path / "configs"
    monkeypatch.setattr("sys.argv", ["generate_global_ocean_sweep", "--out", str(out)])

    main()

    configs = sorted(out.glob("run_*.json"))
    assert len(configs) == 243
    assert (out / "run_0000.json").exists()
    assert (out / "run_0242.json").exists()

    with open(out / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["n_runs"] == 243
    assert manifest["grid"] == {k: list(v) for k, v in PARAM_GRID.items()}
    assert len(manifest["runs"]) == 243

    with open(out / "run_0000.json", encoding="utf-8") as f:
        entry = json.load(f)
    assert entry["run_id"] == 0
    assert entry["run_name"] == "run_0000"
    assert "params" in entry
    assert "param_hash" in entry
