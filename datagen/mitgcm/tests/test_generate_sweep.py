"""Tests for the parameter sweep generator.

Verifies the parameter grid structure, run naming, hash stability, and
manifest content. No MITgcm binary required.

Run::

    uv run --project datagen pytest datagen/mitgcm/tests/test_generate_sweep.py -v
"""

from __future__ import annotations

import json
import math

import pytest

from datagen.mitgcm.scripts.generate_sweep import (
    FIXED_PARAMS,
    PARAM_GRID,
    _hash_params,
    iter_grid,
    main,
)


# ─── Parameter grid structure ────────────────────────────────────────────────

class TestIterGrid:
    def test_total_run_count(self):
        """162 = 3 × 3 × 3 × 2 × 3 runs."""
        expected = math.prod(len(v) for v in PARAM_GRID.values())
        assert expected == 162
        assert len(iter_grid()) == 162

    def test_all_params_present_in_each_run(self):
        """Every run dict must contain all PARAM_GRID axes and FIXED_PARAMS."""
        required = set(PARAM_GRID.keys()) | set(FIXED_PARAMS.keys())
        for run in iter_grid():
            assert required.issubset(run.keys()), (
                f"Missing keys: {required - run.keys()}"
            )

    def test_seed_is_int(self):
        """Seed should be stored as int, not float."""
        for run in iter_grid():
            assert isinstance(run["seed"], int)

    def test_physical_params_are_float(self):
        """Non-seed physical parameters should be stored as float."""
        float_keys = [k for k in PARAM_GRID if k != "seed"]
        for run in iter_grid():
            for k in float_keys:
                assert isinstance(run[k], float), (
                    f"{k}={run[k]!r} should be float"
                )

    def test_fixed_params_propagate(self):
        """FIXED_PARAMS values should appear unchanged in every run."""
        for run in iter_grid():
            for k, v in FIXED_PARAMS.items():
                assert run[k] == v

    def test_all_param_grid_values_covered(self):
        """Every value in PARAM_GRID must appear in at least one run."""
        runs = iter_grid()
        for axis, values in PARAM_GRID.items():
            observed = {run[axis] for run in runs}
            for v in values:
                val = int(v) if axis == "seed" else float(v)
                assert val in observed, f"{axis}={v} not found in any run"

    def test_runs_are_unique(self):
        """No two runs should have identical parameter combinations."""
        runs = iter_grid()
        # Use frozenset of items for set uniqueness.
        seen = set()
        for run in runs:
            key = tuple(sorted(run.items()))
            assert key not in seen, f"Duplicate run: {run}"
            seen.add(key)

    def test_row_major_order(self):
        """First axis (tau_drag_days) varies slowest; seed varies fastest."""
        runs = iter_grid()
        first_axis = list(PARAM_GRID.keys())[0]   # tau_drag_days
        last_axis  = list(PARAM_GRID.keys())[-1]  # seed
        first_values = [r[first_axis] for r in runs]
        last_values  = [r[last_axis]  for r in runs]

        # First axis should be constant for the first product-of-others runs.
        n_inner = math.prod(len(v) for k, v in PARAM_GRID.items()
                            if k != first_axis)
        assert all(v == first_values[0] for v in first_values[:n_inner]), (
            "First axis should be constant over the first inner-product block"
        )
        # Last axis should cycle through all values each inner block.
        seed_values = list(PARAM_GRID[last_axis])
        n_seeds = len(seed_values)
        observed_seed_cycle = [int(v) for v in last_values[:n_seeds]]
        expected_seed_cycle = [int(v) for v in seed_values]
        assert observed_seed_cycle == expected_seed_cycle


# ─── _hash_params ────────────────────────────────────────────────────────────

class TestHashParams:
    def test_deterministic(self):
        params = {"a": 1.0, "b": 2.0}
        assert _hash_params(params) == _hash_params(params)

    def test_different_params_different_hash(self):
        p1 = {"tau_drag_days": 1.0, "delta_T_y": 60.0, "seed": 0}
        p2 = {"tau_drag_days": 2.0, "delta_T_y": 60.0, "seed": 0}
        assert _hash_params(p1) != _hash_params(p2)

    def test_hash_length_12(self):
        params = {"x": 1.0}
        assert len(_hash_params(params)) == 12

    def test_all_grid_hashes_unique(self):
        """Every run in the tensor product should have a distinct hash."""
        runs = iter_grid()
        hashes = [_hash_params(r) for r in runs]
        assert len(set(hashes)) == len(hashes), (
            "Hash collision detected in parameter grid"
        )

    def test_order_independent(self):
        """Hash should be the same regardless of key insertion order."""
        p1 = {"a": 1.0, "b": 2.0}
        p2 = {"b": 2.0, "a": 1.0}
        assert _hash_params(p1) == _hash_params(p2)


# ─── main (file I/O) ─────────────────────────────────────────────────────────

class TestMain:
    def test_creates_correct_number_of_configs(self, tmp_path):
        import sys
        sys.argv = ["generate_sweep", "--out", str(tmp_path / "configs")]
        main()
        configs = list((tmp_path / "configs").glob("run_*.json"))
        assert len(configs) == 162

    def test_run_naming_pattern(self, tmp_path):
        """Run files must be named run_0000.json … run_0161.json."""
        import sys
        sys.argv = ["generate_sweep", "--out", str(tmp_path / "configs")]
        main()
        cfg_dir = tmp_path / "configs"
        for i in range(162):
            assert (cfg_dir / f"run_{i:04d}.json").exists()

    def test_manifest_written(self, tmp_path):
        import sys
        sys.argv = ["generate_sweep", "--out", str(tmp_path / "configs")]
        main()
        assert (tmp_path / "configs" / "manifest.json").exists()

    def test_manifest_structure(self, tmp_path):
        import sys
        sys.argv = ["generate_sweep", "--out", str(tmp_path / "configs")]
        main()
        with open(tmp_path / "configs" / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["n_runs"] == 162
        assert "grid" in manifest
        assert "runs" in manifest
        assert len(manifest["runs"]) == 162

    def test_run_config_is_valid_json(self, tmp_path):
        import sys
        sys.argv = ["generate_sweep", "--out", str(tmp_path / "configs")]
        main()
        with open(tmp_path / "configs" / "run_0000.json") as f:
            entry = json.load(f)
        assert "run_id"     in entry
        assert "run_name"   in entry
        assert "params"     in entry
        assert "param_hash" in entry
        assert entry["run_id"]   == 0
        assert entry["run_name"] == "run_0000"

    def test_manifest_run_ids_sequential(self, tmp_path):
        import sys
        sys.argv = ["generate_sweep", "--out", str(tmp_path / "configs")]
        main()
        with open(tmp_path / "configs" / "manifest.json") as f:
            manifest = json.load(f)
        ids = [r["run_id"] for r in manifest["runs"]]
        assert ids == list(range(162))

    def test_custom_manifest_path(self, tmp_path):
        import sys
        manifest_path = tmp_path / "my_manifest.json"
        sys.argv = [
            "generate_sweep",
            "--out", str(tmp_path / "configs"),
            "--manifest", str(manifest_path),
        ]
        main()
        assert manifest_path.exists()
