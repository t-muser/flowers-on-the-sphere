"""Tests for ``datagen/cpl_aim_ocn/scripts/generate_sweep.py``.

Verifies the 4-axis parameter grid (CO2 × solar × GM-Redi κ × seed),
deterministic hashing, manifest layout, and CLI behaviour. No MITgcm
binary or MPI required.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_generate_sweep.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from datagen.cpl_aim_ocn.scripts.generate_sweep import (
    FIXED_PARAMS,
    PARAM_GRID,
    _hash_params,
    iter_grid,
    main,
)


# ─── Parameter grid structure ────────────────────────────────────────────────

class TestIterGrid:
    def test_total_run_count(self):
        """4 × 3 × 3 × 5 = 180."""
        expected = math.prod(len(v) for v in PARAM_GRID.values())
        assert expected == 180
        assert len(iter_grid()) == 180

    def test_all_params_present_in_each_run(self):
        required = set(PARAM_GRID.keys()) | set(FIXED_PARAMS.keys())
        for run in iter_grid():
            assert required.issubset(run.keys()), (
                f"Missing keys: {required - run.keys()}"
            )

    def test_seed_stored_as_int(self):
        for run in iter_grid():
            assert isinstance(run["seed"], int)

    def test_physical_params_stored_as_float(self):
        physical_keys = ("co2_ppm", "solar_scale", "gm_kappa")
        for run in iter_grid():
            for k in physical_keys:
                assert isinstance(run[k], float), \
                    f"{k} should be float in run, got {type(run[k]).__name__}"

    def test_fixed_params_present_with_correct_values(self):
        for run in iter_grid():
            for k, v in FIXED_PARAMS.items():
                assert run[k] == v

    def test_all_grid_values_appear_at_least_once(self):
        runs = iter_grid()
        for axis, values in PARAM_GRID.items():
            seen = {run[axis] for run in runs}
            for v in values:
                target = int(v) if axis == "seed" else float(v)
                assert target in seen, (
                    f"value {v} never appeared on axis {axis}"
                )

    def test_all_run_param_dicts_unique(self):
        runs = iter_grid()
        seen = set()
        for run in runs:
            key = tuple(sorted(run.items()))
            assert key not in seen, f"duplicate run params: {run}"
            seen.add(key)

    def test_row_major_seed_varies_fastest(self):
        # First five rows differ only in seed (and all share the
        # CO2/solar/κ tuple of the slowest axes' first values).
        runs = iter_grid()
        first_five = runs[:5]
        for r in first_five:
            assert r["co2_ppm"] == PARAM_GRID["co2_ppm"][0]
            assert r["solar_scale"] == PARAM_GRID["solar_scale"][0]
            assert r["gm_kappa"] == PARAM_GRID["gm_kappa"][0]
        assert [r["seed"] for r in first_five] == list(PARAM_GRID["seed"])


# ─── _hash_params ────────────────────────────────────────────────────────────

class TestHashParams:
    def test_deterministic(self):
        d = {"co2_ppm": 348.0, "solar_scale": 1.00, "gm_kappa": 1000.0,
             "seed": 0}
        assert _hash_params(d) == _hash_params(d)

    def test_order_independent(self):
        a = {"co2_ppm": 348.0, "solar_scale": 1.00, "gm_kappa": 1000.0,
             "seed": 0}
        b = {"seed": 0, "gm_kappa": 1000.0, "solar_scale": 1.00,
             "co2_ppm": 348.0}
        assert _hash_params(a) == _hash_params(b)

    def test_value_change_changes_hash(self):
        a = {"co2_ppm": 348.0, "seed": 0}
        b = {"co2_ppm": 348.5, "seed": 0}
        assert _hash_params(a) != _hash_params(b)

    def test_length_is_12(self):
        assert len(_hash_params({"x": 1})) == 12

    def test_all_180_hashes_unique(self):
        hashes = {_hash_params(r) for r in iter_grid()}
        assert len(hashes) == 180


# ─── CLI / main() ────────────────────────────────────────────────────────────

class TestMain:
    def test_writes_180_per_run_files(self, tmp_path: Path):
        main(["--out", str(tmp_path)])
        run_files = sorted(tmp_path.glob("run_*.json"))
        assert len(run_files) == 180

    def test_run_filename_format(self, tmp_path: Path):
        main(["--out", str(tmp_path)])
        names = [p.name for p in sorted(tmp_path.glob("run_*.json"))]
        # zero-padded to 4 digits
        assert names[0]   == "run_0000.json"
        assert names[179] == "run_0179.json"

    def test_each_run_file_has_required_top_level_keys(self, tmp_path: Path):
        main(["--out", str(tmp_path)])
        with open(tmp_path / "run_0000.json") as f:
            entry = json.load(f)
        for k in ("run_id", "run_name", "params", "param_hash"):
            assert k in entry

    def test_run_ids_are_sequential(self, tmp_path: Path):
        main(["--out", str(tmp_path)])
        run_files = sorted(tmp_path.glob("run_*.json"))
        for i, rf in enumerate(run_files):
            with open(rf) as f:
                assert json.load(f)["run_id"] == i

    def test_manifest_structure(self, tmp_path: Path):
        main(["--out", str(tmp_path)])
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["n_runs"] == 180
        assert set(manifest["grid"]) == set(PARAM_GRID.keys())
        assert manifest["fixed"] == FIXED_PARAMS
        assert len(manifest["runs"]) == 180

    def test_manifest_path_overridable(self, tmp_path: Path):
        custom = tmp_path / "custom_manifest.json"
        main(["--out", str(tmp_path), "--manifest", str(custom)])
        assert custom.is_file()
        # And the default location is NOT written.
        assert not (tmp_path / "manifest.json").is_file()
