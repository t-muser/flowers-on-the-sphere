"""Tests for ``datagen/cpl_aim_ocn/staging.py``.

All tests are pure-Python (no MITgcm binary or MPI required) — they
build a synthetic ``inputs/`` tree under ``tmp_path`` and a stub
executable layout, then assert that ``stage_run`` produces the right
rank-dir structure and that ``mpmd_command`` builds a sane argv.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_staging.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datagen.cpl_aim_ocn import staging
from datagen.cpl_aim_ocn.staging import (
    COMPONENTS,
    LAYOUTS,
    SUCCESS_MARKER,
    check_run_completed,
    layout_for,
    mpmd_command,
    stage_run,
)


# ─── New: per-component log-marker fields ────────────────────────────────────

class TestRankLayoutLogFields:
    def test_atm_and_ocn_use_stdout_with_ended_normally(self):
        for layout in LAYOUTS:
            if layout.component in ("atm", "ocn"):
                assert layout.log_filename == "STDOUT.0000"
                assert layout.log_marker == SUCCESS_MARKER

    def test_cpl_uses_coupler_clog_with_no_marker(self):
        cpl = layout_for("cpl")
        assert cpl.log_filename == "Coupler.0000.clog"
        # An empty marker means "existence + non-empty" check, used
        # because pkg/atm_ocn_coupler/coupler.F ends with a bare STOP.
        assert cpl.log_marker == ""


# ─── Layout invariants ───────────────────────────────────────────────────────

class TestLayoutInvariants:
    def test_three_components_in_canonical_order(self):
        # Order is significant — it must match the colon-separated
        # mpirun MPMD form (rank 0 = cpl, rank 1 = ocn, rank 2 = atm).
        assert COMPONENTS == ("cpl", "ocn", "atm")
        assert tuple(l.component for l in LAYOUTS) == COMPONENTS

    def test_rank_indices_are_0_1_2(self):
        assert [l.rank_idx for l in LAYOUTS] == [0, 1, 2]

    def test_rank_dirname_format(self):
        assert [l.rank_dirname for l in LAYOUTS] == ["rank_0", "rank_1", "rank_2"]

    def test_cpl_pulls_only_cpl_inputs(self):
        # The coupler does not need atm/ocn forcing or grid files.
        assert layout_for("cpl").input_subdirs == ("cpl",)

    def test_ocn_pulls_ocn_and_grid(self):
        # The ocean reads its own forcing AND the shared cs32 grid.
        assert set(layout_for("ocn").input_subdirs) == {"ocn", "grid"}

    def test_atm_pulls_atm_and_grid(self):
        assert set(layout_for("atm").input_subdirs) == {"atm", "grid"}

    def test_layout_for_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown component"):
            layout_for("seaice")


# ─── Synthetic inputs/ tree fixture ──────────────────────────────────────────

@pytest.fixture
def fake_inputs(tmp_path: Path) -> Path:
    """Build a minimal ``inputs/`` tree with one or two stub binary files
    per subdir, enough to exercise the symlinking logic without depending
    on a real MITgcm checkout."""
    root = tmp_path / "inputs"
    files = {
        "atm":  ["albedo_cs32.bin", "topo.cpl_FM.bin", "seaSurfT.cpl_FM.bin"],
        "ocn":  ["bathy_Hmin50.bin", "lev_T_cs_15k.bin",
                 "pickup.0000072000", "pickup.0000072000.meta"],
        "grid": [f"grid_cs32.face00{i}.bin" for i in range(1, 7)],
        "cpl":  ["RA.bin", "runOff_cs32_3644.bin"],
    }
    for sub, names in files.items():
        d = root / sub
        d.mkdir(parents=True)
        for n in names:
            (d / n).write_bytes(n.encode())
    return root


# ─── stage_run: layout + symlink behaviour ───────────────────────────────────

class TestStageRun:
    def test_creates_three_rank_dirs(self, tmp_path: Path, fake_inputs: Path):
        run_dir = tmp_path / "run"
        rdirs = stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        assert set(rdirs) == {"cpl", "ocn", "atm"}
        assert (run_dir / "rank_0").is_dir()
        assert (run_dir / "rank_1").is_dir()
        assert (run_dir / "rank_2").is_dir()

    def test_returns_dirs_match_layout(self, tmp_path: Path, fake_inputs: Path):
        run_dir = tmp_path / "run"
        rdirs = stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        assert rdirs["cpl"] == run_dir / "rank_0"
        assert rdirs["ocn"] == run_dir / "rank_1"
        assert rdirs["atm"] == run_dir / "rank_2"

    def test_cpl_symlinks_only_cpl_inputs(self, tmp_path: Path, fake_inputs: Path):
        # rank_0 must NOT contain bathy or atm-side files — only cpl.
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        names = sorted(p.name for p in (run_dir / "rank_0").iterdir())
        assert names == ["RA.bin", "runOff_cs32_3644.bin"]

    def test_ocn_symlinks_ocn_and_grid(self, tmp_path: Path, fake_inputs: Path):
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        names = set(p.name for p in (run_dir / "rank_1").iterdir())
        # ocn binaries
        assert "bathy_Hmin50.bin" in names
        assert "lev_T_cs_15k.bin" in names
        # ocn pickup pair
        assert "pickup.0000072000" in names
        assert "pickup.0000072000.meta" in names
        # shared grid
        for i in range(1, 7):
            assert f"grid_cs32.face00{i}.bin" in names

    def test_atm_symlinks_atm_and_grid(self, tmp_path: Path, fake_inputs: Path):
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        names = set(p.name for p in (run_dir / "rank_2").iterdir())
        assert "albedo_cs32.bin" in names
        assert "topo.cpl_FM.bin" in names
        for i in range(1, 7):
            assert f"grid_cs32.face00{i}.bin" in names

    def test_atm_does_not_get_ocn_pickup(self, tmp_path: Path, fake_inputs: Path):
        # Pickup belongs in rank_1 only.
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        names = set(p.name for p in (run_dir / "rank_2").iterdir())
        assert "pickup.0000072000" not in names

    def test_files_are_symlinks_not_copies(self, tmp_path: Path, fake_inputs: Path):
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        for f in (run_dir / "rank_0").iterdir():
            assert f.is_symlink(), f"{f} should be a symlink"

    def test_symlinks_resolve_to_inputs_dir(self, tmp_path: Path, fake_inputs: Path):
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        target = (run_dir / "rank_0" / "RA.bin").resolve()
        assert target == (fake_inputs / "cpl" / "RA.bin").resolve()

    def test_symlinks_use_absolute_paths(self, tmp_path: Path, fake_inputs: Path):
        # Absolute targets so the rank dir survives being moved.
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        link = run_dir / "rank_0" / "RA.bin"
        assert Path(link.readlink()).is_absolute()

    def test_writes_namelist_files_with_provided_content(
        self, tmp_path: Path, fake_inputs: Path
    ):
        run_dir = tmp_path / "run"
        stage_run(
            run_dir,
            namelists={
                "cpl": {"data.cpl": "&CPL_NML\n cpl_atmSendFrq = 3600.,\n&\n"},
                "ocn": {"data": "&PARM01\n nIter0 = 0,\n&"},
                "atm": {"eedata": "&EEPARMS\n useCoupler=.TRUE.,\n&"},
            },
            inputs_root=fake_inputs,
        )
        assert (run_dir / "rank_0" / "data.cpl").read_text().startswith("&CPL_NML")
        assert "nIter0 = 0" in (run_dir / "rank_1" / "data").read_text()
        assert "useCoupler" in (run_dir / "rank_2" / "eedata").read_text()

    def test_omitted_components_get_no_namelists(
        self, tmp_path: Path, fake_inputs: Path
    ):
        run_dir = tmp_path / "run"
        # Only supply for atm; cpl and ocn rank dirs should still exist
        # and be populated with input symlinks, but contain no namelists.
        stage_run(
            run_dir,
            namelists={"atm": {"data": "abc"}},
            inputs_root=fake_inputs,
        )
        assert (run_dir / "rank_2" / "data").read_text() == "abc"
        # rank_0 and rank_1 contain symlinks only
        assert not (run_dir / "rank_0" / "data").exists()
        assert not (run_dir / "rank_1" / "data").exists()

    def test_unknown_component_in_namelists_raises(
        self, tmp_path: Path, fake_inputs: Path
    ):
        run_dir = tmp_path / "run"
        with pytest.raises(ValueError, match="Unknown component"):
            stage_run(
                run_dir,
                namelists={"seaice": {"data": "x"}},
                inputs_root=fake_inputs,
            )

    def test_idempotent_replays_overwrite_namelists(
        self, tmp_path: Path, fake_inputs: Path
    ):
        # Phase-1 → phase-2 use case: same run_dir, second call must
        # overwrite namelist files with new contents (and not error).
        run_dir = tmp_path / "run"
        stage_run(
            run_dir,
            namelists={"atm": {"data": "phase1"}},
            inputs_root=fake_inputs,
        )
        stage_run(
            run_dir,
            namelists={"atm": {"data": "phase2"}},
            inputs_root=fake_inputs,
        )
        assert (run_dir / "rank_2" / "data").read_text() == "phase2"

    def test_idempotent_replays_refresh_symlinks(
        self, tmp_path: Path, fake_inputs: Path
    ):
        # If an upstream input file is replaced, a re-stage must update
        # the symlink target. (Symlinks themselves don't change, but the
        # logic should not error on pre-existing symlinks.)
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        # Re-stage; should not raise even though all symlinks already exist.
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        # And the target still resolves correctly.
        target = (run_dir / "rank_0" / "RA.bin").resolve()
        assert target.is_file()

    def test_missing_input_subdir_is_clear_error(self, tmp_path: Path):
        # No fake_inputs created — simulates running before build.py.
        run_dir = tmp_path / "run"
        empty_inputs = tmp_path / "empty_inputs"
        empty_inputs.mkdir()
        with pytest.raises(FileNotFoundError, match="Input source dir"):
            stage_run(run_dir, namelists={}, inputs_root=empty_inputs)


# ─── mpmd_command ────────────────────────────────────────────────────────────

@pytest.fixture
def fake_run(tmp_path: Path, fake_inputs: Path) -> tuple[Path, dict[str, Path]]:
    """Stand up a run dir + fake build_*/mitgcmuv files. Returns
    ``(run_dir, build_dirs)`` ready for mpmd_command."""
    run_dir = tmp_path / "run"
    stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
    build_dirs = {}
    for c in COMPONENTS:
        d = tmp_path / f"build_{c}"
        d.mkdir()
        exe = d / "mitgcmuv"
        exe.write_bytes(b"#!/bin/sh\n")
        exe.chmod(0o755)
        build_dirs[c] = d
    return run_dir, build_dirs


class TestMpmdCommand:
    def test_returns_three_colon_separated_segments(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        run_dir, build_dirs = fake_run
        cmd = mpmd_command(run_dir, build_dirs=build_dirs)
        # Two ":" separators between three executable specs.
        assert cmd.count(":") == 2

    def test_first_executable_is_cpl(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        # Order matters: rank 0 → coupler.
        run_dir, build_dirs = fake_run
        cmd = mpmd_command(run_dir, build_dirs=build_dirs)
        # Find the first executable path in the argv.
        first_exe = next(s for s in cmd if s.endswith("mitgcmuv"))
        assert "build_cpl" in first_exe

    def test_default_one_rank_per_component(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        run_dir, build_dirs = fake_run
        cmd = mpmd_command(run_dir, build_dirs=build_dirs)
        # Three "-np 1" pairs.
        assert cmd.count("-np") == 3
        # Indices of "-np" tokens; the next token is the rank count.
        np_positions = [i for i, s in enumerate(cmd) if s == "-np"]
        assert all(cmd[i + 1] == "1" for i in np_positions)

    def test_atm_ocn_ranks_can_be_overridden(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        run_dir, build_dirs = fake_run
        cmd = mpmd_command(
            run_dir, build_dirs=build_dirs,
            n_mpi={"cpl": 1, "ocn": 4, "atm": 2},
        )
        # Find the rank counts in order.
        np_values = [cmd[i + 1] for i, s in enumerate(cmd) if s == "-np"]
        assert np_values == ["1", "4", "2"]

    def test_cpl_rank_count_must_be_1(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        run_dir, build_dirs = fake_run
        with pytest.raises(ValueError, match="cpl"):
            mpmd_command(run_dir, build_dirs=build_dirs,
                         n_mpi={"cpl": 2, "ocn": 1, "atm": 1})

    def test_executable_paths_are_absolute(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        run_dir, build_dirs = fake_run
        cmd = mpmd_command(run_dir, build_dirs=build_dirs)
        for s in cmd:
            if s.endswith("mitgcmuv"):
                assert Path(s).is_absolute(), s

    def test_missing_rank_dir_raises(
        self, fake_run: tuple[Path, dict[str, Path]], tmp_path: Path
    ):
        run_dir, build_dirs = fake_run
        # Wipe one rank dir.
        import shutil
        shutil.rmtree(run_dir / "rank_1")
        with pytest.raises(FileNotFoundError, match="rank_1"):
            mpmd_command(run_dir, build_dirs=build_dirs)

    def test_missing_executable_raises(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        run_dir, build_dirs = fake_run
        (build_dirs["atm"] / "mitgcmuv").unlink()
        with pytest.raises(FileNotFoundError, match="atm"):
            mpmd_command(run_dir, build_dirs=build_dirs)

    def test_custom_mpirun_name(
        self, fake_run: tuple[Path, dict[str, Path]]
    ):
        run_dir, build_dirs = fake_run
        cmd = mpmd_command(run_dir, build_dirs=build_dirs, mpirun="srun")
        assert cmd[0] == "srun"


# ─── check_run_completed ─────────────────────────────────────────────────────

class TestCheckRunCompleted:
    def _write_log(self, run_dir: Path, layout, content: str) -> None:
        """Write `content` to the layout's expected log file."""
        (run_dir / layout.rank_dirname / layout.log_filename).write_text(content)

    def test_all_normal_with_per_component_markers(
        self, tmp_path: Path, fake_inputs: Path
    ):
        # Each layout's success criterion is satisfied: cpl gets a
        # non-empty Coupler.0000.clog, atm/ocn get STDOUT.0000 with the
        # "ended Normally" substring.
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        for layout in LAYOUTS:
            if layout.log_marker:
                self._write_log(run_dir, layout, f"… {layout.log_marker} …\n")
            else:
                self._write_log(run_dir, layout, "started\nfinished\n")
        result = check_run_completed(run_dir)
        assert result == {"cpl": True, "ocn": True, "atm": True}

    def test_cpl_log_existence_only(self, tmp_path: Path, fake_inputs: Path):
        # The coupler has log_marker == "" — non-empty file is enough.
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        (run_dir / "rank_0" / "Coupler.0000.clog").write_text("anything\n")
        result = check_run_completed(run_dir)
        assert result["cpl"] is True

    def test_cpl_empty_log_reported_as_false(
        self, tmp_path: Path, fake_inputs: Path
    ):
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        (run_dir / "rank_0" / "Coupler.0000.clog").write_text("")
        assert check_run_completed(run_dir)["cpl"] is False

    def test_missing_log_reported_as_false(
        self, tmp_path: Path, fake_inputs: Path
    ):
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        # Only write the coupler log; atm/ocn STDOUT.0000 absent.
        (run_dir / "rank_0" / "Coupler.0000.clog").write_text("ok\n")
        result = check_run_completed(run_dir)
        assert result == {"cpl": True, "ocn": False, "atm": False}

    def test_atm_stdout_without_marker_reported_as_false(
        self, tmp_path: Path, fake_inputs: Path
    ):
        run_dir = tmp_path / "run"
        stage_run(run_dir, namelists={}, inputs_root=fake_inputs)
        (run_dir / "rank_2" / "STDOUT.0000").write_text("crashed early")
        assert check_run_completed(run_dir)["atm"] is False

    def test_layouts_have_consistent_log_metadata(self):
        # The two non-coupler layouts must declare a textual marker;
        # the coupler must declare the empty marker (existence check).
        by_comp = {l.component: l for l in LAYOUTS}
        assert by_comp["cpl"].log_marker == "", (
            "Coupler binary has no end-of-run marker; layout must use "
            "log_marker=''"
        )
        assert by_comp["cpl"].log_filename == "Coupler.0000.clog"
        for c in ("atm", "ocn"):
            assert by_comp[c].log_filename == "STDOUT.0000"
            assert by_comp[c].log_marker == SUCCESS_MARKER
