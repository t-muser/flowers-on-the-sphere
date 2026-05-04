"""Tests for ``datagen/cpl_aim_ocn/scripts/build.py``.

These tests are pure-Python (no MITgcm checkout required) — they
exercise the staging logic, helper functions, and CLI plumbing using a
fake MITgcm tree built inside ``tmp_path``. The actual three-binary
compilation is not exercised here (it needs a real toolchain) but is
covered by the manual smoke-build documented in the package README.

Run::

    uv run --project datagen pytest datagen/cpl_aim_ocn/tests/test_build.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from datagen.cpl_aim_ocn.scripts import build as build_mod
from datagen.cpl_aim_ocn.scripts.build import (
    COMPONENTS,
    _SIBLING_DIRS,
    _git_head,
    _sha1_file,
    main,
    stage_inputs,
    write_build_info,
)


# ─── Static-layout invariants ────────────────────────────────────────────────

class TestPackageLayout:
    """The vendored upstream files must be in the expected places."""

    def test_three_components_named(self):
        assert set(COMPONENTS) == {"atm", "ocn", "cpl"}

    def test_build_dirs_have_genmake_local(self):
        for comp in COMPONENTS:
            path = build_mod._BUILD_DIRS[comp] / "genmake_local"
            assert path.is_file(), f"missing {path} — vendor it from upstream"

    def test_build_atm_genmake_local_references_correct_mods(self):
        text = (build_mod._BUILD_DIRS["atm"] / "genmake_local").read_text()
        assert "../code_atm" in text
        assert "../shared_code" in text

    def test_build_ocn_genmake_local_references_correct_mods(self):
        text = (build_mod._BUILD_DIRS["ocn"] / "genmake_local").read_text()
        assert "../code_ocn" in text
        assert "../shared_code" in text

    def test_build_cpl_genmake_local_drops_standarddirs(self):
        """The coupler is intentionally serial / minimal-source — its
        genmake_local sets STANDARDDIRS="" so only pkg/atm_ocn_coupler
        and pkg/compon_communic are compiled in."""
        text = (build_mod._BUILD_DIRS["cpl"] / "genmake_local").read_text()
        assert "../code_cpl" in text
        assert "../shared_code" in text
        assert 'STANDARDDIRS=""' in text

    def test_code_atm_packages_conf_has_aim(self):
        text = (build_mod._PKG / "code_atm" / "packages.conf").read_text()
        # The whole point of this experiment vs Held-Suarez.
        assert "aim_v23" in text
        assert "atm_compon_interf" in text

    def test_code_ocn_packages_conf_has_gmredi(self):
        text = (build_mod._PKG / "code_ocn" / "packages.conf").read_text()
        assert "gmredi" in text
        assert "ocn_compon_interf" in text

    def test_code_cpl_packages_conf_has_atm_ocn_coupler(self):
        text = (build_mod._PKG / "code_cpl" / "packages.conf").read_text()
        assert "atm_ocn_coupler" in text
        assert "compon_communic" in text

    def test_size_h_atm_has_5_levels(self):
        text = (build_mod._PKG / "code_atm" / "SIZE.h").read_text()
        # cs32 atmos → 192×32 × 5 sigma levels; we don't try to match
        # the exact whitespace but the literal number is present.
        assert "Nr  =   5" in text

    def test_size_h_ocn_has_15_levels(self):
        text = (build_mod._PKG / "code_ocn" / "SIZE.h").read_text()
        assert "Nr  =  15" in text


# ─── Helper functions ────────────────────────────────────────────────────────

class TestSha1File:
    def test_deterministic(self, tmp_path: Path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello world")
        assert _sha1_file(f) == _sha1_file(f)

    def test_known_value(self, tmp_path: Path):
        # SHA1("hello world") = 2aae6c35c94fcfb415dbe95f408b9ce91ee846ed
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello world")
        assert _sha1_file(f) == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"

    def test_streams_large_files(self, tmp_path: Path):
        # > 1 MB chunk size to exercise the streaming branch
        f = tmp_path / "big.bin"
        f.write_bytes(b"x" * (3 * (1 << 20) + 7))
        h = _sha1_file(f, chunk=1 << 20)
        # cross-check with hashlib directly
        import hashlib
        ref = hashlib.sha1(f.read_bytes()).hexdigest()
        assert h == ref


class TestGitHead:
    def test_returns_unknown_for_non_repo(self, tmp_path: Path):
        assert _git_head(tmp_path) == "unknown"

    def test_returns_hash_for_real_repo(self, tmp_path: Path):
        if not _has_git():
            pytest.skip("git not on PATH")
        subprocess.check_call(["git", "init", "-q"], cwd=tmp_path)
        subprocess.check_call(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "--allow-empty", "-m", "init", "-q"],
            cwd=tmp_path,
        )
        h = _git_head(tmp_path)
        assert len(h) == 40 and all(c in "0123456789abcdef" for c in h)


def _has_git() -> bool:
    from shutil import which
    return which("git") is not None


# ─── stage_inputs against a fake MITgcm tree ─────────────────────────────────

@pytest.fixture
def fake_mitgcm(tmp_path: Path) -> Path:
    """Build a minimal MITgcm-shaped tree with the four sibling input dirs.

    Each sibling gets a couple of plausible ``.bin`` files so the tests
    can verify they're picked up by the right glob.
    """
    root = tmp_path / "MITgcm_fake"
    verif = root / "verification"
    # tools/genmake2 is required by build_component but stage_inputs only
    # checks `verification/` exists.
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "genmake2").write_text("#!/bin/sh\necho fake genmake2\n")
    (root / "tools" / "genmake2").chmod(0o755)

    # Build each sibling dir with synthetic content.
    files_by_dir = {
        "aim.5l_cs/input.thSI":          ["albedo_cs32.bin",
                                          "topo.cpl_FM.bin",
                                          "land_grT_ini.cpl.bin",
                                          "regMask_lat24.bin",
                                          "README"],  # non-.bin to test glob
        "global_ocean.cs32x15/input":     ["bathy_Hmin50.bin",
                                          "lev_T_cs_15k.bin",
                                          "trenberth_taux.bin",
                                          "pickup.0000072000.data",     # non-.bin caught by *
                                          "pickup.0000072000.meta"],
        "tutorial_held_suarez_cs/input": [f"grid_cs32.face00{i}.bin" for i in range(1, 7)] +
                                         ["unrelated.bin"],   # NOT face* → must NOT be picked
        "cpl_aim+ocn/input_cpl":         ["RA.bin", "runOff_cs32_3644.bin",
                                          "data.cpl"],         # non-.bin to test glob
    }
    for sub, fnames in files_by_dir.items():
        d = verif / sub
        d.mkdir(parents=True)
        for fn in fnames:
            (d / fn).write_bytes(fn.encode())  # unique content per file
    return root


class TestStageInputs:
    def test_copies_into_inputs_subdirs(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        # Redirect the package's _INPUTS to a tmp dir so we don't pollute
        # the real package tree.
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        staged = stage_inputs(fake_mitgcm)

        assert (target / "atm").is_dir()
        assert (target / "ocn").is_dir()
        assert (target / "grid").is_dir()
        assert (target / "cpl").is_dir()
        assert "atm" in staged and "ocn" in staged and "grid" in staged and "cpl" in staged

    def test_atm_picks_only_bin_files(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        staged = stage_inputs(fake_mitgcm)
        # README was NOT a .bin, so must not appear.
        assert "README" not in staged["atm"]
        assert "albedo_cs32.bin" in staged["atm"]

    def test_grid_picks_only_face_files(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        staged = stage_inputs(fake_mitgcm)
        # Should be exactly 6 face files; the bare unrelated.bin must NOT match.
        assert sorted(staged["grid"]) == [f"grid_cs32.face00{i}.bin" for i in range(1, 7)]
        assert "unrelated.bin" not in staged["grid"]

    def test_ocn_glob_picks_pickup_pair(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        # The ocean glob is the union of "*.bin" and "pickup.*" so both
        # the binary forcing files AND the pickup pair (which has no
        # extension on the data half) are staged for restart.
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        staged = stage_inputs(fake_mitgcm)
        assert "pickup.0000072000.data" in staged["ocn"]
        assert "pickup.0000072000.meta" in staged["ocn"]

    def test_ocn_glob_excludes_namelists_and_scripts(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        """We regenerate namelists ourselves; the upstream `data`,
        `eedata`, and `prepare_run` files must NOT pollute inputs/ocn/."""
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        # Drop synthetic stand-ins for the upstream namelist clutter
        # into the fake source dir to make sure they get filtered out.
        ocn_src = fake_mitgcm / "verification" / "global_ocean.cs32x15" / "input"
        for noise in ("data", "data.gmredi", "eedata", "prepare_run", "rdwr_grid.m"):
            (ocn_src / noise).write_bytes(b"NOISE")
        staged = stage_inputs(fake_mitgcm)
        for noise in ("data", "data.gmredi", "eedata", "prepare_run", "rdwr_grid.m"):
            assert noise not in staged["ocn"], (
                f"{noise} should not have been staged — fix the ocn glob"
            )

    def test_multi_pattern_union_no_duplicates(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        """If a file matches multiple patterns, it must appear once."""
        # Add a synthetic file that matches both "*.bin" and "pickup.*"
        ocn_src = fake_mitgcm / "verification" / "global_ocean.cs32x15" / "input"
        (ocn_src / "pickup.0000072000.bin").write_bytes(b"x")
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        staged = stage_inputs(fake_mitgcm)
        # No duplicates regardless of how many patterns matched it.
        assert staged["ocn"].count("pickup.0000072000.bin") == 1

    def test_files_are_copies_not_symlinks(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        # Copies, not symlinks → run dirs portable away from MITgcm.
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        stage_inputs(fake_mitgcm)
        for f in (target / "atm").iterdir():
            assert not f.is_symlink(), f"{f} is a symlink — should be a copy"

    def test_idempotent_overwrite(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        # Running twice should not error; second run overwrites.
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        stage_inputs(fake_mitgcm)
        stage_inputs(fake_mitgcm)  # must not raise

    def test_missing_sibling_dir_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        # Remove one required dir; staging must abort cleanly.
        target = tmp_path / "inputs"
        monkeypatch.setattr(build_mod, "_INPUTS", target)
        import shutil
        shutil.rmtree(fake_mitgcm / "verification" / "aim.5l_cs")
        with pytest.raises(SystemExit) as ei:
            stage_inputs(fake_mitgcm)
        assert "input.thSI" in str(ei.value) or "aim.5l_cs" in str(ei.value)

    def test_non_mitgcm_root_is_fatal(self, tmp_path: Path):
        # Pointing at a dir without a verification/ subtree → clean error.
        with pytest.raises(SystemExit) as ei:
            stage_inputs(tmp_path)
        assert "MITgcm" in str(ei.value)


# ─── _SIBLING_DIRS metadata sanity ────────────────────────────────────────────

class TestSiblingDirsConfig:
    def test_all_four_targets_distinct(self):
        targets = [v[1] for v in _SIBLING_DIRS.values()]
        assert len(set(targets)) == len(targets), targets

    def test_targets_are_inputs_subdir_names(self):
        for v in _SIBLING_DIRS.values():
            target = v[1]
            assert target in {"atm", "ocn", "grid", "cpl"}, target

    def test_all_have_descriptions(self):
        for v in _SIBLING_DIRS.values():
            assert isinstance(v[3], str) and len(v[3]) > 5


# ─── write_build_info schema ──────────────────────────────────────────────────

class TestWriteBuildInfo:
    def test_writes_valid_json_with_required_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # Stand up a fake inputs/ dir + dummy executables.
        inputs = tmp_path / "inputs"
        for sub in ("atm", "ocn", "grid", "cpl"):
            (inputs / sub).mkdir(parents=True)
        (inputs / "atm" / "albedo_cs32.bin").write_bytes(b"\x00" * 16)
        (inputs / "ocn" / "bathy.bin").write_bytes(b"\x00" * 32)
        (inputs / "grid" / "grid_cs32.face001.bin").write_bytes(b"\x00" * 64)
        (inputs / "cpl" / "RA.bin").write_bytes(b"\x00" * 8)
        monkeypatch.setattr(build_mod, "_INPUTS", inputs)

        info_path = tmp_path / "build_info.json"
        monkeypatch.setattr(build_mod, "_BUILD_INFO", info_path)

        exes = {}
        for comp in ("atm", "ocn", "cpl"):
            ex = tmp_path / f"build_{comp}" / "mitgcmuv"
            ex.parent.mkdir()
            ex.write_bytes(b"FAKEBIN")
            exes[comp] = ex

        staged = {
            "atm":  ["albedo_cs32.bin"],
            "ocn":  ["bathy.bin"],
            "grid": ["grid_cs32.face001.bin"],
            "cpl":  ["RA.bin"],
        }
        write_build_info(tmp_path, None, exes, staged)

        info = json.loads(info_path.read_text())
        # Top-level keys
        for k in ("built_at", "mitgcm_root", "mitgcm_git_hash",
                  "optfile", "executables", "staged_inputs", "input_provenance"):
            assert k in info, k
        # All three executables recorded with size + path
        assert set(info["executables"]) == {"atm", "ocn", "cpl"}
        for comp, blob in info["executables"].items():
            assert "path" in blob and "size_bytes" in blob
            assert blob["size_bytes"] == len(b"FAKEBIN")
        # Provenance: SHA1 + sample for each input subdir
        for sub in ("atm", "ocn", "grid", "cpl"):
            assert sub in info["input_provenance"]
            assert len(info["input_provenance"][sub]["sample_sha1"]) == 40

    def test_optfile_serialised_correctly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        info_path = tmp_path / "build_info.json"
        monkeypatch.setattr(build_mod, "_BUILD_INFO", info_path)
        monkeypatch.setattr(build_mod, "_INPUTS", tmp_path / "inputs")

        opt = tmp_path / "linux_amd64_gfortran"
        opt.write_text("# fake optfile\n")
        write_build_info(tmp_path, opt, {}, {})

        info = json.loads(info_path.read_text())
        assert info["optfile"] == str(opt.resolve())

    def test_optfile_none_serialises_to_null(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        info_path = tmp_path / "build_info.json"
        monkeypatch.setattr(build_mod, "_BUILD_INFO", info_path)
        monkeypatch.setattr(build_mod, "_INPUTS", tmp_path / "inputs")
        write_build_info(tmp_path, None, {}, {})
        info = json.loads(info_path.read_text())
        assert info["optfile"] is None


# ─── CLI plumbing ────────────────────────────────────────────────────────────

class TestCLI:
    def test_missing_root_with_no_skips_aborts(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        # No --mitgcm-root, no $MITGCM_ROOT, no --skip-* → must abort.
        monkeypatch.delenv("MITGCM_ROOT", raising=False)
        with pytest.raises(SystemExit) as ei:
            main(["--mitgcm-root", "/nonexistent/path"])
        assert "MITgcm" in str(ei.value)

    def test_skip_stage_and_compile_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # With both skips, no build is attempted, no error raised, and
        # no build_info.json gets written (since neither side ran).
        monkeypatch.setattr(build_mod, "_INPUTS", tmp_path / "inputs")
        info_path = tmp_path / "build_info.json"
        monkeypatch.setattr(build_mod, "_BUILD_INFO", info_path)
        monkeypatch.delenv("MITGCM_ROOT", raising=False)
        main(["--skip-stage", "--skip-compile"])
        assert "Build script complete." in capsys.readouterr().out
        assert not info_path.exists(), \
            "build_info.json should not be written when both steps are skipped"


class TestBuildInfoMergesExistingExecutables:
    """Rebuilding a subset of components must not drop the others' entries."""

    def test_skip_compile_after_stage_includes_pre_existing_exes(
        self, monkeypatch: pytest.MonkeyPatch, fake_mitgcm: Path, tmp_path: Path
    ):
        # Stand up "pre-existing" mitgcmuv binaries in build_atm/ and
        # build_ocn/ (but not build_cpl/), redirect _BUILD_DIRS at them,
        # and run stage-only. The resulting build_info.json must list
        # both atm and ocn — even though we did not compile this run.
        fake_builds = {}
        for comp in ("atm", "ocn"):
            d = tmp_path / f"build_{comp}"
            d.mkdir()
            (d / "genmake_local").write_text(f"MODS=../code_{comp}\n")
            ex = d / "mitgcmuv"
            ex.write_bytes(b"PRE-EXISTING")
            fake_builds[comp] = d
        # build_cpl exists with genmake_local but no binary
        d_cpl = tmp_path / "build_cpl"
        d_cpl.mkdir()
        (d_cpl / "genmake_local").write_text("MODS=../code_cpl\n")
        fake_builds["cpl"] = d_cpl

        monkeypatch.setattr(build_mod, "_BUILD_DIRS", fake_builds)
        monkeypatch.setattr(build_mod, "_INPUTS", tmp_path / "inputs")
        info_path = tmp_path / "build_info.json"
        monkeypatch.setattr(build_mod, "_BUILD_INFO", info_path)

        main([
            "--mitgcm-root", str(fake_mitgcm),
            "--skip-compile",
        ])
        info = json.loads(info_path.read_text())
        assert set(info["executables"].keys()) == {"atm", "ocn"}, info["executables"]
