"""End-to-end smoke test for the coupled AIM+ocean wrapper.

Reproduces a 5-hour ``cpl_aim+ocn`` integration from Python by:

1. Generating namelists for a 5-hour run (40 atm steps × 450 s,
   5 ocn steps × 3600 s, no diagnostics, no pickup) — either from our
   :mod:`datagen.cpl_aim_ocn.namelist` generators (``--source rendered``,
   the default; this is the authoritative regression test) or
   verbatim from a local MITgcm checkout (``--source upstream``;
   useful for bisecting bugs by comparing the two against each other).
2. Calling :func:`datagen.cpl_aim_ocn.staging.stage_run` to materialise
   ``rank_0/``, ``rank_1/``, ``rank_2/`` inside a chosen run directory.
3. Launching the colon-separated MPMD ``mpirun`` command (built by
   :func:`datagen.cpl_aim_ocn.staging.mpmd_command`).
4. Verifying each rank's runtime log ends with the MITgcm success marker.

This is **not a unit test** — it requires:

* a built ``mitgcmuv`` in each of ``build_atm/``, ``build_ocn/``, ``build_cpl/``
  (run ``scripts/build.py`` first),
* a working ``mpirun`` on PATH,
* a populated ``inputs/`` tree (also produced by ``scripts/build.py``).
* For ``--source upstream`` only: an MITgcm checkout.

Usage::

    uv run --project datagen python -m datagen.cpl_aim_ocn.scripts.smoke_test \
        [--source rendered|upstream] \
        [--mitgcm-root /path/to/MITgcm]    # only required for --source upstream
        [--run-dir /tmp/cpl_smoke] \
        [--timeout-s 600]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from datagen.cpl_aim_ocn.namelist import (
    PhaseTimeConfig,
    SweepParams,
    render_phase_namelists,
)
from datagen.cpl_aim_ocn.staging import (
    LAYOUTS,
    check_run_completed,
    mpmd_command,
    stage_run,
)

# ─── Files to copy (verbatim) from each upstream input dir ───────────────────
# ``prepare_run`` is the bash helper script — we don't need it (build.py
# already staged the binary forcing files into our inputs/ tree).

_UPSTREAM_LAYOUTS = {
    # component: (subdir under MITgcm/verification/cpl_aim+ocn/, files to skip)
    "atm": ("input_atm", {"prepare_run"}),
    "ocn": ("input_ocn", {"prepare_run"}),
    "cpl": ("input_cpl",
            # input_cpl ships with two .bin files (RA.bin and runOff_*.bin)
            # — we already symlinked those via stage_run; here we want
            # only the namelist `data.cpl`.
            {"RA.bin", "runOff_cs32_3644.bin"}),
}


def _read_upstream_namelists(verif_dir: Path) -> dict[str, dict[str, str]]:
    """Slurp every namelist file from the three upstream input dirs.

    Returns ``{component: {filename: file_content}}``.
    """
    out: dict[str, dict[str, str]] = {}
    for comp, (subdir, skip) in _UPSTREAM_LAYOUTS.items():
        src = verif_dir / subdir
        if not src.is_dir():
            sys.exit(f"Upstream input dir missing: {src}")
        files: dict[str, str] = {}
        for f in sorted(src.iterdir()):
            if not f.is_file() or f.name in skip:
                continue
            files[f.name] = f.read_text()
        if not files:
            sys.exit(f"No namelist files found in {src}")
        out[comp] = files
    return out


def _render_smoke_namelists() -> dict[str, dict[str, str]]:
    """Build a 5-hour smoke namelist set from our generators.

    Matches the upstream verification configuration:
      * 40 atm steps × 450 s = 18 000 s
      *  5 ocn steps × 3600 s = 18 000 s
      * no diagnostics (we only check the run terminates cleanly)
      * cold start (no pickup, no theta perturbation)

    Sweep parameters use representative mid-grid values (348 ppm CO2,
    1× solar, 1000 m²/s κ, seed=0) — the run is too short for any of
    these to materially affect the success criterion, but they exercise
    the substitution paths.
    """
    cfg = PhaseTimeConfig(
        n_atm_steps=40,
        n_ocn_steps=5,
        snapshot_interval_s=None,
        write_pickup_at_end=False,
    )
    sweep = SweepParams(
        co2_ppm=348.0, solar_scale=1.00, gm_kappa=1000.0, seed=0,
    )
    return render_phase_namelists(time_cfg=cfg, sweep=sweep)


# ─── Driver ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", choices=("rendered", "upstream"), default="rendered",
        help="Namelist source: 'rendered' (default) uses our "
             "render_phase_namelists(); 'upstream' reads the verbatim "
             "files from <MITGCM_ROOT>/verification/cpl_aim+ocn/input_*/.",
    )
    ap.add_argument(
        "--mitgcm-root", type=Path,
        default=Path(os.environ.get("MITGCM_ROOT", "")),
        help="MITgcm checkout root — required only for --source upstream "
             "(also looks at $MITGCM_ROOT).",
    )
    ap.add_argument(
        "--run-dir", type=Path, default=Path("/tmp/cpl_aim_ocn_smoke"),
        help="Run directory (created/wiped). Default: /tmp/cpl_aim_ocn_smoke",
    )
    ap.add_argument(
        "--timeout-s", type=int, default=600,
        help="Hard timeout for the mpirun (default: 10 min — verification "
             "normally finishes in <1 min).",
    )
    ap.add_argument(
        "--keep", action="store_true",
        help="Preserve any pre-existing contents of --run-dir instead of wiping.",
    )
    ap.add_argument(
        "--mpirun", default="mpirun",
        help="MPI launcher to use (default: mpirun).",
    )
    args = ap.parse_args(argv)

    # ── Wipe / prepare run dir ──
    run_dir: Path = args.run_dir
    if run_dir.exists() and not args.keep:
        print(f"Wiping {run_dir}", flush=True)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Build namelist set ──
    if args.source == "upstream":
        if not args.mitgcm_root or not args.mitgcm_root.is_dir():
            sys.exit("--source upstream requires --mitgcm-root or $MITGCM_ROOT.")
        verif = args.mitgcm_root / "verification" / "cpl_aim+ocn"
        if not verif.is_dir():
            sys.exit(f"Not a valid MITgcm checkout: {verif} missing.")
        print(f"Source: upstream verbatim from {verif}/input_*", flush=True)
        namelists = _read_upstream_namelists(verif)
    else:
        print("Source: rendered via datagen.cpl_aim_ocn.namelist", flush=True)
        namelists = _render_smoke_namelists()

    for comp, files in namelists.items():
        print(f"  {comp}: {sorted(files)}", flush=True)

    # ── Stage rank dirs ──
    print(f"\nStaging {run_dir} …", flush=True)
    rank_dirs = stage_run(run_dir, namelists=namelists)
    for comp, p in rank_dirs.items():
        n = sum(1 for _ in p.iterdir())
        print(f"  {comp:3s} → {p.name}/ ({n} entries)", flush=True)

    # ── Launch ──
    cmd = mpmd_command(run_dir, mpirun=args.mpirun)
    print(f"\n$ (cd {run_dir} && {' '.join(cmd)})", flush=True)
    log_file = run_dir / "std_outp"
    with open(log_file, "w") as logf:
        try:
            rc = subprocess.run(
                cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT,
                timeout=args.timeout_s,
            ).returncode
        except subprocess.TimeoutExpired:
            print(f"\nFAIL: mpirun exceeded {args.timeout_s} s", flush=True)
            return 2

    print(f"  mpirun exit code: {rc}", flush=True)

    # ── Verify each rank finished cleanly ──
    # The atm/ocn dycore prints "ended Normally" in STDOUT.0000; the
    # coupler binary uses a bare Fortran STOP and writes no marker, so
    # for cpl `check_run_completed` only verifies its log file (which
    # is `Coupler.0000.clog`) is non-empty. mpirun rc=0 + atm/ocn clean
    # is the transitive guarantee that the coupler reached MPI_Finalize.
    print("\nChecking per-rank runtime logs:", flush=True)
    statuses = check_run_completed(run_dir)
    all_ok = True
    for layout in LAYOUTS:
        ok = statuses[layout.component]
        criterion = (
            f"{layout.log_filename!r} contains {layout.log_marker!r}"
            if layout.log_marker
            else f"{layout.log_filename!r} non-empty"
        )
        flag = "OK" if ok else "FAIL"
        print(f"  {flag:4s} {layout.component:3s} ({criterion})", flush=True)
        if not ok:
            all_ok = False
            log = run_dir / layout.rank_dirname / layout.log_filename
            if log.is_file():
                print("    --- last 20 lines ---", flush=True)
                for line in log.read_text(errors="replace").splitlines()[-20:]:
                    print(f"      {line}", flush=True)
            else:
                print(f"    (no log file at {log})", flush=True)

    if rc != 0 or not all_ok:
        print("\nSMOKE TEST FAILED", flush=True)
        return 1
    print(f"\nSMOKE TEST PASSED  (run preserved at {run_dir})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
