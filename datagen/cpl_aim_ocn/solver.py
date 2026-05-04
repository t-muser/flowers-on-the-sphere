"""End-to-end orchestrator for one coupled AIM+ocean run.

The :func:`run_simulation` entry point follows the same contract as
all other ``datagen`` solvers (``params: dict``, ``out_dir: Path``,
``config: RunConfig | None``, ``**overrides``), but the underlying
machinery is more involved than the dry HS case because:

1. There are **three separate executables** (cpl, ocn, atm) launched
   together via the colon-separated MPMD form of ``mpirun``.
2. The run is **two-phase** — a coupled spin-up followed by a
   diagnostics-on data phase that restarts from the spin-up's pickup
   files. Each phase has its own namelists.
3. The ocean phase 1 starts from a **pre-spun baseline pickup**
   (``pickup.0000072000`` from ``global_ocean.cs32x15``); the
   atmosphere starts cold but with a **per-seed θ perturbation** IC
   that we write to MDS binary up front.
4. The output Zarr is written at the end via :mod:`zarr_writer`,
   which merges the atm and ocn diagnostic streams in native cs32.

All file paths in :class:`RunConfig` default to the package's own
``inputs/``, ``build_*/``, etc. layout produced by
``scripts/build.py``; override only when running against a sibling
checkout or for tests.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from datagen.cpl_aim_ocn.ic import write_atm_theta_ic
from datagen.cpl_aim_ocn.namelist import (
    PhaseTimeConfig,
    SweepParams,
    render_phase_namelists,
)
from datagen.cpl_aim_ocn.staging import (
    COMPONENTS,
    check_run_completed,
    mpmd_command,
    stage_run,
)
from datagen.cpl_aim_ocn.zarr_writer import write_cs32_zarr

logger = logging.getLogger(__name__)

# ─── Package-relative defaults ───────────────────────────────────────────────

_PKG = Path(__file__).resolve().parent
_DEFAULT_INPUTS = _PKG / "inputs"
_DEFAULT_BUILD_DIRS: Mapping[str, Path] = {
    "cpl": _PKG / "build_cpl",
    "ocn": _PKG / "build_ocn",
    "atm": _PKG / "build_atm",
}


# ─── Configuration dataclasses ───────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    """Numerical and infrastructure settings for one ensemble run.

    Physical parameters that vary per ensemble member live in
    :class:`SimulationParams`. Settings here are fixed by the compiled
    binaries and the chosen run length.

    Attributes
    ----------
    delta_t_atm, delta_t_ocn
        Time-step lengths [s] for atm and ocn dycores. Defaults match
        the cs32 verification (450 s atm, 3600 s ocn → 8 atm steps per
        ocean step, integer ratio).
    cpl_atm_send_freq_s
        Coupler exchange period [s]. Must be an integer multiple of
        both ``delta_t_atm`` and ``delta_t_ocn``; defaults to one
        ocean step.
    spinup_days, data_days
        Phase durations [days]. Defaults match the production-sweep
        plan: 30 d spin-up + 365 d data.
    snapshot_interval_days
        Diagnostic-output cadence in the data phase. Default 1 d.
    pickup_suff_baseline_ocn, n_iter0_baseline_ocn
        Tag and starting-iteration for the ocean's parent
        ``global_ocean.cs32x15`` pickup file (staged in inputs/ocn/).
    inputs_root
        Where stage_run looks for binary forcing files. Default:
        ``<package>/inputs``.
    build_dirs
        Mapping ``{"cpl": ..., "ocn": ..., "atm": ...}`` of build dirs
        each containing a ``mitgcmuv`` binary. Default: package-local.
    mpirun
        Launcher binary (``"mpirun"`` or ``"srun"`` etc.).
    timeout_factor
        Wall-clock timeout per phase = simulated-seconds × factor.
        For cs32 the model runs faster than realtime by orders of
        magnitude, so factor=3 is generous.
    """
    # Time-stepping
    delta_t_atm:           float = 450.0
    delta_t_ocn:           float = 3600.0
    cpl_atm_send_freq_s:   float = 3600.0

    # Phase lengths
    spinup_days:           float = 30.0
    data_days:             float = 365.0
    snapshot_interval_days: float = 1.0

    # Ocean baseline pickup (used for phase 1 ocean restart)
    pickup_suff_baseline_ocn: str = "0000072000"
    n_iter0_baseline_ocn:     int = 72000

    # Filesystem
    inputs_root: Path = field(default_factory=lambda: _DEFAULT_INPUTS)
    build_dirs: Mapping[str, Path] = field(
        default_factory=lambda: dict(_DEFAULT_BUILD_DIRS)
    )

    # Launcher
    mpirun: str = "mpirun"
    timeout_factor: float = 3.0

    # ── Derived quantities ───────────────────────────────────────────────────

    @property
    def spinup_seconds(self) -> float:
        return self.spinup_days * 86400.0

    @property
    def data_seconds(self) -> float:
        return self.data_days * 86400.0

    @property
    def snapshot_interval_s(self) -> float:
        return self.snapshot_interval_days * 86400.0

    @property
    def n_atm_steps_spinup(self) -> int:
        return round(self.spinup_seconds / self.delta_t_atm)

    @property
    def n_ocn_steps_spinup(self) -> int:
        return round(self.spinup_seconds / self.delta_t_ocn)

    @property
    def n_atm_steps_data(self) -> int:
        return round(self.data_seconds / self.delta_t_atm)

    @property
    def n_ocn_steps_data(self) -> int:
        return round(self.data_seconds / self.delta_t_ocn)

    def timeout_for_phase_s(self, sim_seconds: float) -> float:
        # Add a 60 s floor so very short test runs still allow startup.
        return max(60.0, sim_seconds * self.timeout_factor)


@dataclass(frozen=True)
class SimulationParams:
    """Physical (per-ensemble-member) parameters."""
    co2_ppm:     float
    solar_scale: float
    gm_kappa:    float
    seed:        int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SimulationParams":
        return cls(
            co2_ppm=float(d["co2_ppm"]),
            solar_scale=float(d["solar_scale"]),
            gm_kappa=float(d["gm_kappa"]),
            seed=int(d["seed"]),
        )

    def as_sweep(self) -> SweepParams:
        return SweepParams(
            co2_ppm=self.co2_ppm, solar_scale=self.solar_scale,
            gm_kappa=self.gm_kappa, seed=self.seed,
        )


# ─── Internal launch helper ──────────────────────────────────────────────────

def _launch_mpmd(
    run_dir: Path, cfg: RunConfig, *, phase_name: str, sim_seconds: float,
) -> None:
    """Run the three binaries via mpirun MPMD; raise on failure.

    Streams combined stdout/stderr to ``run_dir/<phase>_std_outp`` and
    parses each rank's own log for the per-component success marker
    (see :func:`datagen.cpl_aim_ocn.staging.check_run_completed`).
    """
    cmd = mpmd_command(run_dir, build_dirs=cfg.build_dirs, mpirun=cfg.mpirun)
    log_path = run_dir / f"{phase_name}_std_outp"
    timeout = cfg.timeout_for_phase_s(sim_seconds)

    logger.info("Phase %s: launching MPMD (cwd=%s, timeout=%.0f s)",
                phase_name, run_dir, timeout)
    t0 = time.time()
    with open(log_path, "w") as logf:
        try:
            rc = subprocess.run(
                cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Phase {phase_name}: mpirun exceeded {timeout:.0f}s timeout "
                f"(see {log_path})"
            ) from exc
    elapsed = time.time() - t0
    logger.info("Phase %s: mpirun rc=%d in %.1fs", phase_name, rc, elapsed)

    if rc != 0:
        raise RuntimeError(
            f"Phase {phase_name}: mpirun returned {rc} (see {log_path})"
        )

    statuses = check_run_completed(run_dir)
    bad = [c for c, ok in statuses.items() if not ok]
    if bad:
        raise RuntimeError(
            f"Phase {phase_name}: components {bad} did not finish cleanly "
            f"(see rank_<n>/STDOUT.0000)"
        )


# ─── Phase orchestration ─────────────────────────────────────────────────────

_THETA_PERT_FILENAME = "theta_pert.bin"


def _write_atm_ic(run_dir: Path, sim: SimulationParams) -> str:
    """Place the seeded atm θ-perturbation IC at ``rank_2/`` and return
    the relative filename to put into ``hydrogThetaFile``.

    rank_2/ is ensured to exist; the file is written in MDS layout
    (cs32 5×32×192 big-endian float32 + .meta companion).
    """
    rank_atm = run_dir / "rank_2"
    rank_atm.mkdir(parents=True, exist_ok=True)
    write_atm_theta_ic(rank_atm / _THETA_PERT_FILENAME, seed=sim.seed)
    return _THETA_PERT_FILENAME


def _phase_1_namelists(cfg: RunConfig, sim: SimulationParams) -> dict[str, dict[str, str]]:
    """Build the spin-up phase namelist dict.

    No diagnostics; ocean restarts from the baseline pickup; atm
    cold-starts but reads ``hydrogThetaFile`` for the seeded
    perturbation. End-of-phase pickups are written for both components.
    """
    return render_phase_namelists(
        time_cfg=PhaseTimeConfig(
            n_atm_steps=cfg.n_atm_steps_spinup,
            n_ocn_steps=cfg.n_ocn_steps_spinup,
            delta_t_atm=cfg.delta_t_atm,
            delta_t_ocn=cfg.delta_t_ocn,
            snapshot_interval_s=None,
            pickup_suff_ocn=cfg.pickup_suff_baseline_ocn,
            pickup_suff_atm=None,            # cold start
            write_pickup_at_end=True,
            hydrog_theta_file=_THETA_PERT_FILENAME,
            cpl_atm_send_freq_s=cfg.cpl_atm_send_freq_s,
        ),
        sweep=sim.as_sweep(),
    )


def _phase_2_namelists(
    cfg: RunConfig, sim: SimulationParams,
    *, pickup_suff_atm: str, pickup_suff_ocn: str,
) -> dict[str, dict[str, str]]:
    """Build the data-phase namelist dict.

    Restarts from the phase-1 pickups, enables daily diagnostics, does
    not write further pickups (only one per ensemble member, at the
    end of phase 1).
    """
    return render_phase_namelists(
        time_cfg=PhaseTimeConfig(
            n_atm_steps=cfg.n_atm_steps_data,
            n_ocn_steps=cfg.n_ocn_steps_data,
            delta_t_atm=cfg.delta_t_atm,
            delta_t_ocn=cfg.delta_t_ocn,
            snapshot_interval_s=cfg.snapshot_interval_s,
            pickup_suff_ocn=pickup_suff_ocn,
            pickup_suff_atm=pickup_suff_atm,
            write_pickup_at_end=False,
            # Do NOT rewrite the perturbation IC in phase 2 — restart
            # is from the pickup, which already encodes the spun-up
            # state of the perturbed atmosphere.
            hydrog_theta_file=None,
            cpl_atm_send_freq_s=cfg.cpl_atm_send_freq_s,
        ),
        sweep=sim.as_sweep(),
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def run_simulation(
    params: Mapping[str, Any],
    out_dir: Path,
    config: RunConfig | None = None,
    **overrides: Any,
) -> Path:
    """Run one ensemble member end-to-end and write the result Zarr.

    Workflow
    --------
    1. Resolve ``RunConfig`` (apply ``**overrides`` if any).
    2. Write the seeded atm θ-perturbation IC to ``run_dir/rank_2/``.
    3. **Phase 1 (spin-up)** — stage rank_0/1/2 with the no-diag
       namelist set, launch MPMD ``mpirun``, verify clean termination.
       Each component writes a pickup at the final step.
    4. **Phase 2 (data)** — re-stage with the diagnostics-on namelist
       set whose ``pickupSuff`` points at the phase-1 pickups; launch
       again.
    5. Read both rank dirs' MDS output and write a native cs32 Zarr
       at ``out_dir/run.zarr``.

    Parameters
    ----------
    params
        Sweep parameter dict (the ``"params"`` field of a config JSON).
        Must contain ``co2_ppm``, ``solar_scale``, ``gm_kappa``,
        ``seed``. Extra keys (e.g. ``spinup_days``) are ignored at this
        level — those map onto ``RunConfig`` fields and should be
        passed via ``**overrides`` from the CLI driver.
    out_dir
        Output directory. Created if absent. The MITgcm working dir is
        ``out_dir/mitgcm_run/``; the final Zarr is ``out_dir/run.zarr``.
    config
        Numerical / infrastructure config. ``RunConfig()`` if None.
    **overrides
        Per-call ``RunConfig`` field overrides (e.g.
        ``spinup_days=0.5`` for a smoke test).

    Returns
    -------
    Path to the written Zarr store.

    Raises
    ------
    RuntimeError
        If any phase's MPMD launch returns non-zero or fails the
        per-rank success check.
    FileNotFoundError
        If a required binary or input file is missing.
    """
    cfg = config if config is not None else RunConfig()
    if overrides:
        cfg = replace(cfg, **overrides)
    sim = SimulationParams.from_dict(params)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "mitgcm_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight sanity: the three binaries must exist before we waste
    # time staging gigabytes of files.
    for c in COMPONENTS:
        exe = Path(cfg.build_dirs[c]) / "mitgcmuv"
        if not exe.is_file():
            raise FileNotFoundError(
                f"Missing executable: {exe}. Run "
                f"`scripts/build.py --components {c}` first."
            )

    logger.info(
        "Starting cpl_aim+ocn run: CO2=%.0f ppm  solar=%.3f×TSI  "
        "κ_GM=%.0f m²/s  seed=%d",
        sim.co2_ppm, sim.solar_scale, sim.gm_kappa, sim.seed,
    )
    logger.info(
        "  spinup=%.1f d (%d atm × %d ocn steps)  data=%.1f d (%d × %d)",
        cfg.spinup_days, cfg.n_atm_steps_spinup, cfg.n_ocn_steps_spinup,
        cfg.data_days, cfg.n_atm_steps_data, cfg.n_ocn_steps_data,
    )

    # ── Phase 1: spin-up ──
    logger.info("Phase 1: writing atm θ perturbation IC (seed=%d)", sim.seed)
    _write_atm_ic(run_dir, sim)

    logger.info("Phase 1: staging spin-up namelists")
    stage_run(run_dir, namelists=_phase_1_namelists(cfg, sim),
              inputs_root=cfg.inputs_root)

    _launch_mpmd(run_dir, cfg, phase_name="phase1",
                 sim_seconds=cfg.spinup_seconds)

    # ── Phase 2: data collection ──
    # End-of-spin-up iteration counts: atm starts at 0, ocn starts at
    # the baseline pickup iter; both add their phase-1 step counts.
    iter_atm_end = cfg.n_atm_steps_spinup
    iter_ocn_end = cfg.n_iter0_baseline_ocn + cfg.n_ocn_steps_spinup
    pickup_suff_atm = f"{iter_atm_end:010d}"
    pickup_suff_ocn = f"{iter_ocn_end:010d}"
    logger.info("Phase 2: restarting from pickups atm=%s ocn=%s",
                pickup_suff_atm, pickup_suff_ocn)

    stage_run(
        run_dir,
        namelists=_phase_2_namelists(
            cfg, sim,
            pickup_suff_atm=pickup_suff_atm,
            pickup_suff_ocn=pickup_suff_ocn,
        ),
        inputs_root=cfg.inputs_root,
    )

    _launch_mpmd(run_dir, cfg, phase_name="phase2",
                 sim_seconds=cfg.data_seconds)

    # ── Postprocess: read MDS, write Zarr ──
    zarr_path = out_dir / "run.zarr"
    logger.info("Writing native cs32 Zarr → %s", zarr_path)

    attrs = {
        "co2_ppm":            sim.co2_ppm,
        "solar_scale":        sim.solar_scale,
        "solar_const_w_m2":   sim.as_sweep().solar_const_w_m2,
        "gm_kappa":           sim.gm_kappa,
        "seed":               sim.seed,
        "spinup_days":        cfg.spinup_days,
        "data_days":          cfg.data_days,
        "snapshot_interval_days": cfg.snapshot_interval_days,
        "delta_t_atm":        cfg.delta_t_atm,
        "delta_t_ocn":        cfg.delta_t_ocn,
        "params":             json.dumps(dict(params)),
    }
    write_cs32_zarr(run_dir, zarr_path, attrs=attrs)

    logger.info("Run complete: %s", zarr_path)
    return zarr_path
