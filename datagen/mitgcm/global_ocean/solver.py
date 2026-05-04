"""MITgcm global ocean tutorial runner.

This module stages and runs the MITgcm
``verification/tutorial_global_oce_latlon`` experiment from the vendored
MITgcm tree, then converts selected MDS output fields into the repository's
standard ``(time, field, lat, lon)`` Zarr layout.

The physical setup follows the MITgcm tutorial:

* 4 degree spherical-polar grid, 90 longitude by 40 latitude cells.
* Latitudinal extent 80S to 80N.
* 15 vertical z-levels with layer thicknesses from 50 m to 690 m.
* Realistic bathymetry, Levitus hydrography/restoring, Trenberth wind stress,
  NCEP heat and freshwater fluxes, and GM/Redi mixing.

Only runtime details that matter for dataset production are patched in Python
(``nTimeSteps``, output cadence, checkpoint cadence, timestep values, and a
small GM/Redi sweep hook). The tutorial namelists and binary inputs remain the
source of truth.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from datagen.mitgcm.mds import mds_dtype, parse_mds_meta
from datagen.resample import write_latlon_zarr

logger = logging.getLogger(__name__)

_PKG = Path(__file__).resolve().parent
_REPO_ROOT = _PKG.parents[2]
_TUTORIAL = _REPO_ROOT / "mitgcm" / "verification" / "tutorial_global_oce_latlon"
_TUTORIAL_INPUT = _TUTORIAL / "input"
_DEFAULT_EXECUTABLE = _PKG / "build" / "mitgcmuv"

GLOBAL_OCEAN_DEL_R: tuple[float, ...] = (
    50.0, 70.0, 100.0, 140.0, 190.0,
    240.0, 290.0, 340.0, 390.0, 440.0,
    490.0, 540.0, 590.0, 640.0, 690.0,
)


@dataclass(frozen=True)
class GlobalOceanRunConfig:
    """Numerical and infrastructure settings for the global ocean run."""

    # Tutorial compile-time grid. These must match code/SIZE.h.
    Nlon: int = 90
    Nlat: int = 40
    Nr: int = 15

    # Production default follows the tutorial recommendation: 100 model years
    # on the 360-day MITgcm calendar.
    n_timesteps: int = 36000
    delta_t_mom: float = 1800.0
    delta_t_tracer: float = 86400.0
    delta_t_clock: float = 86400.0
    delta_t_freesurf: float = 86400.0

    # Monthly output keeps 100-year runs manageable while retaining seasonal-scale
    # variability.
    snapshot_interval_days: float = 30.0
    monitor_freq_s: float = 1.0
    write_pickup: bool = True

    # Dataset extraction. The tutorial diagnostics example writes theta/salt
    # at level 1 and velocity at level 2, so mirror that convention.
    tracer_level: int = 1
    velocity_level: int = 2

    # Physical sweep hooks. Defaults are the tutorial values.
    gm_background_k: float = 1.0e3
    visc_ah: float = 5.0e5
    diff_kr: float = 3.0e-5
    tau_theta_relax_days: float = 60.0
    tau_salt_relax_days: float = 180.0

    # Infrastructure.
    executable: Path = field(default_factory=lambda: _DEFAULT_EXECUTABLE)
    input_dir: Path = field(default_factory=lambda: _TUTORIAL_INPUT)
    mpirun_cmd: tuple[str, ...] = ("mpirun", "-n", "1")
    timeout_s: float = 3600.0
    run_name: str = "global_ocean_latlon"

    @property
    def snapshot_interval_s(self) -> float:
        return self.snapshot_interval_days * 86400.0

    @property
    def run_seconds(self) -> float:
        return float(self.n_timesteps) * self.delta_t_clock

    @property
    def checkpoint_freq_s(self) -> float:
        return self.run_seconds if self.write_pickup else 0.0

    @property
    def tau_theta_relax_s(self) -> float:
        return self.tau_theta_relax_days * 86400.0

    @property
    def tau_salt_relax_s(self) -> float:
        return self.tau_salt_relax_days * 86400.0


def global_ocean_lat_grid(Nlat: int = 40, *, yg_origin: float = -80.0) -> np.ndarray:
    """Return MITgcm cell-center latitudes in radians."""
    dy = 160.0 / Nlat
    return np.deg2rad(yg_origin + (np.arange(Nlat) + 0.5) * dy)


def global_ocean_lon_grid(Nlon: int = 90, *, xg_origin: float = 0.0) -> np.ndarray:
    """Return MITgcm cell-center longitudes in radians."""
    dx = 360.0 / Nlon
    return np.deg2rad(xg_origin + (np.arange(Nlon) + 0.5) * dx)


def _fmt(value: bool | int | float | str) -> str:
    """Render a Python value as a Fortran namelist scalar."""
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.8g}"
        if "." not in text and "e" not in text.lower():
            text += "."
        return text
    if "'" in value:
        raise ValueError(f"Cannot embed apostrophe in namelist string: {value!r}")
    return f"'{value}'"


_BLOCK_OPEN_RE = re.compile(r"^\s*&([A-Za-z][A-Za-z0-9_]*)\s*$")
_BLOCK_CLOSE_RE = re.compile(r"^\s*[&/]\s*$")


def _is_comment(line: str) -> bool:
    stripped = line.lstrip()
    return not stripped or stripped.startswith(("#", "!"))


def _set_namelist_value(
        text: str,
        block: str,
        key: str,
        value: bool | int | float | str,
) -> str:
    """Set the last active occurrence of ``key`` inside ``&block``."""
    lines = text.splitlines(keepends=True)
    starts = [
        i for i, line in enumerate(lines)
        if (m := _BLOCK_OPEN_RE.match(line)) and m.group(1).upper() == block.upper()
    ]
    if not starts:
        raise ValueError(f"Namelist block &{block} not found")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines))
                if _BLOCK_CLOSE_RE.match(lines[i])), None)
    if end is None:
        raise ValueError(f"Namelist block &{block} is not closed")

    key_re = re.compile(rf"^(\s*){re.escape(key)}\s*=\s*(.*?)(,?\s*)$",
                        re.IGNORECASE)
    last_match: int | None = None
    for i in range(start + 1, end):
        if _is_comment(lines[i]):
            continue
        if key_re.match(lines[i]):
            last_match = i

    formatted = _fmt(value)
    if last_match is not None:
        m = key_re.match(lines[last_match])
        assert m is not None
        indent, _old_value, trailer = m.groups()
        comma = "," if "," in trailer else ""
        lines[last_match] = f"{indent}{key}= {formatted}{comma}\n"
    else:
        indent = " "
        for i in range(start + 1, end):
            if not _is_comment(lines[i]) and "=" in lines[i]:
                indent = re.match(r"^(\s*)", lines[i]).group(1) or " "
                break
        lines.insert(end, f"{indent}{key}= {formatted},\n")
    return "".join(lines)


def _read_input_template(input_dir: Path, name: str) -> str:
    path = Path(input_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing global-ocean input template: {path}")
    return path.read_text()


def render_data(cfg: GlobalOceanRunConfig) -> str:
    """Render the tutorial ``data`` namelist with runtime overrides."""
    text = _read_input_template(cfg.input_dir, "data")
    replacements: Mapping[tuple[str, str], bool | int | float | str] = {
        ("PARM01", "viscAh"): cfg.visc_ah,
        ("PARM01", "diffKrT"): cfg.diff_kr,
        ("PARM01", "diffKrS"): cfg.diff_kr,
        ("PARM03", "nTimeSteps"): cfg.n_timesteps,
        ("PARM03", "deltaTmom"): cfg.delta_t_mom,
        ("PARM03", "deltaTtracer"): cfg.delta_t_tracer,
        ("PARM03", "deltaTClock"): cfg.delta_t_clock,
        ("PARM03", "deltaTfreesurf"): cfg.delta_t_freesurf,
        ("PARM03", "pChkptFreq"): cfg.checkpoint_freq_s,
        ("PARM03", "dumpFreq"): cfg.snapshot_interval_s,
        ("PARM03", "monitorFreq"): cfg.monitor_freq_s,
        ("PARM03", "tauThetaClimRelax"): cfg.tau_theta_relax_s,
        ("PARM03", "tauSaltClimRelax"): cfg.tau_salt_relax_s,
        ("PARM05", "the_run_name"): cfg.run_name,
    }
    for (block, key), value in replacements.items():
        text = _set_namelist_value(text, block, key, value)
    return text


def render_data_gmredi(cfg: GlobalOceanRunConfig) -> str:
    """Render ``data.gmredi`` with the configurable background diffusivity."""
    text = _read_input_template(cfg.input_dir, "data.gmredi")
    return _set_namelist_value(
        text, "GM_PARM01", "GM_background_K", cfg.gm_background_k
    )


def render_static_namelist(cfg: GlobalOceanRunConfig, name: str) -> str:
    """Return an unchanged tutorial namelist from the configured input dir."""
    return _read_input_template(cfg.input_dir, name)


def _symlink_inputs(run_dir: Path, input_dir: Path) -> None:
    """Symlink tutorial binary inputs into ``run_dir``."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Global-ocean input directory not found: {input_dir}"
        )
    generated = {"data", "data.gmredi", "data.pkg", "data.ptracers", "eedata"}
    for src in sorted(input_dir.iterdir()):
        if not src.is_file() or src.name in generated:
            continue
        dst = run_dir / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def _write_text(path: Path, content: str) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    path.write_text(content)


def stage_global_ocean_run(
        run_dir: Path,
        cfg: GlobalOceanRunConfig,
) -> Path:
    """Populate a run directory with executable, inputs, and namelists."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    executable = Path(cfg.executable)
    if not executable.exists():
        raise FileNotFoundError(
            f"MITgcm global-ocean executable not found: {executable}. "
            "Build it with `python -m datagen.mitgcm.global_ocean.scripts.build`."
        )

    exe_link = run_dir / "mitgcmuv"
    if exe_link.is_symlink() or exe_link.exists():
        exe_link.unlink()
    exe_link.symlink_to(executable.resolve())

    _symlink_inputs(run_dir, Path(cfg.input_dir))

    _write_text(run_dir / "data", render_data(cfg))
    _write_text(run_dir / "data.gmredi", render_data_gmredi(cfg))
    for name in ("data.pkg", "data.ptracers", "eedata"):
        _write_text(run_dir / name, render_static_namelist(cfg, name))

    return run_dir


def _launch_mitgcm(run_dir: Path, cfg: GlobalOceanRunConfig) -> None:
    cmd = [*cfg.mpirun_cmd, "./mitgcmuv"]
    log_path = run_dir / "STDOUT_global_ocean.log"
    for pattern in ("STDOUT.????", "STDERR.????"):
        for old_log in run_dir.glob(pattern):
            old_log.unlink()

    logger.info("Launching global ocean MITgcm: %s (cwd=%s)", cmd, run_dir)
    t0 = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            cwd=run_dir,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            timeout=cfg.timeout_s,
        )
    logger.info(
        "Global ocean MITgcm finished in %.1f s (exit=%d)",
        time.time() - t0,
        result.returncode,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MITgcm global ocean exited with code {result.returncode}. "
            f"See {log_path}"
        )

    fatal_lines: list[str] = []
    for rank_log in sorted(run_dir.glob("STDERR.????")):
        text = rank_log.read_text(errors="replace")
        for line in text.splitlines():
            if "*** ERROR ***" in line or "ABNORMAL END" in line:
                fatal_lines.append(f"{rank_log.name}: {line}")
                break
    for rank_log in sorted(run_dir.glob("STDOUT.????")):
        text = rank_log.read_text(errors="replace")
        for line in text.splitlines():
            if "PROGRAM MAIN: ends with fatal Error" in line:
                fatal_lines.append(f"{rank_log.name}: {line}")
                break
    if fatal_lines:
        raise RuntimeError(
            "MITgcm global ocean reported fatal errors despite exit code 0.\n"
            + "\n".join(fatal_lines[:8])
        )


def _find_mds_iters(run_dir: Path, prefix: str) -> list[int]:
    iters: list[int] = []
    for p in sorted(run_dir.glob(f"{prefix}.*.meta")):
        parts = p.stem.split(".")
        if len(parts) >= 2:
            try:
                iters.append(int(parts[1]))
            except ValueError:
                pass
    return sorted(set(iters))


def _read_mds_single_field(
        run_dir: Path,
        prefix: str,
        iters: list[int],
) -> np.ndarray:
    """Read a standard MITgcm MDS prefix containing one model field.

    Supports both global files (``T.0000000020.meta``) and tiled files
    (``T.0000000020.001.001.meta``).
    """
    if not iters:
        raise FileNotFoundError(f"No {prefix}.*.meta files found in {run_dir}")

    out: np.ndarray | None = None
    for t_idx, iter_num in enumerate(iters):
        tile_metas = sorted(run_dir.glob(f"{prefix}.{iter_num:010d}.*.*.meta"))
        global_meta = run_dir / f"{prefix}.{iter_num:010d}.meta"
        if global_meta.exists():
            tile_metas = [global_meta]
        if not tile_metas:
            raise FileNotFoundError(
                f"No metadata files found for {prefix}.{iter_num:010d}"
            )

        for meta_path in tile_metas:
            meta = parse_mds_meta(meta_path)
            if meta["nrecords"] != 1:
                raise ValueError(
                    f"{meta_path}: expected one record, got {meta['nrecords']}"
                )

            dim_list = meta["dim_list"]
            if len(dim_list) == 3:
                nx_g, x_start, x_end = dim_list[0]
                ny_g, y_start, y_end = dim_list[1]
                nz_g, z_start, z_end = dim_list[2]
                shape_g = (nz_g, ny_g, nx_g)
                shape_tile = (
                    z_end - z_start + 1,
                    y_end - y_start + 1,
                    x_end - x_start + 1,
                )
                dst = (
                    slice(z_start - 1, z_end),
                    slice(y_start - 1, y_end),
                    slice(x_start - 1, x_end),
                )
            elif len(dim_list) == 2:
                nx_g, x_start, x_end = dim_list[0]
                ny_g, y_start, y_end = dim_list[1]
                shape_g = (ny_g, nx_g)
                shape_tile = (
                    y_end - y_start + 1,
                    x_end - x_start + 1,
                )
                dst = (
                    slice(y_start - 1, y_end),
                    slice(x_start - 1, x_end),
                )
            else:
                raise ValueError(f"{meta_path}: unsupported nDims={len(dim_list)}")

            expected = int(np.prod(shape_tile))
            raw = np.fromfile(meta_path.with_suffix(".data"),
                              dtype=mds_dtype(meta["dataprec"]))
            if raw.size != expected:
                raise ValueError(
                    f"{meta_path.with_suffix('.data')}: expected {expected} values, "
                    f"found {raw.size}"
                )
            field = raw.astype(np.float32, copy=False).reshape(shape_tile)

            if out is None:
                out = np.empty((len(iters), *shape_g), dtype=np.float32)
            out[t_idx][dst] = field

    assert out is not None
    return out


def read_global_ocean_output(
        run_dir: Path,
        cfg: GlobalOceanRunConfig,
) -> dict[str, np.ndarray]:
    """Read tutorial state-dump prefixes into memory."""
    run_dir = Path(run_dir)
    iters = _find_mds_iters(run_dir, "T")
    if not iters:
        raise FileNotFoundError(
            f"No T.*.meta files found in {run_dir}. "
            "Did MITgcm produce state dumps? Check dumpFreq in the data namelist."
        )
    return {
        "time": np.asarray(iters, dtype=np.float64) * cfg.delta_t_clock,
        "THETA": _read_mds_single_field(run_dir, "T", iters),
        "SALT": _read_mds_single_field(run_dir, "S", iters),
        "UVEL": _read_mds_single_field(run_dir, "U", iters),
        "VVEL": _read_mds_single_field(run_dir, "V", iters),
        "ETAN": _read_mds_single_field(run_dir, "Eta", iters),
    }


def _level_index(level: int, Nr: int, name: str) -> int:
    if not 1 <= level <= Nr:
        raise ValueError(f"{name} must be in 1..{Nr}, got {level}")
    return level - 1


def extract_global_ocean_fields(
        data: Mapping[str, np.ndarray],
        cfg: GlobalOceanRunConfig,
) -> tuple[list[np.ndarray], list[str], np.ndarray]:
    """Extract the 2-D fields written to the benchmark Zarr store."""
    tracer_k = _level_index(cfg.tracer_level, cfg.Nr, "tracer_level")
    vel_k = _level_index(cfg.velocity_level, cfg.Nr, "velocity_level")

    theta = data["THETA"][:, tracer_k].astype(np.float32)
    salt = data["SALT"][:, tracer_k].astype(np.float32)
    u = data["UVEL"][:, vel_k].astype(np.float32)
    v = data["VVEL"][:, vel_k].astype(np.float32)
    eta = data["ETAN"].astype(np.float32)

    time_arr = np.asarray(data["time"], dtype=np.float64)
    time_arr = time_arr - time_arr[0]

    return (
        [theta, salt, u, v, eta],
        [
            f"theta_k{cfg.tracer_level}",
            f"salt_k{cfg.tracer_level}",
            f"u_k{cfg.velocity_level}",
            f"v_k{cfg.velocity_level}",
            "eta",
        ],
        time_arr,
    )


def write_global_ocean_zarr(
        data: Mapping[str, np.ndarray],
        out_path: Path,
        cfg: GlobalOceanRunConfig,
        params: Mapping[str, Any] | None = None,
) -> None:
    """Write global ocean output to a Zarr store."""
    field_arrays, field_names, time_arr = extract_global_ocean_fields(data, cfg)
    write_latlon_zarr(
        out_path,
        time_arr=time_arr,
        field_arrays=field_arrays,
        field_names=field_names,
        lat_target=global_ocean_lat_grid(cfg.Nlat),
        lon_target=global_ocean_lon_grid(cfg.Nlon),
        description=(
            "MITgcm global ocean tutorial on a 4 degree spherical-polar grid; "
            "surface temperature/salinity, level-2 velocity, and sea surface height."
        ),
        run_id=None,
        params=dict(params or {}),
    )


def run_simulation(
        params: Mapping[str, Any] | None,
        out_dir: Path,
        config: GlobalOceanRunConfig | None = None,
        **overrides: Any,
) -> None:
    """Run one global ocean simulation and write ``run.zarr``.

    ``params`` is optional and currently supports the same keys as
    ``GlobalOceanRunConfig`` overrides for runtime/sweep values. Explicit
    ``overrides`` take precedence.
    """
    base = config if config is not None else GlobalOceanRunConfig()
    param_overrides = {
        key: value
        for key, value in dict(params or {}).items()
        if key in GlobalOceanRunConfig.__dataclass_fields__
    }
    cfg = replace(base, **param_overrides, **overrides)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "global_ocean_run"

    logger.info(
        "Starting MITgcm global ocean run: n_timesteps=%d, dump every %.3g days",
        cfg.n_timesteps,
        cfg.snapshot_interval_days,
    )
    stage_global_ocean_run(run_dir, cfg)
    _launch_mitgcm(run_dir, cfg)

    logger.info("Reading MITgcm MDS output and writing Zarr")
    data = read_global_ocean_output(run_dir, cfg)
    write_global_ocean_zarr(data, out_dir / "run.zarr", cfg, params=params)
