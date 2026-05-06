"""MITgcm global-ocean cubed-sphere runner.

This module stages and runs the MITgcm
``verification/global_ocean.cs32x15`` experiment from the vendored MITgcm
tree, then converts MDS state dumps into a native cubed-sphere Zarr layout
``(time, field, face, y, x)`` with ``face=6, y=x=32``.

The physical setup follows the MITgcm tutorial:

* Cubed-sphere grid, six 32 by 32 faces, 15 vertical z-levels with layer
  thicknesses from 50 m to 690 m.
* Realistic bathymetry (``bathy_Hmin50.bin``), Levitus hydrography (``lev_*``)
  for both initial state and surface restoring, Trenberth wind stress, and
  shi*/ncep heat and freshwater fluxes.
* Warm-start from ``pickup.0000072000`` shipped with the tutorial.
* GM/Redi mixing with a configurable background diffusivity.

Only runtime details that matter for dataset production are patched in Python
(``nIter0``, ``nTimeSteps``, output cadence, checkpoint cadence, timestep
values, restoring timescales, viscosities, GM/Redi knob). The tutorial
namelists and binary inputs remain the source of truth.
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
import xarray as xr

from datagen.mitgcm.mds import mds_dtype, parse_mds_meta

logger = logging.getLogger(__name__)

_PKG = Path(__file__).resolve().parent
_REPO_ROOT = _PKG.parents[2]
_TUTORIAL = _REPO_ROOT / "mitgcm" / "verification" / "global_ocean.cs32x15"
_TUTORIAL_INPUT = _TUTORIAL / "input"
# grid_cs32.face00?.bin lives in the held-suarez tutorial; cs32x15's
# prepare_run script symlinks them. We replicate that at stage time.
_CS_GRID_DIR = _REPO_ROOT / "mitgcm" / "verification" / "tutorial_held_suarez_cs" / "input"
_DEFAULT_EXECUTABLE = _PKG / "build" / "mitgcmuv"

GLOBAL_OCEAN_DEL_R: tuple[float, ...] = (
    50.0, 70.0, 100.0, 140.0, 190.0,
    240.0, 290.0, 340.0, 390.0, 440.0,
    490.0, 540.0, 590.0, 640.0, 690.0,
)

# Number of grid_cs32 face files to symlink into the run directory.
_N_FACES = 6
# Tile dimensions baked into datagen/mitgcm/global_ocean/code/SIZE.h:
# six 32x32 tiles laid out in X with single-rank execution.
_FACE_SIZE = 32


@dataclass(frozen=True)
class GlobalOceanRunConfig:
    """Numerical and infrastructure settings for the global ocean run."""

    # Tutorial compile-time grid. These must match code/SIZE.h.
    n_face: int = _N_FACES
    face_size: int = _FACE_SIZE
    Nr: int = 15

    # Production default follows the tutorial recommendation: 100 model years
    # on the 360-day MITgcm calendar, restarted from the shipped pickup.
    n_iter0: int = 72000
    n_timesteps: int = 36000
    delta_t_mom: float = 1800.0
    delta_t_tracer: float = 86400.0
    delta_t_clock: float = 86400.0
    delta_t_freesurf: float = 86400.0

    # Monthly output keeps 100-year runs manageable while retaining seasonal
    # variability.
    snapshot_interval_days: float = 30.0
    monitor_freq_s: float = 1.0
    write_pickup: bool = True

    # Dataset extraction. The tutorial diagnostics example writes theta/salt
    # at level 1 and velocity at level 2, so mirror that convention.
    tracer_level: int = 1
    velocity_level: int = 2

    # Optional 3-D output mode. When ``levels`` is a tuple of 1-indexed depth
    # levels, the writer emits per-variable arrays with a ``level`` axis
    # (theta/salt/u/v at each level + 2-D eta) instead of the legacy
    # single-level stack. ``tracer_level`` / ``velocity_level`` are ignored.
    levels: tuple[int, ...] | None = None

    # Physical sweep hooks. Defaults are the tutorial values.
    gm_background_k: float = 1.0e3
    visc_ah: float = 5.0e5
    diff_kr: float = 3.0e-5
    tau_theta_relax_days: float = 60.0
    tau_salt_relax_days: float = 180.0

    # Infrastructure.
    executable: Path = field(default_factory=lambda: _DEFAULT_EXECUTABLE)
    input_dir: Path = field(default_factory=lambda: _TUTORIAL_INPUT)
    grid_dir: Path = field(default_factory=lambda: _CS_GRID_DIR)
    mpirun_cmd: tuple[str, ...] = ("mpirun", "-n", "1")
    timeout_s: float = 3600.0
    run_name: str = "global_ocean_cs32x15"

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


def load_grid_cs32(grid_dir: Path) -> dict[str, np.ndarray]:
    """Load cell-center longitudes and latitudes from grid_cs32.face00?.bin.

    Each face file contains 18 records of (face_size+1)x(face_size+1) double-
    precision big-endian values; record 1 is xC (cell-center longitude in
    degrees) and record 2 is yC (cell-center latitude in degrees). We return
    only the interior face_size x face_size cells.
    """
    rec_n = _FACE_SIZE + 1
    rec_bytes = rec_n * rec_n * 8
    xc = np.empty((_N_FACES, _FACE_SIZE, _FACE_SIZE), dtype=np.float64)
    yc = np.empty_like(xc)
    for i in range(_N_FACES):
        path = Path(grid_dir) / f"grid_cs32.face{i + 1:03d}.bin"
        with open(path, "rb") as f:
            xc_raw = np.frombuffer(f.read(rec_bytes), dtype=">f8").reshape(rec_n, rec_n)
            yc_raw = np.frombuffer(f.read(rec_bytes), dtype=">f8").reshape(rec_n, rec_n)
        xc[i] = xc_raw[:_FACE_SIZE, :_FACE_SIZE]
        yc[i] = yc_raw[:_FACE_SIZE, :_FACE_SIZE]
    return {"xc": xc, "yc": yc}


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
        # useSingleCpuIO emits one global MDS file per field per iteration,
        # which our reader concatenates trivially.
        ("PARM01", "useSingleCpuIO"): True,
        ("PARM03", "nIter0"): cfg.n_iter0,
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


def render_data_pkg(cfg: GlobalOceanRunConfig) -> str:
    """Render ``data.pkg`` with diagnostics and MNC turned off.

    The cs32x15 tutorial enables ``useDiagnostics`` and ``useMNC``; we don't
    ship those packages, and reading raw MDS state dumps gives us everything
    we need.
    """
    text = _read_input_template(cfg.input_dir, "data.pkg")
    text = _set_namelist_value(text, "PACKAGES", "useDiagnostics", False)
    text = _set_namelist_value(text, "PACKAGES", "useMNC", False)
    return text


def render_static_namelist(cfg: GlobalOceanRunConfig, name: str) -> str:
    """Return an unchanged tutorial namelist from the configured input dir."""
    return _read_input_template(cfg.input_dir, name)


# Files we materialize ourselves in the run directory; these must not be
# overwritten by the symlink pass. Excluded files in the tutorial input/ tree
# (data.diagnostics, data.mnc, prepare_run, MATLAB helpers) are also skipped
# because they are useless or actively misleading once we patch out the pkgs.
_GENERATED_NAMES = frozenset({"data", "data.gmredi", "data.pkg"})
_SKIP_INPUT_NAMES = frozenset({
    "data.diagnostics",
    "data.mnc",
    "prepare_run",
    "mk_bathy4gcm.m",
    "rdwr_grid.m",
})


def _symlink_inputs(run_dir: Path, input_dir: Path, grid_dir: Path) -> None:
    """Symlink tutorial binary inputs and grid files into ``run_dir``."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Global-ocean input directory not found: {input_dir}"
        )
    for src in sorted(input_dir.iterdir()):
        if not src.is_file():
            continue
        if src.name in _GENERATED_NAMES or src.name in _SKIP_INPUT_NAMES:
            continue
        dst = run_dir / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    if not grid_dir.is_dir():
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")
    for i in range(_N_FACES):
        name = f"grid_cs32.face{i + 1:03d}.bin"
        src = grid_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing cs32 grid file: {src}")
        dst = run_dir / name
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

    _symlink_inputs(run_dir, Path(cfg.input_dir), Path(cfg.grid_dir))

    _write_text(run_dir / "data", render_data(cfg))
    _write_text(run_dir / "data.gmredi", render_data_gmredi(cfg))
    _write_text(run_dir / "data.pkg", render_data_pkg(cfg))
    for name in ("eedata",):
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


def _to_face_layout(
        arr: np.ndarray,
        *,
        n_face: int = _N_FACES,
        face_size: int = _FACE_SIZE,
) -> np.ndarray:
    """Reshape MITgcm global-tile output into face-major form.

    With ``nSx = n_face`` tiles laid out in X (each ``face_size`` wide), a
    global field of shape ``(..., face_size, n_face*face_size)`` is reshaped
    to ``(..., n_face, face_size, face_size)`` by splitting the X axis and
    moving the face index in front of the per-tile spatial axes.
    """
    if arr.shape[-2] != face_size or arr.shape[-1] != n_face * face_size:
        raise ValueError(
            f"_to_face_layout expected (..., {face_size}, {n_face * face_size}); "
            f"got shape {arr.shape}"
        )
    head = arr.shape[:-2]
    arr = arr.reshape(*head, face_size, n_face, face_size)
    return np.moveaxis(arr, -2, -3).copy()


def read_global_ocean_output(
        run_dir: Path,
        cfg: GlobalOceanRunConfig,
) -> dict[str, np.ndarray]:
    """Read tutorial state-dump prefixes into face-major arrays in memory.

    3-D fields come back as ``(time, Nr, n_face, face_size, face_size)`` and
    the 2-D ``Eta`` as ``(time, n_face, face_size, face_size)``.
    """
    run_dir = Path(run_dir)
    iters = _find_mds_iters(run_dir, "T")
    if not iters:
        raise FileNotFoundError(
            f"No T.*.meta files found in {run_dir}. "
            "Did MITgcm produce state dumps? Check dumpFreq in the data namelist."
        )
    return {
        "time": np.asarray(iters, dtype=np.float64) * cfg.delta_t_clock,
        "THETA": _to_face_layout(
            _read_mds_single_field(run_dir, "T", iters),
            n_face=cfg.n_face, face_size=cfg.face_size),
        "SALT": _to_face_layout(
            _read_mds_single_field(run_dir, "S", iters),
            n_face=cfg.n_face, face_size=cfg.face_size),
        "UVEL": _to_face_layout(
            _read_mds_single_field(run_dir, "U", iters),
            n_face=cfg.n_face, face_size=cfg.face_size),
        "VVEL": _to_face_layout(
            _read_mds_single_field(run_dir, "V", iters),
            n_face=cfg.n_face, face_size=cfg.face_size),
        "ETAN": _to_face_layout(
            _read_mds_single_field(run_dir, "Eta", iters),
            n_face=cfg.n_face, face_size=cfg.face_size),
    }


def _level_index(level: int, Nr: int, name: str) -> int:
    if not 1 <= level <= Nr:
        raise ValueError(f"{name} must be in 1..{Nr}, got {level}")
    return level - 1


def depth_centers(del_r: tuple[float, ...] = GLOBAL_OCEAN_DEL_R) -> np.ndarray:
    """Cell-center depth [m] for each vertical level (positive downward)."""
    thicknesses = np.asarray(del_r, dtype=np.float64)
    upper_edges = np.concatenate([[0.0], np.cumsum(thicknesses[:-1])])
    return upper_edges + 0.5 * thicknesses


def extract_global_ocean_fields(
        data: Mapping[str, np.ndarray],
        cfg: GlobalOceanRunConfig,
) -> tuple[list[np.ndarray], list[str], np.ndarray]:
    """Extract the 2-D (per-face) fields written to the benchmark Zarr store."""
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


def extract_global_ocean_fields_3d(
        data: Mapping[str, np.ndarray],
        cfg: GlobalOceanRunConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Extract per-variable 3-D and 2-D arrays for multi-level output.

    Returns:
        fields_3d:  ``{'theta', 'salt', 'u', 'v'}`` each shaped
                    ``(time, level, face, y, x)`` float32.
        fields_2d:  ``{'eta'}`` shaped ``(time, face, y, x)`` float32.
        level_idx:  Selected 1-indexed model levels, shape ``(Nlevel,)``.
        time_arr:   Time coordinate in seconds since collection start.
    """
    if not cfg.levels:
        raise ValueError("extract_global_ocean_fields_3d requires cfg.levels to be set")
    k_indices = np.array(
        [_level_index(int(lvl), cfg.Nr, "levels") for lvl in cfg.levels],
        dtype=int,
    )
    level_idx = np.array(cfg.levels, dtype=np.int64)

    fields_3d = {
        "theta": data["THETA"][:, k_indices].astype(np.float32),
        "salt": data["SALT"][:, k_indices].astype(np.float32),
        "u": data["UVEL"][:, k_indices].astype(np.float32),
        "v": data["VVEL"][:, k_indices].astype(np.float32),
    }
    fields_2d = {"eta": data["ETAN"].astype(np.float32)}

    time_arr = np.asarray(data["time"], dtype=np.float64)
    time_arr = time_arr - time_arr[0]

    return fields_3d, fields_2d, level_idx, time_arr


def write_cubed_sphere_zarr(
        out_path: Path,
        *,
        time_arr: np.ndarray,
        field_arrays: list[np.ndarray],
        field_names: list[str],
        xc: np.ndarray,
        yc: np.ndarray,
        params: Mapping[str, Any] | None = None,
        description: str = "",
) -> None:
    """Write a native cubed-sphere dataset to ``out_path`` (Zarr store).

    Layout: dims ``(time, field, face, y, x)`` with face=6 and y=x=face_size.
    Per-cell longitudes and latitudes are stored as 3-D coordinates ``xc`` /
    ``yc`` of shape ``(face, y, x)``.
    """
    if not field_arrays:
        raise ValueError("write_cubed_sphere_zarr: field_arrays is empty")
    stacked = np.stack(field_arrays, axis=1).astype(np.float32)  # (time, field, face, y, x)

    ds = xr.Dataset(
        data_vars={
            "data": (("time", "field", "face", "y", "x"), stacked),
        },
        coords={
            "time": ("time", np.asarray(time_arr, dtype=np.float64)),
            "field": ("field", np.asarray(field_names)),
            "xc": (("face", "y", "x"), np.asarray(xc, dtype=np.float64)),
            "yc": (("face", "y", "x"), np.asarray(yc, dtype=np.float64)),
        },
        attrs={
            "description": description,
            "grid": "cubed-sphere cs32 (6 faces, 32x32 each)",
            "params": dict(params or {}),
        },
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out_path, mode="w")


def write_cubed_sphere_zarr_3d(
        out_path: Path,
        *,
        time_arr: np.ndarray,
        fields_3d: Mapping[str, np.ndarray],
        fields_2d: Mapping[str, np.ndarray],
        level_idx: np.ndarray,
        depth_m: np.ndarray,
        xc: np.ndarray,
        yc: np.ndarray,
        params: Mapping[str, Any] | None = None,
        description: str = "",
) -> None:
    """Write a multi-level cubed-sphere dataset to ``out_path`` (Zarr store).

    Schema:
      - ``<name>(time, level, face, y, x)`` for each entry in ``fields_3d``.
      - ``<name>(time, face, y, x)`` for each entry in ``fields_2d``.
      - Coords: ``time``, ``level`` (1-indexed model level), ``depth`` (m,
        cell-center, on the ``level`` axis), ``xc`` / ``yc`` ``(face, y, x)``.
    """
    if not fields_3d and not fields_2d:
        raise ValueError("write_cubed_sphere_zarr_3d: no fields supplied")

    n_face, ny, nx = xc.shape
    Nlevel = int(level_idx.size)
    for name, arr in fields_3d.items():
        if arr.shape[1:] != (Nlevel, n_face, ny, nx):
            raise ValueError(
                f"fields_3d[{name!r}] has shape {arr.shape}, "
                f"expected (*, {Nlevel}, {n_face}, {ny}, {nx})."
            )
    for name, arr in fields_2d.items():
        if arr.shape[1:] != (n_face, ny, nx):
            raise ValueError(
                f"fields_2d[{name!r}] has shape {arr.shape}, "
                f"expected (*, {n_face}, {ny}, {nx})."
            )

    data_vars: dict = {}
    for name, arr in fields_3d.items():
        data_vars[name] = (("time", "level", "face", "y", "x"), arr.astype(np.float32))
    for name, arr in fields_2d.items():
        data_vars[name] = (("time", "face", "y", "x"), arr.astype(np.float32))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", np.asarray(time_arr, dtype=np.float64)),
            "level": ("level", level_idx.astype(np.int64)),
            "depth": ("level", depth_m.astype(np.float64)),
            "xc": (("face", "y", "x"), np.asarray(xc, dtype=np.float64)),
            "yc": (("face", "y", "x"), np.asarray(yc, dtype=np.float64)),
        },
        attrs={
            "description": description,
            "grid": f"cubed-sphere cs32 (6 faces, {ny}x{nx} each)",
            "level_units": "1-indexed model level (positive downward)",
            "depth_units": "meters",
            "params": dict(params or {}),
        },
    )

    encoding = {
        name: {"chunks": (1, Nlevel, n_face, ny, nx)} for name in fields_3d
    }
    encoding.update({
        name: {"chunks": (1, n_face, ny, nx)} for name in fields_2d
    })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out_path, mode="w", encoding=encoding)


def write_global_ocean_zarr(
        data: Mapping[str, np.ndarray],
        out_path: Path,
        cfg: GlobalOceanRunConfig,
        params: Mapping[str, Any] | None = None,
) -> None:
    """Write global-ocean output to a native cubed-sphere Zarr store.

    Branches on ``cfg.levels``: when set, emits a per-variable 3-D Zarr
    (theta/salt/u/v at each requested level + 2-D eta); otherwise emits the
    legacy single-level stacked Zarr.
    """
    grid = load_grid_cs32(cfg.grid_dir)

    if cfg.levels:
        fields_3d, fields_2d, level_idx, time_arr = extract_global_ocean_fields_3d(
            data, cfg
        )
        all_depth = depth_centers()
        depth_m = all_depth[level_idx - 1]
        levels_label = ", ".join(str(int(lvl)) for lvl in level_idx)
        write_cubed_sphere_zarr_3d(
            out_path,
            time_arr=time_arr,
            fields_3d=fields_3d,
            fields_2d=fields_2d,
            level_idx=level_idx,
            depth_m=depth_m,
            xc=grid["xc"],
            yc=grid["yc"],
            params=params,
            description=(
                "MITgcm global ocean cubed-sphere (cs32x15) tutorial; "
                f"theta/salt/u/v at levels [{levels_label}] + sea surface height."
            ),
        )
        return

    field_arrays, field_names, time_arr = extract_global_ocean_fields(data, cfg)
    write_cubed_sphere_zarr(
        out_path,
        time_arr=time_arr,
        field_arrays=field_arrays,
        field_names=field_names,
        xc=grid["xc"],
        yc=grid["yc"],
        params=params,
        description=(
            "MITgcm global ocean cubed-sphere (cs32x15) tutorial; "
            "surface temperature/salinity, level-2 velocity, and sea surface height."
        ),
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
        "Starting MITgcm global ocean run: n_iter0=%d, n_timesteps=%d, "
        "dump every %.3g days",
        cfg.n_iter0,
        cfg.n_timesteps,
        cfg.snapshot_interval_days,
    )
    stage_global_ocean_run(run_dir, cfg)
    _launch_mitgcm(run_dir, cfg)

    logger.info("Reading MITgcm MDS output and writing Zarr")
    data = read_global_ocean_output(run_dir, cfg)
    write_global_ocean_zarr(data, out_dir / "run.zarr", cfg, params=params)
