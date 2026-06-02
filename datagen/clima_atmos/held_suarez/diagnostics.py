"""Physics-validation diagnostics for the ClimaAtmos Held-Suarez sweep.

All 162 corners ran without blowing up, but a stable run can still be
under-resolved or statistically under-converged. This module turns "none
crashed" into an actual validation report, in two stages:

  stage 1 (``reduce``)  — stream one ``run.zarr``, drop the 200-day spin-up,
      and write a small ``run_XXXX.npz`` of time-mean zonal-mean fields,
      eddy statistics, a regridded KE spectrum, and a global-mean ``ps``
      series. Cheap per corner; dispatched as a SLURM array.

  stage 2 (``report``) — load all 162 ``.npz`` and emit PNG panels + a
      ``report.md`` with a per-layer verdict:
        L1  anchor (omega=1, dTy=60, dThz=10; run_0078-0083) vs Held &
            Suarez (1994);
        L2  per-axis physics response (sign / monotonicity);
        L3  internal consistency (seed convergence, hemispheric symmetry);
        numerics — KE spectrum (regridded — see caveat) + ps stationarity.

Methodology and the data constraints behind it are documented in
``HS-notes.md`` and ``docs/held_suarez_clima.md``.

Important data facts (verified on disk, see schema in
``datagen/resample.py::write_latlon_zarr_3d``):
  - Each store holds the **full 565-day run starting at t=0** (2261 6-hourly
    steps); spin-up is NOT excluded upstream, so we drop the first 200 days
    here before any time-averaging.
  - Fields: ``u, v, T (time, level, lat, lon)`` on 8 ERA5 pressure levels,
    ``ps (time, lat, lon)``; lat/lon in degrees; grid ~1.25 deg.
  - The raw native-grid NetCDF was deleted (``--cleanup-clima``), so the
    native-grid KE spectrum and 3-D mass/energy budgets are unavailable —
    the KE spectrum below is computed on the regridded lat-lon field and
    therefore CANNOT see native-grid pile-up (loud caveat in the report).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# ─── constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = Path(
    "/scicore/home/dokman0000/GROUP/PDEDatasets/SphericalPDEs/held-suarez-clima"
)

SECONDS_PER_DAY = 86_400.0
SPINUP_DAYS = 200.0            # docs/held_suarez_clima.md: first 200 d are spin-up
SPINUP_SECONDS = SPINUP_DAYS * SECONDS_PER_DAY

# Mirror postprocess.ERA5_LEVELS_HPA ordering.
LEVELS_HPA: tuple[int, ...] = (50, 100, 250, 500, 700, 850, 925, 1000)
JET_LEVEL_HPA = 250            # midlatitude jet / EKE / spectrum reference level
HEAT_FLUX_LEVEL_HPA = 850      # poleward eddy heat flux peaks in the lower trop

BLOCK = 200                    # timesteps loaded per streaming block

# The anchor regime *is* the canonical Held & Suarez (1994) setup; its 6 seed
# runs are run_0078..run_0083 (omega=1.0, dTy=60, dThz=10), verified on disk.
ANCHOR_PARAMS = {"omega_factor": 1.0, "delta_T_y": 60.0, "delta_theta_z": 10.0}

# Digitized Held & Suarez (1994), BAMS 75:1825, Figs 1-7. These are hand-read
# off the published figures (there is no machine-readable HS94), so each target
# carries a deliberately generous tolerance band.
HS94 = {
    "jet_peak_ms": 28.0, "jet_peak_tol": 7.0,      # westerly jet max ~28 m/s
    "jet_lat_deg": 45.0, "jet_lat_tol": 12.0,      # near +/-45 deg
    "jet_level_hpa": 250,                          # ... at ~250 hPa
    "eq_sfc_T_k": 300.0, "eq_sfc_T_tol": 10.0,     # equatorial 1000 hPa ~295-305 K
    "pole_sfc_T_k": 265.0, "pole_sfc_T_tol": 15.0, # polar 1000 hPa ~260-270 K
}

KE_SPECTRUM_CAVEAT = (
    "Computed on the lat-lon REGRIDDED field, not native he24 spectral output "
    "(the raw NetCDF was deleted by --cleanup-clima). The regrid low-pass-"
    "filters exactly the truncation-scale pile-up this is meant to catch, so a "
    "clean spectrum here is NOT evidence of a resolved run. No verdict is gated "
    "on it."
)

# ─── pure reductions (no I/O — unit-tested on synthetic fields) ─────────────────


def area_weights(lat_deg) -> np.ndarray:
    """cos(lat) area weights for a lat-lon grid (poles down-weighted to ~0)."""
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=float)))
    return np.clip(w, 0.0, None)


def weighted_lat_mean(field: np.ndarray, lat_deg, axis: int = -1) -> np.ndarray:
    """Area-weighted mean over the latitude axis."""
    w = area_weights(lat_deg)
    shape = [1] * field.ndim
    shape[axis] = w.size
    w = w.reshape(shape)
    return np.sum(field * w, axis=axis) / np.sum(w, axis=axis)


def zonal_mean(field: np.ndarray) -> np.ndarray:
    """Mean over the longitude axis (assumed last)."""
    return field.mean(axis=-1)


def _eddy(field: np.ndarray) -> np.ndarray:
    """Deviation from the instantaneous zonal mean (a prime ' quantity)."""
    return field - field.mean(axis=-1, keepdims=True)


def eke(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Eddy kinetic energy 0.5*[u'^2 + v'^2], zonal-averaged. Last axis = lon."""
    return 0.5 * zonal_mean(_eddy(u) ** 2 + _eddy(v) ** 2)


def mke_from_zonal(u_zm: np.ndarray, v_zm: np.ndarray) -> np.ndarray:
    """Mean kinetic energy 0.5*(ubar^2 + vbar^2) of the zonal-mean flow."""
    return 0.5 * (u_zm ** 2 + v_zm ** 2)


def eddy_heat_flux(v: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Poleward eddy heat flux [v'T'] (zonal mean of the eddy product)."""
    return zonal_mean(_eddy(v) * _eddy(T))


def eddy_mom_flux(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Eddy momentum flux [u'v'] (zonal mean of the eddy product)."""
    return zonal_mean(_eddy(u) * _eddy(v))


def jet_metrics(ubar: np.ndarray, lat_deg) -> dict[str, float]:
    """Jet diagnostics from a zonal-mean zonal-wind profile ``ubar(lat)``.

    Returns per-hemisphere westerly peak + latitude, the number of distinct
    westerly jets (local maxima above 5 m/s), and the NH jet full-width at
    half-maximum in degrees.
    """
    lat = np.asarray(lat_deg, dtype=float)
    ubar = np.asarray(ubar, dtype=float)
    nh, sh = lat > 0, lat < 0
    out: dict[str, float] = {
        "jet_max_nh": float(ubar[nh].max()),
        "jet_lat_nh": float(lat[nh][np.argmax(ubar[nh])]),
        "jet_max_sh": float(ubar[sh].max()),
        "jet_lat_sh": float(lat[sh][np.argmax(ubar[sh])]),
    }
    thr = 5.0
    peaks = [
        i for i in range(1, len(ubar) - 1)
        if ubar[i] > ubar[i - 1] and ubar[i] >= ubar[i + 1] and ubar[i] > thr
    ]
    out["n_jets"] = float(len(peaks))
    out["jet_fwhm_nh"] = _fwhm_deg(lat, ubar, nh)
    return out


def _fwhm_deg(lat: np.ndarray, ubar: np.ndarray, mask: np.ndarray) -> float:
    """Full-width at half-max (deg) of the dominant peak within ``mask``."""
    sub_lat, sub_u = lat[mask], ubar[mask]
    if sub_u.size == 0:
        return float("nan")
    i_peak = int(np.argmax(sub_u))
    half = sub_u[i_peak] / 2.0
    above = sub_u >= half
    # contiguous run of `above` containing the peak
    lo = i_peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i_peak
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    return float(abs(sub_lat[hi] - sub_lat[lo]))


def lon_power_spectrum(field: np.ndarray) -> np.ndarray:
    """One-sided zonal-wavenumber power spectrum along the last (lon) axis."""
    nlon = field.shape[-1]
    fk = np.fft.rfft(field, axis=-1) / nlon
    p = np.abs(fk) ** 2
    if p.shape[-1] > 2:
        p[..., 1:-1] *= 2.0  # fold negative wavenumbers onto positive
    return p


def ke_spectrum(u: np.ndarray, v: np.ndarray, lat_deg):
    """Area-weighted, leading-dim-averaged KE power spectrum vs zonal wavenumber.

    ``u``/``v`` are ``(..., lat, lon)`` at one level. Returns ``(k, power)``.
    """
    p = 0.5 * (lon_power_spectrum(u) + lon_power_spectrum(v))   # (..., lat, nk)
    p = weighted_lat_mean(p, lat_deg, axis=-2)                  # (..., nk)
    p = p.reshape(-1, p.shape[-1]).mean(axis=0)                 # (nk,)
    return np.arange(p.shape[-1]), p


def hemispheric_asymmetry(field: np.ndarray, lat_deg) -> float:
    """Normalized RMS asymmetry of a field about the equator (0 = symmetric).

    Assumes ``lat_deg`` is an equator-symmetric grid so a simple flip of the
    lat axis (assumed last) mirrors the hemispheres.
    """
    mirror = field[..., ::-1]
    num = np.sqrt(np.mean((field - mirror) ** 2))
    den = np.sqrt(np.mean(field ** 2)) + 1e-12
    return float(num / den)


def global_mean_ps_series(ps: np.ndarray, lat_deg) -> np.ndarray:
    """Area-weighted global-mean surface pressure per timestep, ``ps(time,lat,lon)``."""
    return weighted_lat_mean(ps.mean(axis=-1), lat_deg, axis=-1)


# ─── stage 1: streaming per-corner reduction ───────────────────────────────────


def reduce_corner(zarr_path: Path) -> dict:
    """Stream one ``run.zarr``, drop spin-up, and return reduced arrays + scalars.

    Memory-bounded: loads ``BLOCK`` timesteps at a time and accumulates running
    sums; never holds the whole store.
    """
    ds = xr.open_zarr(str(zarr_path), decode_times=False)
    lat = ds["lat"].values.astype(float)
    level = ds["level"].values.astype(float)
    time = ds["time"].values.astype(float)
    nlat, nlon = lat.size, ds["lon"].size

    keep = np.flatnonzero(time >= SPINUP_SECONDS)
    if keep.size == 0:
        raise ValueError(
            f"{zarr_path}: no timesteps past the {SPINUP_DAYS:.0f}-day spin-up "
            f"(time max = {time.max() / SECONDS_PER_DAY:.1f} d)"
        )
    t0, t1 = int(keep[0]), int(keep[-1]) + 1
    n_keep = t1 - t0
    jet_idx = LEVELS_HPA.index(JET_LEVEL_HPA)

    nlev = level.size
    acc = {k: np.zeros((nlev, nlat)) for k in ("u_zm", "v_zm", "T_zm", "eke", "vT", "uv")}
    spec_sum = None
    spec_n = 0
    ps_series = np.empty(n_keep, dtype=float)
    ps_filled = 0
    n_time = 0

    for s in range(t0, t1, BLOCK):
        e = min(s + BLOCK, t1)
        sl = slice(s, e)
        u = ds["u"].isel(time=sl).values.astype(np.float64)   # (b, lev, lat, lon)
        v = ds["v"].isel(time=sl).values.astype(np.float64)
        T = ds["T"].isel(time=sl).values.astype(np.float64)
        ps = ds["ps"].isel(time=sl).values.astype(np.float64)  # (b, lat, lon)
        b = u.shape[0]

        acc["u_zm"] += zonal_mean(u).sum(axis=0)
        acc["v_zm"] += zonal_mean(v).sum(axis=0)
        acc["T_zm"] += zonal_mean(T).sum(axis=0)
        acc["eke"] += eke(u, v).sum(axis=0)
        acc["vT"] += eddy_heat_flux(v, T).sum(axis=0)
        acc["uv"] += eddy_mom_flux(u, v).sum(axis=0)

        _, p_blk = ke_spectrum(u[:, jet_idx], v[:, jet_idx], lat)
        spec_sum = p_blk * b if spec_sum is None else spec_sum + p_blk * b
        spec_n += b

        ps_series[ps_filled:ps_filled + b] = global_mean_ps_series(ps, lat)
        ps_filled += b
        n_time += b

    for k in acc:
        acc[k] /= n_time
    spec_power = spec_sum / spec_n
    k_wave = np.arange(spec_power.size)

    u_zm = acc["u_zm"]
    mke = mke_from_zonal(u_zm, acc["v_zm"])
    ps_time = (time[t0:t1] - time[t0]) / SECONDS_PER_DAY  # days into analysis window

    p = {kk: float(ds.attrs[f"param_{kk}"]) for kk in
         ("omega_factor", "delta_T_y", "delta_theta_z", "seed")}

    return {
        "run_id": int(ds.attrs["run_id"]),
        "lat": lat, "level": level, "k_wave": k_wave,
        "u_zm": u_zm, "v_zm": acc["v_zm"], "T_zm": acc["T_zm"],
        "eke": acc["eke"], "mke": mke, "vT": acc["vT"], "uv": acc["uv"],
        "ke_spectrum": spec_power, "spectrum_level_hpa": float(JET_LEVEL_HPA),
        "ps_global_mean": ps_series, "ps_time_days": ps_time,
        "n_time": n_time, "n_total": time.size, "n_spinup_dropped": t0,
        "omega_factor": p["omega_factor"], "delta_T_y": p["delta_T_y"],
        "delta_theta_z": p["delta_theta_z"], "seed": p["seed"],
        "ke_spectrum_caveat": KE_SPECTRUM_CAVEAT,
    }


def write_corner_npz(out_path: Path, reduced: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **reduced)
    log.info("wrote %s (%d post-spin-up steps, dropped %d)",
             out_path, reduced["n_time"], reduced["n_spinup_dropped"])


# ─── stage 2: scalars + report ─────────────────────────────────────────────────


def corner_scalars(d) -> dict[str, float]:
    """Headline scalars used by Layers 1-3, computed from a reduced corner."""
    lat = d["lat"]
    jet_idx = LEVELS_HPA.index(JET_LEVEL_HPA)
    heat_idx = LEVELS_HPA.index(HEAT_FLUX_LEVEL_HPA)
    u250 = d["u_zm"][jet_idx]
    jm = jet_metrics(u250, lat)
    eke250 = weighted_lat_mean(d["eke"][jet_idx], lat)
    mke250 = weighted_lat_mean(d["mke"][jet_idx], lat)
    # Lower-trop EKE (850 hPa) is the right baroclinic-eddy proxy: that is
    # where baroclinic generation lives. 250-hPa EKE can rise with static
    # stability even as eddies weaken, because they become shallower.
    eke850 = weighted_lat_mean(d["eke"][heat_idx], lat)
    return {
        "jet_max": jm["jet_max_nh"],
        "jet_lat": jm["jet_lat_nh"],          # NH jet latitude (deg); ↓ = equatorward
        "n_jets": jm["n_jets"],
        "jet_fwhm": jm["jet_fwhm_nh"],
        "eke_250": float(eke250),
        "eke_850": float(eke850),
        "eke_mke_ratio": float(eke250 / (mke250 + 1e-12)),
        "vT_max": float(np.max(np.abs(d["vT"][heat_idx]))),
        "asym_u": hemispheric_asymmetry(d["u_zm"], lat),
        "ps_drift_pa_per_day": _linear_slope(d["ps_time_days"], d["ps_global_mean"]),
    }


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _load_corners(diag_dir: Path) -> dict[int, dict]:
    corners: dict[int, dict] = {}
    for npz in sorted(diag_dir.glob("run_*.npz")):
        with np.load(npz, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        corners[int(d["run_id"])] = d
    return corners


def _regime_id(run_id: int) -> int:
    return run_id // 6


def build_report(diag_dir: Path, report_dir: Path) -> None:
    """Load all reduced corners and emit PNG panels + report.md."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corners = _load_corners(diag_dir)
    if not corners:
        raise FileNotFoundError(f"no run_*.npz under {diag_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    scal = {rid: corner_scalars(d) for rid, d in corners.items()}
    lines: list[str] = ["# Held-Suarez ClimaAtmos validation report", ""]
    lines.append(f"Corners reduced: **{len(corners)} / 162**. "
                 f"Spin-up dropped: first {SPINUP_DAYS:.0f} days.")
    lines.append("")

    anchor_ids = [r for r in range(78, 84) if r in corners]
    _layer1_anchor(corners, scal, anchor_ids, report_dir, plt, lines)
    _layer2_response(corners, scal, report_dir, plt, lines)
    _layer3_consistency(corners, scal, report_dir, plt, lines)
    _numerics(corners, scal, report_dir, plt, lines)
    _not_checked(lines)

    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote report → %s", report_dir / "report.md")


def _find_regime(corners, omega, dTy, dThz) -> list[int]:
    out = []
    for rid, d in corners.items():
        if (float(d["omega_factor"]) == omega and float(d["delta_T_y"]) == dTy
                and float(d["delta_theta_z"]) == dThz):
            out.append(rid)
    return sorted(out)


def _mean_std(scal, ids, key):
    vals = np.array([scal[i][key] for i in ids], dtype=float)
    return float(vals.mean()), float(vals.std()), vals


def _layer1_anchor(corners, scal, anchor_ids, report_dir, plt, lines):
    lines += ["## Layer 1 — Anchor vs Held & Suarez (1994)", ""]
    if not anchor_ids:
        lines += ["**SKIPPED** — anchor corners (run_0078-0083) not present.", ""]
        return
    lat = corners[anchor_ids[0]]["lat"]
    jet_idx = LEVELS_HPA.index(JET_LEVEL_HPA)
    sfc_idx = LEVELS_HPA.index(1000)
    # seed-mean fields
    u_zm = np.mean([corners[i]["u_zm"] for i in anchor_ids], axis=0)
    T_zm = np.mean([corners[i]["T_zm"] for i in anchor_ids], axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    levels = corners[anchor_ids[0]]["level"]
    cs = ax[0].contourf(lat, levels, u_zm, levels=21, cmap="RdBu_r")
    ax[0].invert_yaxis(); ax[0].set_title("anchor zonal-mean u (m/s)")
    ax[0].set_xlabel("lat"); ax[0].set_ylabel("level (hPa)")
    fig.colorbar(cs, ax=ax[0])
    ax[1].plot(lat, u_zm[jet_idx], label=f"u @ {JET_LEVEL_HPA} hPa")
    ax[1].axhline(HS94["jet_peak_ms"], color="k", ls="--", lw=0.8,
                  label=f"HS94 jet ~{HS94['jet_peak_ms']:.0f} m/s")
    for s in (-1, 1):
        ax[1].axvspan(s * HS94["jet_lat_deg"] - HS94["jet_lat_tol"],
                      s * HS94["jet_lat_deg"] + HS94["jet_lat_tol"],
                      color="orange", alpha=0.15)
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_xlabel("lat"); ax[1].set_ylabel("u (m/s)"); ax[1].legend(fontsize=8)
    ax[1].set_title("anchor jet vs HS94 target")
    fig.tight_layout(); fig.savefig(report_dir / "L1_anchor_u.png", dpi=110)
    plt.close(fig)

    jet, jet_std, _ = _mean_std(scal, anchor_ids, "jet_max")
    jlat, jlat_std, _ = _mean_std(scal, anchor_ids, "jet_lat")
    eq_mask = np.abs(lat) < 10
    eq_T = float(weighted_lat_mean(T_zm[sfc_idx, eq_mask], lat[eq_mask]))
    pole_T = float(T_zm[sfc_idx, np.abs(lat) > 75].mean())

    def _chk(val, target, tol):
        return "PASS" if abs(val - target) <= tol else "FAIL"

    rows = [
        ("jet peak (m/s)", jet, HS94["jet_peak_ms"], HS94["jet_peak_tol"]),
        ("jet lat (deg)", jlat, HS94["jet_lat_deg"], HS94["jet_lat_tol"]),
        ("eq sfc T (K)", eq_T, HS94["eq_sfc_T_k"], HS94["eq_sfc_T_tol"]),
        ("pole sfc T (K)", pole_T, HS94["pole_sfc_T_k"], HS94["pole_sfc_T_tol"]),
    ]
    lines += ["![anchor](L1_anchor_u.png)", "",
              "| metric | anchor | HS94 target | tol | verdict |",
              "|---|---|---|---|---|"]
    verdicts = []
    for name, val, tgt, tol in rows:
        v = _chk(val, tgt, tol)
        verdicts.append(v)
        lines.append(f"| {name} | {val:.1f} | {tgt:.0f} | ±{tol:.0f} | {v} |")
    ok = all(v == "PASS" for v in verdicts)
    lines += ["", f"**Verdict:** {'PASS' if ok else 'CHECK'} — anchor "
              f"{'matches' if ok else 'partially matches'} HS94 within tolerance; "
              "this validates the full chain (config → seed patch → level "
              "selection → Zarr) at one point.", ""]


def _layer2_response(corners, scal, report_dir, plt, lines):
    lines += ["## Layer 2 — Per-axis physics response", ""]
    # Expected SIGNS only (not magnitudes/exponents), per HS-notes.md + thermal
    # wind / Eady physics. Ω: higher rotation → more, narrower, equatorward jets,
    # and weaker peak jet (higher f → less shear for fixed ΔT_y). ΔT_y: stronger
    # baroclinicity → stronger jet, more (lower-trop) EKE & eddy heat flux.
    # Δθ_z: more static stability → weaker baroclinic eddies (lower EKE_850),
    # energy shifts toward the mean flow (EKE/MKE ↓).
    axes = [
        ("omega_factor", [0.5, 1.0, 2.0], 60.0, 10.0,
         {"n_jets": "incr", "jet_lat": "decr", "jet_fwhm": "decr",
          "jet_max": "decr"}),
        ("delta_T_y", [40.0, 60.0, 80.0], None, None,
         {"jet_max": "incr", "eke_850": "incr", "vT_max": "incr"}),
        ("delta_theta_z", [5.0, 10.0, 20.0], None, None,
         {"eke_850": "decr", "eke_mke_ratio": "decr"}),
    ]
    fig, axarr = plt.subplots(1, 3, figsize=(14, 4))
    verdict_rows = []
    for ax_i, (name, levs, dTy, dThz, expect) in enumerate(axes):
        # hold the other two axes at the anchor center
        c = {"omega_factor": 1.0, "delta_T_y": 60.0, "delta_theta_z": 10.0}
        xs, series = [], {k: ([], []) for k in expect}
        for lv in levs:
            sel = dict(c); sel[name] = lv
            ids = _find_regime(corners, sel["omega_factor"],
                               sel["delta_T_y"], sel["delta_theta_z"])
            if not ids:
                continue
            xs.append(lv)
            for key in expect:
                m, s, _ = _mean_std(scal, ids, key)
                series[key][0].append(m); series[key][1].append(s)
        for key, (m, s) in series.items():
            if len(m) >= 2:
                axarr[ax_i].errorbar(xs, m, yerr=s, marker="o", capsize=3,
                                     label=key)
                trend = "incr" if m[-1] > m[0] else "decr"
                ok = trend == expect[key]
                verdict_rows.append((name, key, expect[key], trend,
                                     "PASS" if ok else "FAIL"))
        axarr[ax_i].set_title(name); axarr[ax_i].set_xlabel(name)
        axarr[ax_i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(report_dir / "L2_response.png", dpi=110)
    plt.close(fig)

    lines += ["![response](L2_response.png)", "",
              "Direction only (NOT scaling exponents); error bars = seed spread.",
              "", "| axis | metric | expected | observed | verdict |",
              "|---|---|---|---|---|"]
    for name, key, exp, obs, v in verdict_rows:
        lines.append(f"| {name} | {key} | {exp} | {obs} | {v} |")
    ok = all(r[-1] == "PASS" for r in verdict_rows)
    lines += ["", f"**Verdict:** {'PASS' if ok else 'CHECK'} — responses "
              f"{'have' if ok else 'do not all have'} the expected sign; a "
              "wrong-sign axis flags a broken (often under-resolved) corner.", ""]


def _layer3_consistency(corners, scal, report_dir, plt, lines):
    lines += ["## Layer 3 — Internal consistency", ""]
    # group by regime
    regimes: dict[int, list[int]] = {}
    for rid in corners:
        regimes.setdefault(_regime_id(rid), []).append(rid)
    key = "jet_max"
    seed_spreads, regime_means = [], []
    for ids in regimes.values():
        _, s, vals = _mean_std(scal, ids, key)
        seed_spreads.append(s); regime_means.append(vals.mean())
    seed_spread = float(np.mean(seed_spreads))
    regime_spread = float(np.std(regime_means))
    ratio = seed_spread / (regime_spread + 1e-12)

    asym = np.array([scal[r]["asym_u"] for r in corners])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(len(regimes)), [s for s in seed_spreads])
    ax[0].axhline(regime_spread, color="r", ls="--",
                  label=f"regime-to-regime σ={regime_spread:.1f}")
    ax[0].set_title(f"seed spread of {key} per regime"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("regime"); ax[0].set_ylabel("m/s")
    ax[1].hist(asym, bins=20); ax[1].set_title("hemispheric asymmetry of u_zm")
    ax[1].set_xlabel("normalized RMS asym")
    fig.tight_layout(); fig.savefig(report_dir / "L3_consistency.png", dpi=110)
    plt.close(fig)

    conv = "PASS" if ratio < 0.5 else "CHECK"
    sym = "PASS" if float(np.median(asym)) < 0.25 else "CHECK"
    lines += [
        "![consistency](L3_consistency.png)", "",
        f"- **Seed convergence:** mean seed spread of `{key}` = "
        f"{seed_spread:.2f} m/s vs regime-to-regime spread {regime_spread:.2f} "
        f"m/s (ratio {ratio:.2f}). {conv} — small ratio means 365 d converged "
        "the statistics and the Layer-2 signal is not sampling noise.",
        f"- **Hemispheric symmetry:** median asymmetry {float(np.median(asym)):.3f} "
        f"(max {float(asym.max()):.3f}). {sym} — HS forcing is equator-symmetric, "
        "so residual asymmetry beyond seed spread flags under-averaging or a bug.",
        "", f"**Verdict:** {conv} / {sym}.", "",
    ]


def _numerics(corners, scal, report_dir, plt, lines):
    lines += ["## Numerics — KE spectrum & ps stationarity", ""]
    # one representative corner per omega level
    reps = {}
    for om in (0.5, 1.0, 2.0):
        ids = [r for r, d in corners.items() if float(d["omega_factor"]) == om]
        if ids:
            reps[om] = ids[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    upturn_flags = []
    for om, rid in reps.items():
        d = corners[rid]
        k, p = d["k_wave"], d["ke_spectrum"]
        ax[0].loglog(k[1:], p[1:], label=f"Ω={om}× (run_{rid:04d})")
        # crude grid-scale pile-up flag: last-decade mean above mid-decade mean
        hi = p[int(0.8 * len(p)):].mean()
        mid = p[int(0.4 * len(p)):int(0.6 * len(p))].mean()
        upturn_flags.append((om, hi > mid))
    ax[0].set_title("regridded KE spectrum @ 250 hPa")
    ax[0].set_xlabel("zonal wavenumber"); ax[0].set_ylabel("power")
    ax[0].legend(fontsize=8)

    drifts = []
    for om, rid in reps.items():
        d = corners[rid]
        ax[1].plot(d["ps_time_days"], d["ps_global_mean"], label=f"Ω={om}×")
        drifts.append((om, scal[rid]["ps_drift_pa_per_day"]))
    ax[1].set_title("global-mean ps (mass proxy)")
    ax[1].set_xlabel("days into analysis window"); ax[1].set_ylabel("ps (Pa)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(report_dir / "numerics.png", dpi=110)
    plt.close(fig)

    lines += ["![numerics](numerics.png)", "",
              f"> **KE-spectrum caveat:** {KE_SPECTRUM_CAVEAT}", ""]
    lines.append("Grid-scale upturn flags (regridded — indicative only): " +
                 ", ".join(f"Ω={om}: {'UPTURN' if f else 'clean'}"
                           for om, f in upturn_flags))
    lines.append("")
    lines.append("ps drift (area-weighted global mean, a dry-mass proxy): " +
                 ", ".join(f"Ω={om}: {dr:+.3f} Pa/day" for om, dr in drifts))
    max_drift = max(abs(dr) for _, dr in drifts) if drifts else float("nan")
    ps_v = "PASS" if max_drift < 5.0 else "CHECK"
    lines += ["", f"**Verdict:** ps stationarity {ps_v} (max |drift| "
              f"{max_drift:.3f} Pa/day). KE spectrum reported, no verdict gated "
              "on it (see caveat).", ""]


def _not_checked(lines):
    lines += [
        "## Not checkable from surviving artifacts", "",
        "- **Native-grid KE spectrum** — needs the he24 spectral output, deleted "
        "by `--cleanup-clima`. The regridded spectrum above cannot substitute.",
        "- **3-D dry-mass / total-energy budgets** — need native output; only the "
        "global-mean `ps` proxy is available.",
        "- **Sub-1000 hPa / between-level structure** — only the 8 ERA5 levels "
        "were emitted.",
        "",
    ]


# ─── CLI ────────────────────────────────────────────────────────────────────────


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _data_root(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    return Path(os.environ.get("DATA_ROOT", str(DEFAULT_DATA_ROOT)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("reduce", help="reduce one corner → run_XXXX.npz")
    pr.add_argument("--run-id", required=True,
                    help="zero-padded corner id, e.g. 0078")
    pr.add_argument("--data-root", type=Path, default=None)
    pr.add_argument("--out", type=Path, default=None,
                    help="override output npz path")

    prep = sub.add_parser("report", help="aggregate all npz → report.md + PNGs")
    prep.add_argument("--data-root", type=Path, default=None)
    prep.add_argument("--diag-dir", type=Path, default=None)
    prep.add_argument("--report-dir", type=Path, default=None)

    args = ap.parse_args()
    _setup_logging()
    data_root = _data_root(args.data_root)
    diag_dir = data_root / "diagnostics"

    if args.cmd == "reduce":
        rid = f"{int(args.run_id):04d}"
        zarr_path = data_root / "runs" / f"run_{rid}" / "run.zarr"
        out = args.out or (diag_dir / f"run_{rid}.npz")
        log.info("reducing %s", zarr_path)
        write_corner_npz(out, reduce_corner(zarr_path))
    else:
        dd = args.diag_dir or diag_dir
        rd = args.report_dir or (dd / "report")
        build_report(dd, rd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
