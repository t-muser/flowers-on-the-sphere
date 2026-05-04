"""Physical constants for the MITgcm Held-Suarez atmospheric GCM.

MITgcm runs in physical SI units throughout — no internal rescaling like the
Dedalus solvers. All values here match the defaults used in MITgcm's own
Held-Suarez verification experiment and the Held & Suarez (1994) paper.
"""

from __future__ import annotations

# ── Earth geometry ──────────────────────────────────────────────────────────
R_EARTH: float = 6.371e6      # m  — sphere radius
OMEGA:   float = 7.292e-5     # rad/s — rotation rate

# ── Dry-air thermodynamics ───────────────────────────────────────────────────
G:       float = 9.80616      # m/s²  — gravitational acceleration
CP:      float = 1004.0       # J/(kg·K) — specific heat at constant pressure
R_DRY:   float = 287.04       # J/(kg·K) — specific gas constant, dry air
KAPPA:   float = R_DRY / CP   # ≈ 0.2857 — Poisson exponent R/Cp
P0:      float = 1.0e5        # Pa — reference pressure (= surface pressure for HS)

# ── Held-Suarez default forcing parameters ───────────────────────────────────
# These are the Held & Suarez (1994) standard values. They can all be overridden
# at runtime via the data.hs_forc namelist.
HS_KF:        float = 1.0 / 86400.0         # 1/s — surface drag rate (τ_drag = 1 day)
HS_KA:        float = 1.0 / (40.0 * 86400.0) # 1/s — free-atmosphere cooling rate (τ_atm = 40 days)
HS_KS:        float = 1.0 / (4.0 * 86400.0)  # 1/s — surface cooling rate (τ_surf = 4 days)
HS_DELTA_T_Y: float = 60.0   # K — equator-to-pole temperature difference
HS_DELTA_T_Z: float = 10.0   # K — surface-to-tropopause temperature difference
HS_SIGMAB:    float = 0.7    # — boundary-layer top (normalised pressure σ = p/ps)
HS_T0:        float = 315.0  # K — reference surface temperature
HS_T_MIN:     float = 200.0  # K — minimum equilibrium temperature
