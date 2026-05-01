C     hs_forc.h — COMMON block for Held-Suarez forcing parameters.
C
C     All parameters are read at runtime from the data.hs_forc namelist
C     (HS_FORC_PARM01) by hs_forc_readparms.F.  Default values matching
C     Held & Suarez (1994) are set in hs_forc_readparms.F before the
C     namelist read so that omitting data.hs_forc gives the standard run.
C
C     HS_kf     :: Rayleigh surface-drag rate [1/s]  (τ_drag = 1/HS_kf)
C     HS_ka     :: Free-atmosphere Newtonian cooling rate [1/s]
C     HS_ks     :: Surface Newtonian cooling rate [1/s]
C     HS_DeltaT_y :: Equator-to-pole temperature difference ΔTy [K]
C     HS_DeltaT_z :: Surface-to-tropopause temperature difference Δθz [K]
C     HS_sigmab :: Boundary-layer top (normalised pressure σ_b = p/p_s)
C     HS_T0     :: Reference surface temperature T_0 [K]

      COMMON /HS_FORC_PARAMS/
     &     HS_kf,
     &     HS_ka,
     &     HS_ks,
     &     HS_DeltaT_y,
     &     HS_DeltaT_z,
     &     HS_sigmab,
     &     HS_T0
      _RL HS_kf
      _RL HS_ka
      _RL HS_ks
      _RL HS_DeltaT_y
      _RL HS_DeltaT_z
      _RL HS_sigmab
      _RL HS_T0
