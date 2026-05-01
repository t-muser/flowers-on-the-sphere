C     hs_forc.F — Held-Suarez (1994) idealised atmospheric forcing.
C
C     Implements the two forcing components of the Held-Suarez benchmark:
C
C     1. RAYLEIGH SURFACE FRICTION (HS_FORC_APPLY_UV):
C
C           k_v(σ) = max(0, k_f · (σ − σ_b) / (1 − σ_b))
C           F_u    = −k_v · u
C           F_v    = −k_v · v
C
C        where σ = p / p_s is normalised pressure. Friction is non-zero only
C        in the boundary layer (σ > σ_b).
C
C     2. NEWTONIAN COOLING (HS_FORC_APPLY_T):
C
C           k_T(φ, σ) = k_a + (k_s − k_a) · max(0,(σ−σ_b)/(1−σ_b)) · cos⁴φ
C
C           θ_eq(φ,p) = max(T_min·(p0/p)^κ,
C                           T0 − ΔTy·sin²φ − Δθz·ln(p/p0)·cos²φ)
C
C           F_θ = −k_T · (θ − θ_eq)
C
C        All parameters are runtime-settable via data.hs_forc (hs_forc.h /
C        hs_forc_readparms.F).
C
C     References: Held, I. M. & Suarez, M. J. (1994). A proposal for the
C     intercomparison of the dynamical cores of atmospheric general
C     circulation models. Bull. Amer. Meteor. Soc., 75, 1825–1830.

#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

#ifdef ALLOW_HS_FORC

C ======================================================================
      SUBROUTINE HS_FORC_APPLY_UV(
     I   bi, bj, iMin, iMax, jMin, jMax,
     U   guFld, gvFld,
     I   myTime, myIter, myThid )
C     Apply Held-Suarez Rayleigh friction to u and v tendencies.
C     Called once per timestep by HS_FORC (or packages_step.F).

      IMPLICIT NONE

#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "hs_forc.h"

C     == Arguments ==
      INTEGER bi, bj, iMin, iMax, jMin, jMax
      INTEGER myIter, myThid
      _RL myTime
      _RL guFld(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy)
      _RL gvFld(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy)

C     == Local variables ==
      INTEGER i, j, k
      _RL kv, sigma, pRef_k, pSurf

      pSurf = 1.0 _d 5   ! Reference surface pressure p_0 [Pa]

      DO k = 1, Nr
C       Pressure at level-k centre [Pa].
C       In MITgcm usingPCoords, pC(k) holds the cell-centre pressure.
        pRef_k = pC( k )
        sigma  = pRef_k / pSurf

        IF ( sigma .GT. HS_sigmab ) THEN
          kv = HS_kf * (sigma - HS_sigmab) / (1.0 _d 0 - HS_sigmab)
        ELSE
          kv = 0.0 _d 0
        ENDIF

        DO j = jMin, jMax
          DO i = iMin, iMax
            guFld(i,j,k,bi,bj) = guFld(i,j,k,bi,bj)
     &                          - kv * uVel(i,j,k,bi,bj)
            gvFld(i,j,k,bi,bj) = gvFld(i,j,k,bi,bj)
     &                          - kv * vVel(i,j,k,bi,bj)
          ENDDO
        ENDDO
      ENDDO

      RETURN
      END

C ======================================================================
      SUBROUTINE HS_FORC_APPLY_T(
     I   bi, bj, iMin, iMax, jMin, jMax,
     U   gTFld,
     I   myTime, myIter, myThid )
C     Apply Held-Suarez Newtonian cooling to potential temperature tendency.

      IMPLICIT NONE

#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "hs_forc.h"

C     == Arguments ==
      INTEGER bi, bj, iMin, iMax, jMin, jMax
      INTEGER myIter, myThid
      _RL myTime
      _RL gTFld(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy)

C     == Local variables ==
      INTEGER i, j, k
      _RL kT, sigma, pRef_k, pSurf, kappa_atm
      _RL phi, sin2phi, cos2phi, cos4phi, bnd_frac
      _RL theta_eq, theta_min, log_sigma
      _RL T_min

      pSurf   = 1.0 _d 5            ! p_0 [Pa]
      T_min   = 200.0 _d 0          ! minimum equilibrium temperature [K]
C     Poisson exponent κ = R_dry / Cp
      kappa_atm = atm_Rgas / cp     ! from PARAMS.h

      DO k = 1, Nr
        pRef_k = pC( k )
        sigma  = pRef_k / pSurf
        bnd_frac = MAX( 0.0 _d 0,
     &             (sigma - HS_sigmab) / (1.0 _d 0 - HS_sigmab) )
        log_sigma = LOG( pRef_k / pSurf )

        DO j = jMin, jMax
          phi     = phiC(1,j,bi,bj)     ! latitude in radians
          sin2phi = SIN(phi) ** 2
          cos2phi = COS(phi) ** 2
          cos4phi = cos2phi ** 2

C         Newtonian cooling rate [1/s]
          kT = HS_ka + (HS_ks - HS_ka) * bnd_frac * cos4phi

C         Equilibrium potential temperature [K] — Held & Suarez (1994) Eq.(3)
C         max(T_min·(p0/p)^κ,  T0 − ΔTy·sin²φ − Δθz·ln(p/p0)·cos²φ)
          theta_min = T_min * (pSurf / pRef_k)**kappa_atm
          theta_eq  = HS_T0
     &              - HS_DeltaT_y * sin2phi
     &              - HS_DeltaT_z * log_sigma * cos2phi
          theta_eq  = MAX( theta_min, theta_eq )

          DO i = iMin, iMax
            gTFld(i,j,k,bi,bj) = gTFld(i,j,k,bi,bj)
     &                          - kT * ( theta(i,j,k,bi,bj) - theta_eq )
          ENDDO
        ENDDO
      ENDDO

      RETURN
      END

C ======================================================================
      SUBROUTINE HS_FORC_INIT_FIXED( myThid )
C     Called during model initialisation to read namelist parameters.

      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"

      INTEGER myThid

      CALL HS_FORC_READPARMS( myThid )

      RETURN
      END

C ======================================================================
      SUBROUTINE HS_FORC( bi, bj, iMin, iMax, jMin, jMax,
     I                    myTime, myIter, myThid )
C     Top-level Held-Suarez forcing dispatcher.
C     Called from packages_step.F each timestep for each tile.

      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"

      INTEGER bi, bj, iMin, iMax, jMin, jMax
      INTEGER myIter, myThid
      _RL myTime

      CALL HS_FORC_APPLY_UV(
     I   bi, bj, iMin, iMax, jMin, jMax,
     U   guVel, gvVel,
     I   myTime, myIter, myThid )

      CALL HS_FORC_APPLY_T(
     I   bi, bj, iMin, iMax, jMin, jMax,
     U   gTheta,
     I   myTime, myIter, myThid )

      RETURN
      END

#endif /* ALLOW_HS_FORC */
