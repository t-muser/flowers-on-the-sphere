#include "CPP_OPTIONS.h"

C--  File apply_forcing.F:
C--   Held-Suarez forcing with runtime-settable parameters from data.hs_forc.
C--   This follows MITgcm's verification Held-Suarez forcing hook and uses
C--   the model-core APPLY_FORCING_U/V/T entry points.

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: APPLY_FORCING_U
C     !INTERFACE:
      SUBROUTINE APPLY_FORCING_U(
     U                     gU_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | S/R APPLY_FORCING_U
C     | o Contains Held-Suarez zonal momentum forcing.
C     *==========================================================*
C     \ev

      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "SURFACE.h"
#include "DYNVARS.h"
#include "FFIELDS.h"
#include "hs_forc.h"

      _RL     gU_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid

      INTEGER i, j
CEOP
      _RL recip_P0g, termP, rFullDepth
      _RL kV

      CALL HS_FORC_READPARMS( myThid )

      rFullDepth = rF(1)-rF(Nr+1)
C-jmc: Without CD-scheme, this is OK; with CD-scheme, cover [0:sNy+1].
      DO j=0,sNy+1
       DO i=1,sNx+1
        IF ( maskW(i,j,k,bi,bj).EQ.oneRS ) THEN
         IF ( selectSigmaCoord.EQ.0 ) THEN
          recip_P0g = MAX(recip_Rcol(i,j,bi,bj),recip_Rcol(i-1,j,bi,bj))
          termP = 0.5 _d 0*( MIN( rF(k)*recip_P0g, oneRL )
     &                      +rF(k+1)*recip_P0g )
         ELSE
          termP = aHybSigmC(k)*rFullDepth
#ifdef NONLIN_FRSURF
     &          /(etaHw(i,j,bi,bj)+rSurfW(i,j,bi,bj)-rLowW(i,j,bi,bj))
#else
     &          /(rSurfW(i,j,bi,bj)-rLowW(i,j,bi,bj))
#endif
     &          + bHybSigmC(k)
         ENDIF
         kV = HS_kf*MAX( zeroRL,
     &        (termP-HS_sigmab)/(1. _d 0-HS_sigmab) )
         gU_arr(i,j) = gU_arr(i,j) - kV*uVel(i,j,k,bi,bj)
        ENDIF
       ENDDO
      ENDDO

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: APPLY_FORCING_V
C     !INTERFACE:
      SUBROUTINE APPLY_FORCING_V(
     U                     gV_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | S/R APPLY_FORCING_V
C     | o Contains Held-Suarez meridional momentum forcing.
C     *==========================================================*
C     \ev

      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "SURFACE.h"
#include "DYNVARS.h"
#include "FFIELDS.h"
#include "hs_forc.h"

      _RL     gV_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid

      INTEGER i, j
CEOP
      _RL recip_P0g, termP, rFullDepth
      _RL kV

      CALL HS_FORC_READPARMS( myThid )

      rFullDepth = rF(1)-rF(Nr+1)
      DO j=1,sNy+1
C-jmc: Without CD-scheme, this is OK; with CD-scheme, cover [0:sNx+1].
       DO i=0,sNx+1
        IF ( maskS(i,j,k,bi,bj).EQ.oneRS ) THEN
         IF ( selectSigmaCoord.EQ.0 ) THEN
          recip_P0g = MAX(recip_Rcol(i,j,bi,bj),recip_Rcol(i,j-1,bi,bj))
          termP = 0.5 _d 0*( MIN( rF(k)*recip_P0g, oneRL )
     &                      +rF(k+1)*recip_P0g )
         ELSE
          termP = aHybSigmC(k)*rFullDepth
#ifdef NONLIN_FRSURF
     &          /(etaHs(i,j,bi,bj)+rSurfS(i,j,bi,bj)-rLowS(i,j,bi,bj))
#else
     &          /(rSurfS(i,j,bi,bj)-rLowS(i,j,bi,bj))
#endif
     &          + bHybSigmC(k)
         ENDIF
         kV = HS_kf*MAX( zeroRL,
     &        (termP-HS_sigmab)/(1. _d 0-HS_sigmab) )
         gV_arr(i,j) = gV_arr(i,j) - kV*vVel(i,j,k,bi,bj)
        ENDIF
       ENDDO
      ENDDO

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: APPLY_FORCING_T
C     !INTERFACE:
      SUBROUTINE APPLY_FORCING_T(
     U                     gT_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | S/R APPLY_FORCING_T
C     | o Contains Held-Suarez Newtonian cooling.
C     *==========================================================*
C     \ev

      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "FFIELDS.h"
#include "hs_forc.h"

      _RL     gT_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid

      INTEGER i, j
CEOP
      _RL thetaLim, kT, term1, term2, thetaEq
      _RL termP, rFullDepth

      CALL HS_FORC_READPARMS( myThid )

      rFullDepth = rF(1)-rF(Nr+1)
      DO j=0,sNy+1
       DO i=0,sNx+1
         term1 = HS_DeltaT_y*(SIN(yC(i,j,bi,bj)*deg2rad)**2)
         termP = 0.5 _d 0*( rF(k) + rF(k+1) )
         term2 = HS_DeltaT_z*LOG(termP/atm_po)
     &          *(COS(yC(i,j,bi,bj)*deg2rad)**2)
         thetaLim = 200. _d 0 / ((termP/atm_po)**atm_kappa)
         thetaEq = HS_T0 - term1 - term2
         thetaEq = MAX(thetaLim,thetaEq)

         IF ( selectSigmaCoord.EQ.0 ) THEN
          termP = 0.5 _d 0*( MIN(rF(k),Ro_surf(i,j,bi,bj))
     &                     + rF(k+1) )
     &                    *recip_Rcol(i,j,bi,bj)
         ELSE
          termP = aHybSigmC(k)*rFullDepth
#ifdef NONLIN_FRSURF
     &          /(etaH(i,j,bi,bj)+Ro_surf(i,j,bi,bj)-R_low(i,j,bi,bj))
#else
     &          /(Ro_surf(i,j,bi,bj)-R_low(i,j,bi,bj))
#endif
     &          + bHybSigmC(k)
         ENDIF
         kT = HS_ka+(HS_ks-HS_ka)
     &      *MAX( zeroRL, (termP-HS_sigmab)/(1. _d 0-HS_sigmab) )
     &      *COS((yC(i,j,bi,bj)*deg2rad))**4
         gT_arr(i,j) = gT_arr(i,j)
     &               - kT*( theta(i,j,k,bi,bj)-thetaEq )
     &                *maskC(i,j,k,bi,bj)
       ENDDO
      ENDDO

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: APPLY_FORCING_S
C     !INTERFACE:
      SUBROUTINE APPLY_FORCING_S(
     U                     gS_arr,
     I                     iMin,iMax,jMin,jMax, k, bi, bj,
     I                     myTime, myIter, myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | S/R APPLY_FORCING_S
C     | o No salinity forcing for the dry Held-Suarez benchmark.
C     *==========================================================*
C     \ev

      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"

      _RL     gS_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      INTEGER iMin, iMax, jMin, jMax
      INTEGER k, bi, bj
      _RL     myTime
      INTEGER myIter
      INTEGER myThid

      RETURN
      END
