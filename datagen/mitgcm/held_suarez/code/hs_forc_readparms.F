C     hs_forc_readparms.F - Read Held-Suarez forcing parameters from namelist.
C
C     Called by the local apply_forcing.F override. Reads the HS_FORC_PARM01
C     namelist from data.hs_forc once per process. Default values are set
C     before the read so that an empty namelist gives the standard
C     Held-Suarez (1994) configuration.

#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

      SUBROUTINE HS_FORC_READPARMS( myThid )

C     == Global variables ==
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "hs_forc.h"

C     == Routine arguments ==
      INTEGER myThid

C     == Local variables ==
      INTEGER iUnit, errIO
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      LOGICAL paramsRead
      SAVE paramsRead
      DATA paramsRead /.FALSE./

      NAMELIST /HS_FORC_PARM01/
     &     HS_kf,
     &     HS_ka,
     &     HS_ks,
     &     HS_DeltaT_y,
     &     HS_DeltaT_z,
     &     HS_sigmab,
     &     HS_T0

      IF ( paramsRead ) RETURN

      _BEGIN_MASTER( myThid )

      IF ( .NOT. paramsRead ) THEN
C       Default values (Held-Suarez 1994). Rates are in SI (1/s);
C       the Python namelist writer already converts days.
        HS_kf       = 1.0 _d 0 / 86400.0 _d 0
        HS_ka       = 1.0 _d 0 / ( 40.0 _d 0 * 86400.0 _d 0 )
        HS_ks       = 1.0 _d 0 / (  4.0 _d 0 * 86400.0 _d 0 )
        HS_DeltaT_y = 60.0 _d 0
        HS_DeltaT_z = 10.0 _d 0
        HS_sigmab   =  0.7 _d 0
        HS_T0       = 315.0 _d 0

        CALL OPEN_COPY_DATA_FILE(
     I       'data.hs_forc', 'HS_FORC_READPARMS',
     O       iUnit,
     I       myThid )
        READ( UNIT=iUnit, NML=HS_FORC_PARM01, IOSTAT=errIO )
        IF ( errIO .GT. 0 ) THEN
          WRITE(msgBuf,'(A)')
     &        'HS_FORC_READPARMS: error reading namelist HS_FORC_PARM01'
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R HS_FORC_READPARMS'
        ENDIF
        CLOSE(iUnit)

        WRITE(msgBuf,'(A,1PE14.6)') 'HS_kf       = ', HS_kf
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                       SQUEEZE_RIGHT, myThid )
        WRITE(msgBuf,'(A,1PE14.6)') 'HS_ka       = ', HS_ka
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                       SQUEEZE_RIGHT, myThid )
        WRITE(msgBuf,'(A,1PE14.6)') 'HS_ks       = ', HS_ks
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                       SQUEEZE_RIGHT, myThid )
        WRITE(msgBuf,'(A,F10.4)') 'HS_DeltaT_y = ', HS_DeltaT_y
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                       SQUEEZE_RIGHT, myThid )
        WRITE(msgBuf,'(A,F10.4)') 'HS_DeltaT_z = ', HS_DeltaT_z
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                       SQUEEZE_RIGHT, myThid )
        WRITE(msgBuf,'(A,F10.4)') 'HS_sigmab   = ', HS_sigmab
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                       SQUEEZE_RIGHT, myThid )
        WRITE(msgBuf,'(A,F10.4)') 'HS_T0       = ', HS_T0
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                       SQUEEZE_RIGHT, myThid )

        paramsRead = .TRUE.
      ENDIF

      _END_MASTER( myThid )
      _BARRIER

      RETURN
      END
