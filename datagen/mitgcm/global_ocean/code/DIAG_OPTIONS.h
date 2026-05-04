#ifndef DIAG_OPTIONS_H
#define DIAG_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

#ifdef ALLOW_DIAGNOSTICS
C     Package-specific Options & Macros go here

#define DIAGSTATS_REGION_MASK
#undef  DIAGNOSTICS_HAS_PICKUP

#endif /* ALLOW_DIAGNOSTICS */
#endif /* DIAG_OPTIONS_H */

CEH3 ;;; Local Variables: ***
CEH3 ;;; mode:fortran ***
CEH3 ;;; End: ***
