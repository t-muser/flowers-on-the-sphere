C     *=============================================================*
C     | GGL90_OPTIONS.h
C     | o CPP options file for GGL90 package.
C     *=============================================================*

#ifndef GGL90_OPTIONS_H
#define GGL90_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

#ifdef ALLOW_GGL90
C     Package-specific Options & Macros go here

#undef ALLOW_GGL90_HORIZDIFF
#undef ALLOW_GGL90_SMOOTH

#define ALLOW_GGL90_IDEMIX
#ifdef ALLOW_GGL90_IDEMIX
# define GGL90_IDEMIX_CVMIX_VERSION
#endif

#undef ALLOW_GGL90_LANGMUIR
#undef GGL90_MISSING_HFAC_BUG

#endif /* ALLOW_GGL90 */
#endif /* GGL90_OPTIONS_H */
