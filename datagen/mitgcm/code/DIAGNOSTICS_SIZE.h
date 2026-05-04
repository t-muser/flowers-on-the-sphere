C     DIAGNOSTICS_SIZE.h — overrides pkg/diagnostics default to fit 3 3D
C     fields (UVEL, VVEL, THETA) at all Nr levels plus 1 2D field (ETAN),
C     i.e. 3*Nr + 1 = 61 active diagnostics. Default numDiags = 1*Nr = 20
C     is too small.

      INTEGER    ndiagMax
      INTEGER    numlists, numperlist, numLevels
      INTEGER    numDiags
      INTEGER    nRegions, sizRegMsk, nStats
      INTEGER    diagSt_size
      PARAMETER( ndiagMax = 500 )
      PARAMETER( numlists = 10, numperlist = 50, numLevels=2*Nr )
      PARAMETER( numDiags = 4*Nr )
      PARAMETER( nRegions = 0 , sizRegMsk = 1 , nStats = 4 )
      PARAMETER( diagSt_size = 10*Nr )
