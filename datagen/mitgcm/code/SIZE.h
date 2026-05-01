C     SIZE.h — Tile and domain dimensions for the Held-Suarez benchmark.
C
C     Grid: 128 lon × 64 lat × 20 pressure levels.
C     MPI layout: 4 ranks in Y (nPy=4), each owning 128×16 cells.
C     This file is compile-time; changing any value requires recompilation.
C
C     To run at different resolutions (e.g. 256×128), adjust sNx, sNy, and
C     nPy consistently: nPy = Nlat / sNy, and Nlon = sNx (1 tile in X).

      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
C     sNx :: Number of X points per tile
C     sNy :: Number of Y points per tile
C     OLx :: Halo size in X (ghost cells per side)
C     OLy :: Halo size in Y
C     nSx :: Number of tiles per process in X
C     nSy :: Number of tiles per process in Y
C     nPx :: Number of MPI processes in X
C     nPy :: Number of MPI processes in Y
C     Nx  :: Total number of X points  (= sNx * nSx * nPx)
C     Ny  :: Total number of Y points  (= sNy * nSy * nPy)
C     Nr  :: Number of vertical levels
      PARAMETER (
     &           sNx =  128,
     &           sNy =   16,
     &           OLx =    3,
     &           OLy =    3,
     &           nSx =    1,
     &           nSy =    1,
     &           nPx =    1,
     &           nPy =    4,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  =   20)
