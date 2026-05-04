# Vendored upstream provenance

The Fortran/header/configuration files under
`code_atm/`, `code_ocn/`, `code_cpl/`, `shared_code/`, and the three
`build_*/genmake_local` files are **verbatim copies** from the MITgcm
verification experiment `cpl_aim+ocn`.

**Upstream:** https://github.com/MITgcm/MITgcm — `verification/cpl_aim+ocn/`
**Commit:** `242228367a504f25b0b6397c149b6e5261485299`
**Commit date:** 2026-04-24
**Vendored on:** 2026-05-02

We keep these files bit-equal to upstream (no embedded comments) so that
`diff -r` against a fresh checkout cleanly shows when upstream evolves —
necessary for re-vendoring when MITgcm bumps the AIM or coupler packages.

## Documented deviations from upstream

The following changes are intentional and must be re-applied when re-vendoring:

* **`code_atm/DIAGNOSTICS_SIZE.h`** — `numlists` raised from 10 to 30.
* **`code_ocn/DIAGNOSTICS_SIZE.h`** — `numlists` raised from 10 to 30.

  Reason: ``namelist.py::render_atm_diagnostics`` emits one stream per
  field (13 streams atm + 6 streams ocn) because ``xmitgcm.open_mdsdataset``
  with ``geometry='cs'`` cannot consume multi-record MDS files. The
  upstream ``numlists = 10`` would crash with
  ``DIAGNOSTICS_READPARMS: Exceed Max.Num. of list``. Bumping the limit
  to 30 leaves headroom for additional diagnostic streams without
  affecting any other physics or output.

## File-by-file inventory

| File | Source | Purpose |
|---|---|---|
| `code_atm/SIZE.h`              | `verification/cpl_aim+ocn/code_atm/SIZE.h`              | atm grid: cs32, Nr=5, OLx=OLy=2 |
| `code_atm/packages.conf`       | `verification/cpl_aim+ocn/code_atm/packages.conf`       | atm packages: aim_v23, land, thsice, exch2, atm_compon_interf, … |
| `code_atm/CPP_EEOPTIONS.h`     | `verification/cpl_aim+ocn/code_atm/CPP_EEOPTIONS.h`     | exec-env CPP options (MPI flags) |
| `code_atm/CPP_OPTIONS.h`       | `verification/cpl_aim+ocn/code_atm/CPP_OPTIONS.h`       | model-side CPP options |
| `code_atm/DIAGNOSTICS_SIZE.h`  | `verification/cpl_aim+ocn/code_atm/DIAGNOSTICS_SIZE.h`  | diagnostics buffer sizes |
| `code_ocn/SIZE.h`              | `verification/cpl_aim+ocn/code_ocn/SIZE.h`              | ocn grid: cs32, Nr=15, OLx=OLy=4 |
| `code_ocn/packages.conf`       | `verification/cpl_aim+ocn/code_ocn/packages.conf`       | ocn packages: gmredi, thsice, seaice, salt_plume, ocn_compon_interf, … |
| `code_ocn/CPP_EEOPTIONS.h`     | `verification/cpl_aim+ocn/code_ocn/CPP_EEOPTIONS.h`     | exec-env CPP options |
| `code_ocn/CPP_OPTIONS.h`       | `verification/cpl_aim+ocn/code_ocn/CPP_OPTIONS.h`       | model-side CPP options |
| `code_ocn/DIAGNOSTICS_SIZE.h`  | `verification/cpl_aim+ocn/code_ocn/DIAGNOSTICS_SIZE.h`  | diagnostics buffer sizes |
| `code_ocn/SEAICE_OPTIONS.h`    | `verification/cpl_aim+ocn/code_ocn/SEAICE_OPTIONS.h`    | seaice CPP options (used in icedyn variant) |
| `code_cpl/ATMSIZE.h`           | `verification/cpl_aim+ocn/code_cpl/ATMSIZE.h`           | coupler's view of atm grid (Nx_atm=192, Ny_atm=32) |
| `code_cpl/OCNSIZE.h`           | `verification/cpl_aim+ocn/code_cpl/OCNSIZE.h`           | coupler's view of ocn grid (Nx_ocn=192, Ny_ocn=32) |
| `code_cpl/packages.conf`       | `verification/cpl_aim+ocn/code_cpl/packages.conf`       | coupler packages: atm_ocn_coupler + compon_communic |
| `shared_code/ATMIDS.h`         | `verification/cpl_aim+ocn/shared_code/ATMIDS.h`         | string IDs for atm-side coupling fields |
| `shared_code/OCNIDS.h`         | `verification/cpl_aim+ocn/shared_code/OCNIDS.h`         | string IDs for ocn-side coupling fields |
| `build_atm/genmake_local`      | `verification/cpl_aim+ocn/build_atm/genmake_local`      | per-build MODS=`../code_atm ../shared_code` |
| `build_ocn/genmake_local`      | `verification/cpl_aim+ocn/build_ocn/genmake_local`      | per-build MODS=`../code_ocn ../shared_code` |
| `build_cpl/genmake_local`      | `verification/cpl_aim+ocn/build_cpl/genmake_local`      | per-build MODS=`../code_cpl ../shared_code` + STANDARDDIRS="" |
| `templates/atm/{data,data.aimphys,data.cpl,data.ice,data.land,data.pkg,data.shap,eedata}` | `verification/cpl_aim+ocn/input_atm/*`  | atm namelist starting points (substitution-edited per phase / sweep) |
| `templates/ocn/{data,data.cpl,data.diagnostics,data.gmredi,data.pkg,eedata}` | `verification/cpl_aim+ocn/input_ocn/*`  | ocn namelist starting points |
| `templates/cpl/data.cpl`       | `verification/cpl_aim+ocn/input_cpl/data.cpl`       | coupler-side namelist |

## Updating

To re-vendor against a newer upstream:

```bash
MITGCM=/path/to/MITgcm
SRC=$MITGCM/verification/cpl_aim+ocn
DST=datagen/cpl_aim_ocn
cp -v "$SRC/code_atm/"*           "$DST/code_atm/"
cp -v "$SRC/code_ocn/"*           "$DST/code_ocn/"
cp -v "$SRC/code_cpl/"*           "$DST/code_cpl/"
cp -v "$SRC/shared_code/"*        "$DST/shared_code/"
cp -v "$SRC/build_atm/genmake_local"  "$DST/build_atm/"
cp -v "$SRC/build_ocn/genmake_local"  "$DST/build_ocn/"
cp -v "$SRC/build_cpl/genmake_local"  "$DST/build_cpl/"
for f in data data.aimphys data.cpl data.ice data.land data.pkg data.shap eedata; do
  cp -v "$SRC/input_atm/$f" "$DST/templates/atm/"
done
for f in data data.cpl data.diagnostics data.gmredi data.pkg eedata; do
  cp -v "$SRC/input_ocn/$f" "$DST/templates/ocn/"
done
cp -v "$SRC/input_cpl/data.cpl" "$DST/templates/cpl/"
```

Then update the commit hash at the top of this file and re-run
`scripts/build.py`.
