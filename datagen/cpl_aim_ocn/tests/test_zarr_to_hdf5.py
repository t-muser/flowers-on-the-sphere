"""Round-trip test for the ``cpl-aim-ocn`` cubed-sphere HDF5 conversion.

``scripts/zarr_to_hdf5.py`` is a standalone script (not an importable package
module), so we load it by path. The dataset is *archived natively* on the cs32
cubed sphere: ``spatial_dims = (level, face, j, i)`` instead of (lat, lon), with
the physical lon/lat carried in an ``aux_coords/`` group. This test builds a
tiny synthetic cs32 store, converts it, and asserts:

  * the spatial layout, coords, and aux curvilinear coords are written,
  * 3-D atm fields span ``level`` while surface fields keep a size-1 level
    placeholder (``dim_varying[level]=False``), and
  * ``scripts/compute_hdf5_stats.py`` reads the result with per-level stats for
    the leveled fields and reduced stats for the surface ones.

No real run.zarr exists yet, so this synthetic round-trip is the contract that
pins the converter to the schema ``zarr_writer.write_cs32_zarr`` emits.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

_REPO = Path(__file__).resolve().parents[3]


def _load(script: str, mod_name: str):
    """Import a standalone script under ``scripts/`` by file path."""
    spec = importlib.util.spec_from_file_location(
        mod_name, _REPO / "scripts" / script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_z2h = _load("zarr_to_hdf5.py", "zarr_to_hdf5_undertest")
_stats = _load("compute_hdf5_stats.py", "compute_hdf5_stats_undertest")

# The 19 streams the cpl-aim-ocn spec keeps (mirrors channels.py / zarr_writer).
ATM_2D = (
    "atm_TS", "atm_QS", "atm_PRECON", "atm_PRECLS", "atm_WINDS",
    "atm_UFLUX", "atm_VFLUX", "atm_SI_Fract", "atm_SI_Thick",
)
ATM_3D = ("atm_UVEL", "atm_VVEL", "atm_THETA", "atm_SALT")
OCN_2D = (
    "ocn_THETA", "ocn_SALT", "ocn_UVEL", "ocn_VVEL", "ocn_ETAN", "ocn_MXLDEPTH",
)
N_SIGMA = 5


def _synthetic_cs32_zarr(path: Path, *, nt: int = 4) -> Path:
    """Minimal native cs32 store matching ``zarr_writer.write_cs32_zarr``.

    Tiny grid (6 faces x 4 x 4); the σ dim is named ``Zsigma`` (renamed to
    ``level`` by the converter). face/j/i carry NO 1-D coord (as on a real cube)
    so the integer-index fallback is exercised; physical lon/lat live in 2-D
    ``XC``/``YC`` and corner coords in differently-shaped ``XG``/``YG``."""
    nf, ny, nx = 6, 4, 4
    rng = np.random.default_rng(0)
    data: dict[str, tuple] = {}
    for v in ATM_2D + OCN_2D:
        data[v] = (("time", "face", "j", "i"),
                   rng.normal(0, 1, (nt, nf, ny, nx)).astype(np.float32))
    for v in ATM_3D:
        data[v] = (("time", "Zsigma", "face", "j", "i"),
                   rng.normal(0, 1, (nt, N_SIGMA, nf, ny, nx)).astype(np.float32))

    ds = xr.Dataset(
        data_vars=data,
        coords={
            "time": np.arange(nt) * 86400.0,
            "Zsigma": np.linspace(1e5, 2e4, N_SIGMA),  # Pa, TOA->surface-ish
            "XC": (("face", "j", "i"), rng.uniform(0, 360, (nf, ny, nx))),
            "YC": (("face", "j", "i"), rng.uniform(-90, 90, (nf, ny, nx))),
            "XG": (("face", "j_g", "i_g"), rng.uniform(0, 360, (nf, ny + 1, nx + 1))),
            "YG": (("face", "j_g", "i_g"), rng.uniform(-90, 90, (nf, ny + 1, nx + 1))),
        },
    )
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def test_cpl_aim_ocn_cubed_sphere_roundtrip(tmp_path):
    src = _synthetic_cs32_zarr(tmp_path / "run_0000.zarr")
    dst = tmp_path / "out" / "run_0000.h5"
    spec = _z2h.DATASET_SPECS["cpl-aim-ocn"]

    _, status = _z2h.convert_one(src, dst, spec, "spherical")
    assert status == "ok"
    assert dst.exists()

    with h5py.File(dst, "r") as f:
        # ── root + dimensions ──
        assert f.attrs["dataset_name"] == "cpl_aim_ocn"
        assert int(f.attrs["n_spatial_dims"]) == 4
        spatial = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in f["dimensions"].attrs["spatial_dims"]]
        assert spatial == ["level", "face", "j", "i"]
        assert f["dimensions"]["level"].shape == (N_SIGMA,)
        assert f["dimensions"]["face"].shape == (6,)
        # face/j/i had no coord var -> integer-index fallback
        assert np.array_equal(f["dimensions"]["face"][:], np.arange(6))

        # ── aux curvilinear coords preserved verbatim, incl. corner grid ──
        aux = f["aux_coords"]
        assert set(aux.attrs["field_names"].astype(str)) == {"XC", "YC", "XG", "YG"}
        assert aux["XC"].shape == (6, 4, 4)
        assert aux["XG"].shape == (6, 5, 5)

        # ── all 19 streams are t0 scalars; no t1 vectors imposed ──
        t0 = f["t0_fields"]
        names = set(t0.attrs["field_names"].astype(str))
        assert names == set(ATM_2D + ATM_3D + OCN_2D)
        assert len(f["t1_fields"].attrs["field_names"]) == 0

        # ── 3-D atm field spans level; surface fields are size-1 placeholders ──
        theta = t0["atm_THETA"]            # (traj, time, level, face, j, i)
        assert theta.shape == (1, 4, N_SIGMA, 6, 4, 4)
        assert list(theta.attrs["dim_varying"]) == [True, True, True, True]

        ts = t0["atm_TS"]
        assert ts.shape == (1, 4, 1, 6, 4, 4)      # level placeholder = 1
        assert list(ts.attrs["dim_varying"]) == [False, True, True, True]
        assert list(t0["ocn_ETAN"].attrs["dim_varying"]) == [False, True, True, True]

    # ── stats tool reads it: per-level for leveled fields, reduced otherwise ──
    fields, _ = _stats.process_file(str(dst))
    assert fields["atm_THETA"]["per_level"] is True
    assert fields["atm_THETA"]["nlev"] == N_SIGMA
    assert len(fields["atm_THETA"]["cells"]) == N_SIGMA
    assert fields["atm_TS"]["per_level"] is False
    assert len(fields["atm_TS"]["cells"]) == 1


if __name__ == "__main__":  # allow standalone run without pytest collection
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_cpl_aim_ocn_cubed_sphere_roundtrip(Path(d))
    print("OK: cpl-aim-ocn cubed-sphere HDF5 round-trip passed")
