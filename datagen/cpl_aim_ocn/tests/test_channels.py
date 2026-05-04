"""Tests for ``datagen/cpl_aim_ocn/channels.py``.

Both the finalize/stats step and the fots-side dataloader rely on
`channel_names()` agreeing with `expand_to_channels()` — a mismatch
silently produces the wrong stats per channel. These tests are the
contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from datagen.cpl_aim_ocn.channels import (
    ATM_2D_STREAMS,
    ATM_3D_STREAMS,
    ATM_VERTICAL_DIM,
    N_SIGMA,
    OCN_2D_STREAMS,
    channel_names,
    expand_to_channels,
)


# ─── Constants & ordering ───────────────────────────────────────────────────


class TestChannelNames:
    def test_total_count_is_35(self):
        names = channel_names()
        assert len(names) == 35

    def test_count_decomposes_correctly(self):
        n = len(ATM_2D_STREAMS) + len(ATM_3D_STREAMS) * N_SIGMA + len(OCN_2D_STREAMS)
        assert n == len(channel_names())

    def test_no_duplicates(self):
        names = channel_names()
        assert len(set(names)) == len(names)

    def test_atm_2d_streams_first(self):
        names = channel_names()
        for i, s in enumerate(ATM_2D_STREAMS):
            assert names[i] == s

    def test_atm_3d_expanded_per_sigma(self):
        names = channel_names()
        offset = len(ATM_2D_STREAMS)
        for v_idx, v in enumerate(ATM_3D_STREAMS):
            for k in range(N_SIGMA):
                expected = f"{v}_s{k + 1}"
                assert names[offset + v_idx * N_SIGMA + k] == expected

    def test_ocn_streams_last(self):
        names = channel_names()
        for i, s in enumerate(OCN_2D_STREAMS):
            assert names[-len(OCN_2D_STREAMS) + i] == s


# ─── expand_to_channels ─────────────────────────────────────────────────────


def _synthetic_cube(nt: int = 2, nr: int = N_SIGMA) -> xr.Dataset:
    """Build the minimal cs32 dataset shape ``expand_to_channels`` needs.

    Mirrors what ``zarr_writer.write_cs32_zarr`` actually emits:
    * each ATM_2D_STREAMS var has dims ``(time, face, j, i)``
    * each ATM_3D_STREAMS var has dims ``(time, Zsigma, face, j, i)``
    * each OCN_2D_STREAMS var has dims ``(time, face, j, i)``
    """
    nf, ny, nx = 6, 4, 4   # tiny grid; full cs32 not needed for shape tests
    rng = np.random.default_rng(0)

    data: dict[str, tuple] = {}
    for v in ATM_2D_STREAMS:
        data[v] = (("time", "face", "j", "i"),
                   rng.normal(0, 1, (nt, nf, ny, nx)).astype(np.float32))
    for v in ATM_3D_STREAMS:
        data[v] = (("time", ATM_VERTICAL_DIM, "face", "j", "i"),
                   rng.normal(0, 1, (nt, nr, nf, ny, nx)).astype(np.float32))
    for v in OCN_2D_STREAMS:
        data[v] = (("time", "face", "j", "i"),
                   rng.normal(0, 1, (nt, nf, ny, nx)).astype(np.float32))

    return xr.Dataset(
        data_vars=data,
        coords={
            "time": np.arange(nt) * 86400.0,
            ATM_VERTICAL_DIM: np.linspace(0, 1, nr),
        },
    )


class TestExpandToChannels:
    def test_returns_dataarray_named_fields(self):
        ds = _synthetic_cube()
        out = expand_to_channels(ds)
        assert isinstance(out, xr.DataArray)
        assert out.name == "fields"

    def test_channel_dim_present_with_correct_size(self):
        ds = _synthetic_cube()
        out = expand_to_channels(ds)
        assert "channel" in out.dims
        assert out.sizes["channel"] == 35

    def test_channel_coord_matches_channel_names(self):
        ds = _synthetic_cube()
        out = expand_to_channels(ds)
        assert tuple(out.coords["channel"].values.tolist()) == channel_names()

    def test_face_j_i_dims_preserved(self):
        ds = _synthetic_cube()
        out = expand_to_channels(ds)
        for d in ("time", "face", "j", "i"):
            assert d in out.dims

    def test_atm_3d_split_per_sigma(self):
        """A 3D atm var split into σ levels must match isel of the source."""
        ds = _synthetic_cube()
        out = expand_to_channels(ds)
        names = channel_names()
        # atm_UVEL is the first 3D stream → channels offset..offset+N_SIGMA-1.
        offset = len(ATM_2D_STREAMS)
        for k in range(N_SIGMA):
            ch_idx = offset + k
            ch_name = names[ch_idx]
            assert ch_name == f"atm_UVEL_s{k + 1}"
            level_in = ds["atm_UVEL"].isel({ATM_VERTICAL_DIM: k}).values
            level_out = out.isel(channel=ch_idx).values
            assert np.array_equal(level_in, level_out)

    def test_atm_2d_passes_through(self):
        ds = _synthetic_cube()
        out = expand_to_channels(ds)
        # atm_TS is channel 0.
        assert np.array_equal(ds["atm_TS"].values, out.isel(channel=0).values)

    def test_ocn_2d_at_tail(self):
        ds = _synthetic_cube()
        out = expand_to_channels(ds)
        # ocn_THETA is the first ocn var (channel 35 - len(OCN_2D)).
        first_ocn_idx = 35 - len(OCN_2D_STREAMS)
        assert np.array_equal(
            ds["ocn_THETA"].values,
            out.isel(channel=first_ocn_idx).values,
        )

    def test_missing_atm_2d_raises_keyerror(self):
        ds = _synthetic_cube().drop_vars("atm_TS")
        with pytest.raises(KeyError, match="atm_TS"):
            expand_to_channels(ds)

    def test_missing_atm_3d_raises_keyerror(self):
        ds = _synthetic_cube().drop_vars("atm_THETA")
        with pytest.raises(KeyError, match="atm_THETA"):
            expand_to_channels(ds)

    def test_wrong_sigma_count_raises(self):
        # Build dataset with only 3 σ levels — should fail loudly.
        ds = _synthetic_cube(nr=3)
        with pytest.raises(ValueError, match=r"σ levels"):
            expand_to_channels(ds)

    def test_missing_vertical_dim_raises(self):
        ds = _synthetic_cube()
        # Squeeze the σ dim off atm_UVEL — now it's 2D where we expected 3D.
        ds = ds.assign(atm_UVEL=ds["atm_UVEL"].isel({ATM_VERTICAL_DIM: 0}, drop=True))
        with pytest.raises(ValueError, match=r"missing vertical dim"):
            expand_to_channels(ds)
