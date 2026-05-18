"""Unit tests for the SIZE.h generator.

The generator writes the MITgcm Fortran header that fixes the compile-time
grid + MPI layout. These tests verify the rendered text contains the
expected PARAMETER values for several grids and rejects invalid layouts.

Run::

    uv run --project datagen pytest datagen/mitgcm/held_suarez/tests/test_size_h.py -v
"""

from __future__ import annotations

import re

import pytest

from datagen.mitgcm.held_suarez._size_h import render_size_h, write_size_h


def _get_param(text: str, name: str) -> int:
    """Pull an integer PARAMETER value out of the generated header."""
    match = re.search(rf"{name}\s*=\s*([0-9]+)", text)
    assert match is not None, f"{name} not found in:\n{text}"
    return int(match.group(1))


class TestRenderSizeH:
    def test_legacy_layout(self):
        text = render_size_h(Nlon=128, Nlat=64, Nr=20, n_mpi=4)
        assert _get_param(text, "sNx") == 128
        assert _get_param(text, "sNy") == 16
        assert _get_param(text, "nPy") == 4
        assert _get_param(text, "Nr")  == 20
        assert _get_param(text, "nSx") == 1
        assert _get_param(text, "nSy") == 1
        assert _get_param(text, "nPx") == 1

    def test_64_class_layout(self):
        text = render_size_h(Nlon=256, Nlat=128, Nr=30, n_mpi=8)
        assert _get_param(text, "sNx") == 256
        assert _get_param(text, "sNy") == 16
        assert _get_param(text, "nPy") == 8
        assert _get_param(text, "Nr")  == 30

    def test_alternate_npy(self):
        """Same Nlat, different MPI layout — sNy should adapt."""
        text = render_size_h(Nlon=256, Nlat=128, Nr=30, n_mpi=4)
        assert _get_param(text, "nPy") == 4
        assert _get_param(text, "sNy") == 32

    def test_halo_width(self):
        text = render_size_h(Nlon=128, Nlat=64, Nr=20, n_mpi=4)
        assert _get_param(text, "OLx") == 3
        assert _get_param(text, "OLy") == 3

    def test_includes_derived_extents(self):
        """Nx/Ny should be expressed in terms of the primitive PARAMETERs."""
        text = render_size_h(Nlon=128, Nlat=64, Nr=20, n_mpi=4)
        assert "Nx  = sNx*nSx*nPx" in text
        assert "Ny  = sNy*nSy*nPy" in text

    def test_indivisible_npy_raises(self):
        with pytest.raises(ValueError):
            render_size_h(Nlon=256, Nlat=128, Nr=30, n_mpi=7)

    def test_write_round_trip(self, tmp_path):
        out = tmp_path / "SIZE.h"
        write_size_h(out, Nlon=256, Nlat=128, Nr=30, n_mpi=8)
        text = out.read_text()
        assert _get_param(text, "sNx") == 256
        assert _get_param(text, "nPy") == 8
        assert _get_param(text, "Nr")  == 30
