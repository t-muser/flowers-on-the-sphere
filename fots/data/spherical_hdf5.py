"""Dataset for our spherical-PDE HDF5 spec (written by
``scripts/zarr_to_hdf5.py``).

It reuses all of :class:`~fots.data.well.WellDataset` (windowing,
normalization, lead-time, rollout, delta mode, the sample contract that
``WellDataModule`` and the trainer depend on) and changes only the two places
where our spec deliberately diverges from the Well format:

  * **No boundary conditions.** A closed sphere has no boundary, so our files
    omit the ``boundary_conditions`` group entirely. We skip the BC scan and
    never return BC padding channels.
  * **Explicit vector components.** Vector (t1/t2) fields declare their channel
    names via a ``component_names`` attr, so a genuine k-component tangent
    vector -- e.g. 2-component velocity on a 3-D (level, lat, lon) grid -- is
    counted and labelled correctly instead of being assumed to have
    ``n_spatial_dims ** order`` components.

Everything else (and hence PlanetSWE via the base class) is untouched.
"""
from __future__ import annotations

from fots.data.well import WellDataset


class SphericalHDF5Dataset(WellDataset):
    def __init__(self, *args, **kwargs):
        # Closed sphere: there is no boundary group to read or pad.
        kwargs["boundary_return_type"] = None
        super().__init__(*args, **kwargs)

    def _scan_bc_types(self, _f, spatial_dims, file):
        # No boundary on a sphere; files carry no `boundary_conditions` group.
        return []

    def _field_component_names(self, _f, ti, field, order):
        # Prefer explicit per-component channel labels when present (already the
        # physical variable names, e.g. u_phi / u_theta); fall back to the Well
        # spatial-dim-product convention otherwise.
        comp = _f[ti][field].attrs.get("component_names")
        if comp is not None:
            return [str(c) for c in comp]
        return super()._field_component_names(_f, ti, field, order)
