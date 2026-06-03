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

  * **Per-level normalization.** For leveled datasets, the stats block carries
    one mean/std *per level* (a leveled scalar is a per-level list; a leveled
    vector a per-level list of per-component). The base normalizer stores those
    as a 1-D/2-D tensor; we reshape them to broadcast over the level spatial
    axis so each level is standardized independently.

Everything else (and hence PlanetSWE via the base class) is untouched.

NOTE: ``WellDataModule.denormalize_fn`` assumes (B, C, H, W) and per-channel
scalar stats; it does not yet handle the 3-D / per-level case. That path is
only exercised once a 3-D model trains through the Trainer -- update it then.
"""
from __future__ import annotations

import h5py

from fots.data.well import WellDataset


class SphericalHDF5Dataset(WellDataset):
    def __init__(self, *args, **kwargs):
        # Closed sphere: there is no boundary group to read or pad.
        kwargs["boundary_return_type"] = None
        super().__init__(*args, **kwargs)
        if self.norm is not None:
            self._reshape_per_level_stats()

    def _reshape_per_level_stats(self):
        """Reshape per-level field stats so they broadcast over the level axis.

        The normalizer applies ``(x - mean) / std`` per field on ``x`` of shape
        ``(n_steps, *spatial[, comp])``. A leveled field's stat arrives as a
        1-D ``(nlev,)`` (scalar) or 2-D ``(nlev, ncomp)`` (vector) tensor, which
        would broadcast against the wrong (trailing) axis. We reshape it to put
        ``nlev`` at the level position among the spatial dims and 1 elsewhere
        (component axis last), so each level normalizes independently."""
        with h5py.File(self.files_paths[0], "r") as f:
            spatial_dims = [str(d) for d in f["dimensions"].attrs["spatial_dims"]]
            if "level" not in spatial_dims:
                return  # 2-D dataset: nothing per-level
            level_pos = spatial_dims.index("level")
            n_spatial = len(spatial_dims)
            for grp in ("t0_fields", "t1_fields", "t2_fields"):
                if grp not in f:
                    continue
                for name in f[grp].attrs.get("field_names", []):
                    name = str(name)
                    dset = f[grp][name]  # (traj, time, *spatial[, comp])
                    nlev = dset.shape[2 + level_pos]  # +traj +time
                    if nlev <= 1:  # surface field (size-1 level placeholder)
                        continue
                    is_vec = grp != "t0_fields"
                    ncomp = dset.shape[-1] if is_vec else 1
                    target = [1] * n_spatial
                    target[level_pos] = nlev
                    if is_vec:
                        target = target + [ncomp]
                    for store in (self.norm.means, self.norm.stds,
                                  self.norm.means_delta, self.norm.stds_delta):
                        if name in store:
                            store[name] = store[name].reshape(*target)
        # Keep the flattened views consistent with the reshaped per-field stats.
        self.norm._precompute_flattened_stats()

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
