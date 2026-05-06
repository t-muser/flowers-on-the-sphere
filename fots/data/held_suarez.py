"""Held-Suarez 3-D data module.

The MITgcm Held-Suarez 3-D ensemble stores variables per-quantity:
``u``, ``v``, ``T`` as ``(time, level, lat, lon)`` over 8 ERA5-standard
isobaric levels and ``ps`` as ``(time, lat, lon)``. The shared
``ZarrDataModule`` expects a single consolidated
``fields(time, field, lat, lon)`` array, so this subclass synthesizes
that array on the fly by stacking u/v/T along ``level`` and appending
``ps`` as a singleton field. Channel order is locked to
``stats.json``'s key order: u_50..u_1000, v_50..v_1000, T_50..T_1000, ps.
"""
from __future__ import annotations

import xarray as xr

from fots.data.zarr_dataset import ZarrDataModule, _ZarrWindowDataset


# On-disk level coord, in zarr ordering. Matches stats.json suffixes.
_LEVELS_HPA = (50, 100, 250, 500, 700, 850, 925, 1000)
_VARS_3D = ("u", "v", "T")


def _stack_fields(raw: xr.Dataset) -> xr.DataArray:
    """Stack per-variable HS3D arrays into ``fields(time, field, lat, lon)``.

    Drops the ``level`` and ``pressure_actual_hpa`` coords on rename so
    concat aligns with the surface ``ps`` part, which has neither.
    Ordering is u/v/T over levels in zarr order (50/100/.../1000 hPa)
    then ps — locked to match ``stats.json`` key order.
    """
    parts = []
    for v in _VARS_3D:
        a = (
            raw[v]
            .transpose("time", "level", "lat", "lon")
            .drop_vars(["level", "pressure_actual_hpa"], errors="ignore")
            .rename({"level": "field"})
        )
        parts.append(a)
    ps = raw["ps"].transpose("time", "lat", "lon").expand_dims({"field": 1}, axis=1)
    return xr.concat(parts + [ps], dim="field")


class _HeldSuarezWindowDataset(_ZarrWindowDataset):
    def _open_run(self, run_id: int) -> xr.Dataset:
        ds = self._zarr_cache.get(run_id)
        if ds is None:
            path = self.split_dir / f"run_{run_id:04d}.zarr"
            raw = xr.open_zarr(str(path), consolidated=True)
            ds = xr.Dataset({"fields": _stack_fields(raw)})
            self._zarr_cache[run_id] = ds
        return ds


class HeldSuarezDataModule(ZarrDataModule):
    def _make_window_dataset(self, **kwargs) -> _HeldSuarezWindowDataset:
        return _HeldSuarezWindowDataset(**kwargs)
