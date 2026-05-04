"""Compatibility entry point for global-ocean corner preflight."""

from datagen.mitgcm.global_ocean.scripts.preflight import *  # noqa: F401,F403
from datagen.mitgcm.global_ocean.scripts.preflight import main


if __name__ == "__main__":
    raise SystemExit(main())

