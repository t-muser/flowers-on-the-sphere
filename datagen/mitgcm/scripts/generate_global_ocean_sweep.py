"""Compatibility entry point for the global-ocean sweep generator."""

from datagen.mitgcm.global_ocean.scripts.generate_sweep import *  # noqa: F401,F403
from datagen.mitgcm.global_ocean.scripts.generate_sweep import main


if __name__ == "__main__":
    main()

