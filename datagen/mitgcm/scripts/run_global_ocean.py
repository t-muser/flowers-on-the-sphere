"""Compatibility entry point for global-ocean MITgcm runs."""

from datagen.mitgcm.global_ocean.scripts.run import main


if __name__ == "__main__":
    raise SystemExit(main())

