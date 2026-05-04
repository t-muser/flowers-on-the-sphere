"""Compatibility entry point for the Held-Suarez sweep generator."""

from datagen.mitgcm.held_suarez.scripts.generate_sweep import *  # noqa: F401,F403
from datagen.mitgcm.held_suarez.scripts.generate_sweep import main


if __name__ == "__main__":
    main()

