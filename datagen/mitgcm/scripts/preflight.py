"""Compatibility entry point for Held-Suarez corner preflight."""

from datagen.mitgcm.held_suarez.scripts.preflight import *  # noqa: F401,F403
from datagen.mitgcm.held_suarez.scripts.preflight import main


if __name__ == "__main__":
    raise SystemExit(main())

