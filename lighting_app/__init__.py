"""Lighting simulator package.

The legacy interactive UI still lives in lighting_app.legacy_interactive for a
cleaner project layout while keeping compatibility with the existing
interactive_lighting.py entrypoint.
"""

from .legacy_interactive import main

__all__ = ["main"]
