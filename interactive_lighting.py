"""Compatibility wrapper for the Viser lighting simulator.

The application logic now lives under the lighting_app package so the project has
an easier structure to extend without breaking the historical entrypoint.
"""

from lighting_app.legacy_interactive import main


if __name__ == "__main__":
    main()
