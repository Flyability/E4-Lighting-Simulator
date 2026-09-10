"""Compatibility wrapper for the Viser lighting simulator.

The application now lives in the ``lighting_simulator`` package
(``lighting_simulator.ui.app``); this keeps the historical entrypoint working.
"""

from lighting_simulator.ui.app import main


if __name__ == "__main__":
    main()
