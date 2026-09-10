"""Lighting simulator package.

Layout
------
- ``domain``      LEDs, placement, optics, geometry, panel guides, mirroring
- ``camera``      main-camera / VIO fisheye FOV geometry
- ``raytracing``  CPU (boxes, STL mesh) and GPU intersection kernels
- ``simulation``  Monte Carlo illuminance engine (wall & room modes)
- ``scene``       saved-config → LEDs / absorbers / STL occluder
- ``analysis``    uniformity metrics
- ``pipeline``    headless config → grid → metrics facade
- ``ui``          Viser interactive application (``main``)

The GUI is imported lazily so the headless modules never pull in Viser.
"""

__all__ = ["main"]


def main():
    from .ui.app import main as _main
    return _main()
