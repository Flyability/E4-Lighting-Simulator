"""Cross-check UI legend uniformity vs the headless optimiser objective.

Usage: PYTHONPATH=. python scripts/ui_vs_headless.py CONFIG_NAME [CONFIG_NAME ...]
(config names as listed in configs/, without .json)
"""
import re
import sys
import threading
import time
import traceback
import webbrowser

webbrowser.open = lambda *a, **k: True

import viser  # noqa: E402

_servers = []
_orig_init = viser.ViserServer.__init__


def _capturing_init(self, *a, **k):
    _orig_init(self, *a, **k)
    _servers.append(self)


viser.ViserServer.__init__ = _capturing_init

import lighting_simulator.ui.app as app  # noqa: E402
from lighting_simulator.optimization import CameraSpec, ConstraintSpec, ObjectiveSpec, Problem  # noqa: E402
from lighting_simulator.scene import load_config  # noqa: E402
from lighting_simulator.simulation import EmissionSettings, WallSettings  # noqa: E402

WALL_DIST, GRID, WALL_SIZE, RPP = 50, 30, 250, 10500
CAM = dict(pos_x=10, pos_y=-4, pitch=0, fov_h=75, fov_v=60)
LUMENS = 168

errors = []


def _run():
    try:
        app.main()
    except BaseException:  # noqa: BLE001
        errors.append(traceback.format_exc())


threading.Thread(target=_run, daemon=True).start()
deadline = time.time() + 60
while not _servers and time.time() < deadline:
    time.sleep(0.2)
time.sleep(8)
server = _servers[0]
handles = server.gui._gui_input_handle_from_uuid


def find(label, kind=None):
    out = [h for h in handles.values() if getattr(h, "label", None) == label
           and (kind is None or isinstance(h, kind))]
    assert out, f"no GUI handle labelled {label!r}"
    return out[0]


def click(label):
    btn = find(label, viser.GuiButtonHandle)
    for cb in btn._impl.update_cb:
        cb(viser.GuiEvent(client=None, client_id=None, target=btn))


def _walk(container):
    for h in list(getattr(container, "_children", {}).values()):
        yield h
        yield from _walk(h)


def legend_html():
    for root in list(server.gui._container_handle_from_uuid.values()):
        for h in _walk(root):
            c = getattr(h, "content", None)
            if isinstance(c, str) and "Pattern Uniformity" in c:
                return c
    return ""


# Configure UI to match the optimiser objective
find("Wall distance (cm)").value = WALL_DIST
find("Wall grid resolution").value = GRID
find("Wall view size (cm)").value = WALL_SIZE
find("Rays per pixel (↑quality, ↓speed)").value = RPP
find("LED lumens (lm/LED)").value = LUMENS
find("Camera X pos (cm)").value = CAM["pos_x"]
find("Camera Y pos (cm)").value = CAM["pos_y"]
find("Camera pitch (°)").value = CAM["pitch"]
find("Horizontal FOV (°)").value = CAM["fov_h"]
find("Vertical FOV (°)").value = CAM["fov_v"]
find("Show intensity on wall").value = True

problem = Problem({}, [], WallSettings(wall_dist=WALL_DIST, grid_size=GRID, wall_size=WALL_SIZE, rays_per_pixel=RPP),
                  CameraSpec(**CAM), EmissionSettings(default_lumens=LUMENS),
                  ObjectiveSpec(min_percentile=0.0), ConstraintSpec())

for name in sys.argv[1:]:
    find("Select Configuration").value = name
    click("📂 Load Configuration")
    time.sleep(2)
    click("Update Intensity Map")
    time.sleep(1)
    html = legend_html()
    m = re.search(r"font-weight:700;color:#\w+;margin:2px 0 6px;'>([\d.]+)%", html)
    ui_u = m.group(1) if m else "?"
    rows = dict(re.findall(r"<td[^>]*>(E<sub>\w+</sub>|U<sub>\d</sub>[^<]*)</td><td>([^<]+)</td>", html))
    hl = problem.evaluate_config(load_config(f"tests/data/v1_configs/{name}.json"))
    print(f"\n=== {name} ===")
    print(f"  UI legend : U0={ui_u}%  {rows}")
    print(f"  headless  : {hl.summary()}")
