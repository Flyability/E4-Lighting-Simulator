"""Drive the Optimize tab headlessly: load a config, run a short GPU optimisation, load the result.

Usage: PYTHONPATH=. python scripts/ui_optim_smoke.py
"""
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


def fire_update(handle):
    for cb in handle._impl.update_cb:
        cb(viser.GuiEvent(client=None, client_id=None, target=handle))


find("Select Configuration").value = "ludos_panels_n4"
click("📂 Load Configuration")
time.sleep(3)

MODE_REFINE = "1 · Refine the current panels"
MODE_DUCTS = "2 · Design LEDs on the ducts (from scratch)"
MODE_PRESET = "3 · Run a preset spec file"


def run_and_wait(label):
    t0 = time.time()
    click("▶ Run optimization")
    while find("▶ Run optimization").disabled and time.time() - t0 < 180:
        time.sleep(0.5)
    print(f"[smoke] {label} finished in {time.time() - t0:.1f}s")


# --- mode 1: refine -------------------------------------------------------
find("Design mode").value = MODE_REFINE
fire_update(find("Design mode"))
click("🔄 Refresh group list")
groups = find("Group").options
print("[smoke] groups:", groups)
find("Group").value = groups[0]
find("Move / rotate the panel").value = True
find("LED on / off").value = True
find("Drive current (→ lumens)").value = True
find("Require flash lux target").value = True
find("Require VIO FOV coverage").value = True
find("Wall distances (cm)").value = "50, 150"
find("Wall size").value = "Auto: fit camera FOV at each distance"
find("Min beam angle off camera axis (°, 0 = off)").value = 30
find("Symmetry penalty weight (0 = off)").value = 0.5
find("Max evaluations").value = 40
find("Population").value = 8
find("Rays per pixel").value = 50
find("Method").value = "random_search"
find("Use GPU").value = True
find("Max active LEDs (0 = no limit)").value = 6
find("Load best into scene when finished").value = True
find("Show intensity on wall").value = True
find("Rays per pixel (↑quality, ↓speed)").value = 20
run_and_wait("refine")
print("[smoke] Project Name field:", find("Project Name").value)

# --- mode 3: preset as written, then with overrides -------------------------
find("Design mode").value = MODE_PRESET
fire_update(find("Design mode"))
find("Spec file").value = "ludo_refine"
fire_update(find("Spec file"))
find("Start from the current scene").value = True
find("Use the Evaluation folder (walls, camera, emission)").value = True
fire_update(find("Use the Evaluation folder (walls, camera, emission)"))
find("Max evaluations").value = 20
run_and_wait("preset (overrides)")

# --- copy preset into controls -> switches to the matching mode -------------
find("Spec file").value = "elios4_ducts_flash_vio"
fire_update(find("Spec file"))
click("📋 Copy spec into the controls & switch mode")
print("[smoke] after copy: mode=", find("Design mode").value, "radius=", find("Duct radius (cm)").value,
      "tol arc=", find("± around the duct (cm along circumference)").value, "rows=", find("Max rows (along axis)").value)
assert find("Design mode").value == MODE_DUCTS

# --- mode 2: ducts -----------------------------------------------------------
find("± duct centre shift (cm)").value = (2.0, 2.0, 0.0)
find("Max evaluations").value = 30
find("Method").value = "random_search"
find("Rays per pixel").value = 50
run_and_wait("ducts")

# tilted ducts + VIO scored on the six room walls
find("Duct rotation X/Y/Z (°)").value = (8.0, -5.0, 0.0)
fire_update(find("Duct rotation X/Y/Z (°)"))
n_duct_nodes = sum(1 for n in server.scene._handle_from_node_name if n.startswith("/optim_ducts/"))
print("[smoke] duct preview nodes after rotation:", n_duct_nodes)
assert n_duct_nodes >= 6
find("Evaluate VIO on").value = "Room (6 walls around the rig)"
fire_update(find("Evaluate VIO on"))
find("Room wall distance (cm)").value = 300
find("Room grid per wall").value = 15
find("Max evaluations").value = 12
run_and_wait("ducts (tilted, room VIO)")
print("[smoke] errors:", errors or "none")
