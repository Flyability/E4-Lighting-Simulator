"""Drive the panel system headlessly: load / new project, add panels from templates and a single LED,
select / mirror / move / remove a panel, save + reload the layout and a panel template.

Usage: PYTHONPATH=. python scripts/ui_panels_smoke.py
"""
import os
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

from lighting_simulator.ui.layout_state import LayoutState  # noqa: E402

_states = []
_orig_state_init = LayoutState.__init__


def _capturing_state_init(self, *a, **k):
    _orig_state_init(self, *a, **k)
    _states.append(self)


LayoutState.__init__ = _capturing_state_init

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


def find_all(label, kind=None):
    return [h for h in handles.values() if getattr(h, "label", None) == label
            and (kind is None or isinstance(h, kind))]


def find(label, kind=None, idx=0):
    out = find_all(label, kind)
    assert out, f"no GUI handle labelled {label!r}"
    return out[idx]


def click(label, idx=0):
    btn = find(label, viser.GuiButtonHandle, idx)
    for cb in btn._impl.update_cb:
        cb(viser.GuiEvent(client=None, client_id=None, target=btn))


def n_groups():
    # one "Select" button per panel in the "Panels in the scene" list
    return len(find_all("Select", viser.GuiButtonHandle))


def n_leds():
    return len(app_state.leds())


app_state = _states[0]  # the LayoutState created by main()

# 1. load a layout with two panels, then start a new project
find("Select Configuration").value = "ludos_panels_n4"
click("📂 Load Configuration")
time.sleep(1)
print(f"[smoke] loaded ludos_panels_n4: {n_groups()} panels, {n_leds()} LEDs")
assert n_groups() == 2
click("🆕 New Project (Empty)")
time.sleep(0.5)
print(f"[smoke] new project: {n_groups()} panels, {n_leds()} LEDs")
assert n_groups() == 0 and n_leds() == 0

# 2. panel from template (single + multi-panel template) + a single LED
templates = [t for t in find("Template").options if t != "(none)"]
print("[smoke] templates:", templates[:5], "...")
find("Template").value = "oris"
click("➕ Add panel from template")
find("Template").value = "elios3"
click("➕ Add panel from template")
click("➕ Add single LED (1-LED panel)")
time.sleep(0.5)
print(f"[smoke] after adds: {n_groups()} panels, {n_leds()} LEDs")
assert n_groups() == 6 and n_leds() == 12 + 48 + 1

# 3. select a panel, mirror it, move it, remove one
click("Select", 0)
time.sleep(0.3)
find("Mirror to the other side (left/right twin)").value = True
time.sleep(0.3)
assert n_leds() == 12 * 2 + 48 + 1, n_leds()
find("Position (cm)").value = (10.0, -5.0, 2.0)
time.sleep(0.3)
assert tuple(app_state.panels[0].position) == (10.0, -5.0, 2.0)
click("Remove", 5)  # the 1-LED panel
time.sleep(0.3)
print(f"[smoke] after mirror/move/remove: {n_groups()} panels, {n_leds()} LEDs")
assert n_groups() == 5 and n_leds() == 12 * 2 + 48

# 4. save layout + panel template, reload layout
import atexit
atexit.register(lambda: [os.path.exists(p) and os.remove(p) for p in
                         ("layouts/smoke_panels_tmp.json", "templates/smoke_tpl_tmp.json")])
find("Project Name").value = "smoke_panels_tmp"
find("Save As").value = "Layout"
click("💾 Save Project")
click("Select", 0)
time.sleep(0.3)
find("Project Name").value = "smoke_tpl_tmp"
find("Save As").value = "Panel template (selected panel)"
click("💾 Save Project")
time.sleep(0.3)
assert os.path.exists("layouts/smoke_panels_tmp.json"), "layout not saved"
assert os.path.exists("templates/smoke_tpl_tmp.json"), "template not saved"
before = (n_groups(), n_leds())
click("🆕 New Project (Empty)")
find("Select Configuration").options = find("Select Configuration").options + ("smoke_panels_tmp",) \
    if "smoke_panels_tmp" not in find("Select Configuration").options else find("Select Configuration").options
find("Select Configuration").value = "smoke_panels_tmp"
click("📂 Load Configuration")
time.sleep(1)
after = (n_groups(), n_leds())
print(f"[smoke] save/reload round trip: {before} -> {after}")
assert before == after, "round trip changed the scene"
import json as _json
_lay = _json.load(open("layouts/smoke_panels_tmp.json"))
assert _lay["schema_version"] == 2 and len(_lay["panels"]) == 5 and _lay["panels"][0]["mirror"] is True
_tpl = _json.load(open("templates/smoke_tpl_tmp.json"))
assert len(_tpl["panels"]) == 1 and len(_tpl["panels"][0]["leds"]) == 12 and _tpl["panels"][0]["mirror"] is False

# 5. wall map still works with this scene
find("Show intensity on wall").value = True
find("Rays per pixel (↑quality, ↓speed)").value = 20
click("Update Intensity Map")

# 6. Panel Designer: new panel, edit (LEDs replaced), cancel (untouched), duplicate, LED toggle
click("🆕 New Project (Empty)")
time.sleep(0.3)
click("✏️ New panel in the Designer")
time.sleep(0.5)
find("Panel name").value = "smoke designer"
click("Add LED"); click("Add LED"); click("Add LED")
click("Save")
time.sleep(0.5)
print(f"[smoke] designer new: {n_groups()} panels, {n_leds()} LEDs, names={[p.name for p in app_state.panels]}")
assert n_groups() == 1 and n_leds() == 3
click("Select", 0); time.sleep(0.3)
click("✏️ Edit LEDs in the Designer"); time.sleep(0.5)
click("Add LED"); click("Save"); time.sleep(0.5)
assert n_groups() == 1 and n_leds() == 4
click("Select", 0); time.sleep(0.3)
click("✏️ Edit LEDs in the Designer"); time.sleep(0.5)
click("Add LED"); click("Cancel"); time.sleep(0.5)
assert n_leds() == 4, "Cancel must leave the panel untouched"
click("Select", 0); time.sleep(0.3)
click("Duplicate panel"); time.sleep(0.3)
assert n_groups() == 2 and n_leds() == 8
click("Select", 0); time.sleep(0.3)
click("L1"); time.sleep(0.3)
assert app_state.panels[0].leds[0].on is False
print(f"[smoke] designer edit/cancel/duplicate/toggle OK: {n_groups()} panels, {n_leds()} LEDs")

print("[smoke] errors:", errors or "none")
