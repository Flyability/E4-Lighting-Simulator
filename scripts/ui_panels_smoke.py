"""Drive the config / panel system headlessly: new project, add groups (empty, template, individual),
panel slots (solid + individual), save/load project and template, mirror.

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
    # group folders are hidden but exist; count "Remove Group" buttons instead
    return len(find_all("Remove Group", viser.GuiButtonHandle))


def n_leds():
    return len(find_all("Remove LED", viser.GuiButtonHandle))


# 1. load a config with custom groups, then start a new project
find("Select Configuration").value = "ludos_panels_n4"
click("📂 Load Configuration")
time.sleep(1)
print(f"[smoke] loaded ludos_panels_n4: {n_groups()} groups")
assert n_groups() == 2
click("🆕 New Project (Empty)")
time.sleep(0.5)
print(f"[smoke] new project: {n_groups()} groups, {n_leds()} LEDs")
assert n_groups() == 0

# 2. empty custom group + template as group + template as individual LEDs
find("From Template").value = "Empty"
click("➕ Add Custom Group")
templates = [t for t in find("From Template").options if t != "Empty"]
print("[smoke] templates:", templates[:5], "...")
find("From Template").value = "oris"
find("Load Mode").value = "As Group (Solid)"
click("➕ Add Custom Group")
find("From Template").value = "oris"
find("Load Mode").value = "As Individual LEDs (Editable)"
click("➕ Add Custom Group")
time.sleep(0.5)
print(f"[smoke] after adds: {n_groups()} groups, {n_leds()} individual LEDs")
assert n_groups() >= 2 and n_leds() > 0

# 3. panel slots: solid into slot 0, individual into slot 1, then clear slot 0
slot_dd = find_all("Template")
slot_mode = find_all("Mode")
slot_dd[0].value = "oris"
slot_mode[0].value = "Solid (Group)"
click("✅ Load Panel", 0)
slot_dd[1].value = "oris"
slot_mode[1].value = "Individual LEDs"
click("✅ Load Panel", 1)
time.sleep(0.5)
g_after_slots, l_after_slots = n_groups(), n_leds()
print(f"[smoke] after slots: {g_after_slots} groups, {l_after_slots} LEDs")
click("🗑️ Remove Panel", 0)
time.sleep(0.3)
print(f"[smoke] after clearing slot 0: {n_groups()} groups")
assert n_groups() < g_after_slots

# 4. save project + template, reload project
import atexit
atexit.register(lambda: [os.path.exists(p) and os.remove(p) for p in
                         ("layouts/smoke_panels_tmp.json", "custom_groups_templates/smoke_tpl_tmp.json")])
find("Project Name").value = "smoke_panels_tmp"
find("Save As").value = "Full Configuration"
click("💾 Save Project")
find("Project Name").value = "smoke_tpl_tmp"
find("Save As").value = "Custom Group Template"
click("💾 Save Project")
time.sleep(0.3)
assert os.path.exists("layouts/smoke_panels_tmp.json"), "project not saved"
assert os.path.exists("custom_groups_templates/smoke_tpl_tmp.json"), "template not saved"
before = (n_groups(), n_leds())
click("🆕 New Project (Empty)")
find("Select Configuration").options = find("Select Configuration").options + ("smoke_panels_tmp",) \
    if "smoke_panels_tmp" not in find("Select Configuration").options else find("Select Configuration").options
find("Select Configuration").value = "smoke_panels_tmp"
click("📂 Load Configuration")
time.sleep(1)
after = (n_groups(), n_leds())
print(f"[smoke] save/reload round trip: {before} -> {after}")
# Schema v2 has no individual LEDs: template-sourced ones come back as one panel, standalone
# ones as one-LED panels, so every LED is still there but all of them live in groups now.
import json as _json
_lay = _json.load(open("layouts/smoke_panels_tmp.json"))
assert _lay["schema_version"] == 2 and after[1] == 0, "round trip changed the scene unexpectedly"
assert sum(len(p["leds"]) for p in _lay["panels"]) == 12 * before[0] + before[1], "LED count changed on save"
assert after[0] == len(_lay["panels"]), "not every panel came back"

# 5. wall map still works with this scene
find("Show intensity on wall").value = True
find("Rays per pixel (↑quality, ↓speed)").value = 20
click("Update Intensity Map")

print("[smoke] errors:", errors or "none")
