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
click("🔄 Refresh specs & groups")
groups = find("Group").options
print("[smoke] groups:", groups)
find("Group").value = groups[0]
find("Move / rotate group").value = True
find("LED on / off").value = True
find("Max evaluations").value = 40
find("Population").value = 8
find("Rays per pixel").value = 50
find("Method").value = "random_search"
find("Use GPU").value = True
find("Max active LEDs (0 = no limit)").value = 6
find("Load best into scene when finished").value = True
find("Show intensity on wall").value = True
find("Rays per pixel (↑quality, ↓speed)").value = 20

t0 = time.time()
click("▶ Run optimization")
while find("▶ Run optimization").disabled and time.time() - t0 < 180:
    time.sleep(0.5)
print(f"[smoke] optimisation finished in {time.time() - t0:.1f}s")
print("[smoke] Project Name field:", find("Project Name").value)

# preset spec path
find("Preset spec").value = "ludo_refine"
fire_update(find("Preset spec"))
print("[smoke] after preset: metric=", find("Metric").value, "evals=", find("Max evaluations").value,
      "vars from=", find("Variables from").value)
find("Max evaluations").value = 20
find("Start from current scene").value = True
find("Use UI wall / camera / emission").value = True
t0 = time.time()
click("▶ Run optimization")
while find("▶ Run optimization").disabled and time.time() - t0 < 180:
    time.sleep(0.5)
print(f"[smoke] preset optimisation finished in {time.time() - t0:.1f}s")
print("[smoke] errors:", errors or "none")
