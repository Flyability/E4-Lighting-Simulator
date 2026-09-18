"""Drive the Viser UI headlessly: load a config, run wall + room intensity maps.

Usage: PYTHONPATH=. python scripts/ui_smoke.py
"""
import threading
import time
import traceback
import webbrowser

webbrowser.open = lambda *a, **k: True  # never pop a browser during the smoke run

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
    except BaseException as e:  # noqa: BLE001
        errors.append(traceback.format_exc())


threading.Thread(target=_run, daemon=True).start()
deadline = time.time() + 60
while not _servers and time.time() < deadline:
    time.sleep(0.2)
time.sleep(8)  # let main() finish building the GUI
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


print("[smoke] GUI handles:", len(handles))
print("[smoke] configs listed:", find("Select Configuration").options)
find("Select Configuration").value = "elios3_oris"
click("📂 Load Configuration")
time.sleep(3)
find("Show intensity on wall").value = True
find("Rays per pixel (↑quality, ↓speed)").value = 20
t0 = time.time()
click("Update Intensity Map")
print(f"[smoke] wall intensity map done in {time.time() - t0:.1f}s")
find("Enable Room Mode").value = True
find("Show Room Intensity").value = True
find("Enable Reflections").value = True
find("Room walls grid resolution").value = 10
time.sleep(1)
t0 = time.time()
click("Update Room Intensity")
print(f"[smoke] room intensity map done in {time.time() - t0:.1f}s")
print("[smoke] errors:", errors or "none")
