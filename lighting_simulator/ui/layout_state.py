"""In-memory scene model of the UI: one :class:`Layout` plus selection / guide state.

Every simulation and every widget reads from here (``leds()``) instead of rebuilding LED
lists from scattered GUI handles. Widgets edit ``layout.panels[i]`` directly and call
``notify()``; the scene view and the panel list subscribe with ``on_change``.
"""

from __future__ import annotations

import copy

import numpy as np

from lighting_simulator.domain.geometry import euler_xyz_from_matrix, euler_xyz_matrix, rodrigues_rotation
from lighting_simulator.domain.guides import fit_panel_cylinder_from_rows
from lighting_simulator.scene.layout import Layout, LedSpec, Panel, build_leds_from_layout, layout_to_v1


class LayoutState:
    def __init__(self):
        self.layout = Layout(name="untitled")
        self.platform_name = None
        """Platform file the layout links to (None = frame / VIO settings saved inline)."""
        self.selected = None
        """Index of the selected panel, or None."""
        self.guides = {}
        """panel index -> circular-guide dict (runtime only, see :meth:`fit_guide`)."""
        self.loading = False
        """Suppress change notifications while a whole layout is being applied."""
        self._listeners = []

    # -- observers ---------------------------------------------------------
    def on_change(self, callback):
        self._listeners.append(callback)

    def notify(self, what="layout"):
        if self.loading:
            return
        for cb in list(self._listeners):
            cb(what)

    # -- whole layout --------------------------------------------------------
    def set_layout(self, layout: Layout, platform_name=None):
        self.layout = layout
        self.platform_name = platform_name if platform_name is not None else (
            layout.platform if isinstance(layout.platform, str) else None)
        self.selected = None
        self.guides = {}
        self.notify("structure")

    def clear(self):
        self.set_layout(Layout(name="untitled"))

    def leds(self, lumens=None):
        """Engine placements of the whole layout (mirror twins included); owner = ('panel', i)."""
        return build_leds_from_layout(self.layout, lumens=lumens)

    def v1_config(self):
        """v1 runtime dict for the optimiser and legacy exporters (group i == panel i)."""
        return layout_to_v1(self.layout)

    # -- panels --------------------------------------------------------------
    @property
    def panels(self):
        return self.layout.panels

    def panel(self, index) -> Panel | None:
        if index is None or not (0 <= index < len(self.panels)):
            return None
        return self.panels[index]

    def selected_panel(self) -> Panel | None:
        return self.panel(self.selected)

    def owner_index(self, owner):
        """Panel index behind an LED placement ``owner`` tuple (``None`` for mirror twins)."""
        if isinstance(owner, tuple) and len(owner) == 2 and owner[0] == 'panel':
            return int(owner[1])
        return None

    def unique_name(self, base):
        names = {p.name for p in self.panels}
        if base not in names:
            return base
        k = 2
        while f"{base}_{k}" in names:
            k += 1
        return f"{base}_{k}"

    def add_panel(self, panel: Panel, select=True):
        panel.name = self.unique_name(panel.name or "panel")
        self.panels.append(panel)
        idx = len(self.panels) - 1
        if select:
            self.selected = idx
        self.notify("structure")
        return idx

    def add_panels(self, panels, select_last=True):
        for p in panels:
            p.name = self.unique_name(p.name or "panel")
            self.panels.append(p)
        if panels and select_last:
            self.selected = len(self.panels) - 1
        self.notify("structure")

    def remove_panel(self, index):
        if self.panel(index) is None:
            return
        self.panels.pop(index)
        self.guides = {(i - 1 if i > index else i): g for i, g in self.guides.items() if i != index}
        if self.selected is not None:
            self.selected = None if self.selected == index else (self.selected - 1 if self.selected > index else self.selected)
        self.notify("structure")

    def duplicate_panel(self, index):
        src = self.panel(index)
        if src is None:
            return None
        dup = copy.deepcopy(src)
        dup.name = self.unique_name(src.name)
        return self.add_panel(dup)

    def select(self, index):
        self.selected = index if self.panel(index) is not None else None
        self.notify("selection")

    def new_single_led_panel(self, name="led"):
        return Panel(name=name, leds=[LedSpec(position=(0, 0, 0), direction=(1, 0, 0), row_dir=(0, 1, 0), size=0.5)],
                     position=(0.0, 0.0, 0.0))

    # -- circular guides -----------------------------------------------------
    def fit_guide(self, index):
        """Fit a cylinder through the panel's LED rows (world frame) and lock the pose to it.

        Returns an error string or None. The panel's current pose becomes the θ = 0 reference.
        """
        panel = self.panel(index)
        if panel is None:
            return "no panel"
        rows = panel.rows or [[i for i in range(len(panel.leds))]]
        pos, _, _, _ = panel.world_geometry()
        guide, err = fit_panel_cylinder_from_rows(pos, rows)
        if err:
            return err
        guide['base_position'] = tuple(panel.position)
        guide['base_rotation'] = tuple(panel.rotation)
        guide['theta_deg'] = 0.0
        self.guides[index] = guide
        self.notify("guide")
        return None

    def set_guide_theta(self, index, theta_deg):
        """Orbit the panel about the guide axis: a pure change of the panel transform."""
        guide, panel = self.guides.get(index), self.panel(index)
        if guide is None or panel is None:
            return
        R_g = rodrigues_rotation(np.asarray(guide['axis'], float), np.radians(float(theta_deg)))
        O = np.asarray(guide['origin'], float)
        R_base = euler_xyz_matrix(*guide['base_rotation'])
        p_base = np.asarray(guide['base_position'], float)
        panel.rotation = tuple(euler_xyz_from_matrix(R_g @ R_base))
        panel.position = tuple(float(v) for v in O + R_g @ (p_base - O))
        guide['theta_deg'] = float(theta_deg)
        self.notify("layout")

    def release_guide(self, index):
        """Keep the current pose, forget the guide."""
        if self.guides.pop(index, None) is not None:
            self.notify("guide")
