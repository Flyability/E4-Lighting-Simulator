"""Convert legacy custom_groups_templates/*.json (v1: groups + individual_leds) into templates/*.json
(schema-v2 layouts whose panels carry panel-local LEDs).

Usage: python scripts/convert_templates_v2.py [--src custom_groups_templates] [--dst templates]

* every v1 group becomes one panel (via ``convert_v1``);
* all ``individual_leds`` of a template are merged into ONE panel named after the template
  (their coordinates are already panel-local: templates were placed at a slot pose).
"""
import argparse
import json
from pathlib import Path

from lighting_simulator.scene.layout import Layout, Panel, convert_v1, save_json


def convert_template(cfg, name):
    groups = cfg.get("groups", [])
    indiv = cfg.get("individual_leds", [])
    lay = convert_v1({"custom_groups": groups, "individual_leds": [], "name": name})
    panels = list(lay.panels)
    if indiv:
        one = convert_v1({"custom_groups": [], "individual_leds": indiv, "name": name})
        leds = [p.leds[0] for p in one.panels]  # each individual LED -> a 1-LED panel at world pose
        for p in one.panels:
            # the 1-LED panel pose IS the LED pose: bake it into the LedSpec
            pos, normals, rows, beams = p.world_geometry()
            spec = p.leds[0]
            spec.position = tuple(float(v) for v in pos[0])
            spec.direction = tuple(float(v) for v in normals[0])
            spec.row_dir = tuple(float(v) for v in rows[0])
        panels.append(Panel(name=name, leds=leds, rows=[list(range(len(leds)))]))
    for i, p in enumerate(panels):
        p.position, p.rotation, p.mirror, p.source = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, None
        if len(panels) > 1 and not p.name:
            p.name = f"{name} {i + 1}"
        p.name = p.name or name
    return Layout(name=name, description=f"Panel template converted from custom_groups_templates/{name}.json",
                  panels=panels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="custom_groups_templates")
    ap.add_argument("--dst", default="templates")
    args = ap.parse_args()
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(Path(args.src).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        lay = convert_template(cfg, cfg.get("name") or path.stem)
        if not lay.panels:
            print(f"{path.name:28s} -> skipped (empty)")
            continue
        out = Path(args.dst) / f"{path.stem}.json"
        save_json(lay, out)
        n += 1
        print(f"{path.name:28s} -> {out}  panels={len(lay.panels)} leds={sum(len(p.leds) for p in lay.panels)}")
    print(f"\n{n} templates")


if __name__ == "__main__":
    main()
