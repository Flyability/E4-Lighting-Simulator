"""Convert schema-v1 configs/*.json into layouts/*.json + platforms/*.json (schema v2).

Usage: python scripts/convert_configs_v2.py [--src configs] [--dst layouts] [--platforms platforms]

Every distinct frame-model / VIO-camera setup becomes one platform file named after the
STL/STEP file; layouts link to it by name. Layouts with neither an STL nor custom VIO
poses get ``platform: null``.
"""
import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from lighting_simulator.scene.layout import Platform, convert_v1, save_json


def _slug(s):
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "platform"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="configs")
    ap.add_argument("--dst", default="layouts")
    ap.add_argument("--platforms", default="platforms")
    ap.add_argument("--lumens", type=float, default=168.0, help="flux.vio_lumens for converted files")
    args = ap.parse_args()

    platforms = {}  # signature -> name
    n_lay = 0
    for path in sorted(Path(args.src).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        lay = convert_v1(cfg, default_lumens=args.lumens)
        lay.name = lay.name or path.stem
        if not lay.panels:
            print(f"{path.name:36s} -> skipped (no panels left after dropping the legacy base ring)")
            continue
        plat = lay.platform if isinstance(lay.platform, Platform) else None
        if plat is not None and plat.stl is None and asdict(plat.vio_cameras) == asdict(Platform().vio_cameras):
            plat = None  # nothing platform-specific
        if plat is not None:
            sig = json.dumps({k: v for k, v in asdict(plat).items() if k != 'name'}, sort_keys=True)
            if sig not in platforms:
                base = _slug(re.split(r"[\\/]", plat.stl.file)[-1].rsplit(".", 1)[0]) if plat.stl else "no_frame"
                name, k = base, 2
                while name in platforms.values():
                    name, k = f"{base}_{k}", k + 1
                plat.name = name
                platforms[sig] = name
                save_json(plat, Path(args.platforms) / f"{name}.json")
                print(f"platform {name}: stl={plat.stl.file if plat.stl else '-'}")
            lay.platform = platforms[sig]
        else:
            lay.platform = None
        out = Path(args.dst) / f"{path.stem}.json"
        save_json(lay, out)
        n_lay += 1
        print(f"{path.name:36s} -> {out}  panels={len(lay.panels)} mirrored={sum(p.mirror for p in lay.panels)} "
              f"platform={lay.platform}")
    print(f"\n{n_lay} layouts, {len(platforms)} platforms")


if __name__ == "__main__":
    main()
