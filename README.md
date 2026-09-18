# E4 Lighting Simulator

Monte Carlo illuminance simulator for LED rigs (Elios 3 style panels), with a
Viser-based interactive designer and a headless API for scripted studies such
as LED-placement optimisation.

## Run

```bash
uv sync                      # or: pip install -e .
python -m lighting_simulator # interactive UI at http://localhost:8080
python interactive_lighting.py   # legacy entrypoint, same thing
python -m pytest tests           # engine regression tests
```

## Package layout (`lighting_simulator/`)

| Package | Responsibility |
|---|---|
| `domain/` | `LED`, `LEDPlacement`, `create_leds` factory, optics (Lambertian/lens), geometry, panel guides, XZ mirroring |
| `camera/` | Main pinhole camera and VD66GY fisheye VIO FOV geometry |
| `raytracing/` | Intersection kernels: box occluders, STL meshes (trimesh BVH), GPU (CuPy/Taichi) |
| `simulation/` | The engine: `compute_wall_intensity`, `compute_room_intensity`, ray emission, room geometry, `WallSettings`/`RoomSettings`/`EmissionSettings`; `sphere.py` = analytic illuminance on a sphere around the drone (Advanced → *Sphere Mode*: equal-distance VIO / main-camera coverage, editable radius) |
| `scene/` | Layout / platform files (`layouts/*.json`, `platforms/*.json`, schema v2 in `scene/layout.py`) → LEDs + STL occluder (`build_scene_from_config`, `build_scene_from_layout`); `convert_v1` upgrades pre-v2 configs |
| `analysis/` | `uniformity_metrics` (U0, U1, CV, ΔEV) and legend HTML |
| `pipeline.py` | Headless facade: `simulate_wall(cfg, settings)` / `simulate_room(...)` → grid + metrics |
| `ui/app.py` | The Viser GUI (`main`). Only `ui/` imports Viser. `ui/layout_state.py` holds the one `Layout` the GUI edits (panels + selection), `ui/panels_tab.py` the *Panels & LEDs* tab and the *Selected* inspector |

Everything below `ui/` is UI-free and importable without Viser.

## Files: layouts and platforms

A **layout** (`layouts/<name>.json`) is the lighting design: `flux` (`vio_lumens` for the
flight mode, `flash_lumens` for the pulse) and a list of `panels`, each with a `position` /
`rotation` (extrinsic X-Y-Z, degrees) and its LEDs in panel-local coordinates (`position`,
`direction` = square normal, `row_dir`, `beam_angle`, `tilt` about `row_dir`, `size`, `on`,
`role` = `vio | flash | both`). `mirror: true` builds the XZ-mirrored (left/right) twin at load
time, so a symmetric rig stores one half. Units cm; +X forward, +Y left, +Z up.

A **platform** (`platforms/<name>.json`) is the drone: the CAD frame (`stl`: file, scale,
pose, `occludes`) and the VIO camera poses. A layout links to one by name (`"platform":
"harmony_e4"`) or embeds it inline. Camera FOV, wall distance, grid and rays are session
settings, not part of either file. Pre-v2 configs are converted on load;
`scripts/convert_configs_v2.py` batch-converts a folder (the old files live in
`tests/data/v1_configs/` as fixtures).

A **template** (`templates/<name>.json`) is a layout whose panels sit at the origin: *Panels &
LEDs → Add panel from template* drops its panel(s) into the scene, *Save As → Panel template*
stores the selected panel with its LEDs in panel coordinates. In the GUI there is exactly one
kind of object — the panel: a single LED is a 1-LED panel, a 12-LED board is a panel, and
*Mirror* on a panel gives its left/right twin (the old "custom groups", "individual LEDs" and
the Elios 3 slot configurator are gone). `scripts/convert_templates_v2.py` converted the legacy
`custom_groups_templates/` (now `tests/data/v1_templates/`).

## Headless example (basis for optimisation loops)

```python
from lighting_simulator.pipeline import simulate_wall
from lighting_simulator.scene import load_config
from lighting_simulator.simulation import WallSettings

cfg = load_config("layouts/Elios3.json")
settings = WallSettings(wall_dist=100, grid_size=50, wall_size=80, rays_per_pixel=4)

def objective(cfg):
    result = simulate_wall(cfg, settings, use_gpu=False)   # result.grid is lux (Z rows, Y cols)
    return -result.metrics.u0                              # maximise Emin/Eavg

# mutate cfg["custom_groups"][i]["position"] / ["rotation_*"] / ["led_positions"] and re-evaluate
```

`scripts/ui_smoke.py` drives the real GUI callbacks headlessly (load config →
wall map → room map) and is a quick end-to-end check after UI edits;
`scripts/ui_panels_smoke.py` covers the panel system (templates, mirror, designer, save/reload) and
`scripts/ui_optim_smoke.py` the Optimize tab. `scripts/benchmark_gpu.py`
and `scripts/test_gpu.py` compare GPU vs CPU tracing; `scripts/beam_calibration.py` is a
standalone beam-profile calibration tool.

## LED placement optimisation

```bash
python -m lighting_simulator.optimization optimization_specs/elios3_duct_rings.json
python -m lighting_simulator.optimization optimization_specs/elios3_refine_panels.json --evals 2000 --workers -1
```

A run is described by a JSON *spec* (see `optimization_specs/`) — constraints are
data, not code:

| Section | What you control |
|---|---|
| `wall`, `camera`, `emission` | Same quantities as the UI (wall distance/grid, main-camera FOV, lumens). Uniformity is measured inside the camera footprint, exactly like the UI legend. |
| `objective` | `metric` (`u0`, `u1`, `cv`), `coverage_weight` (penalise unlit FOV cells), `min_avg_lux` |
| `constraints` | `max_leds`, `led_cost` (fewer lights), `min_led_spacing_cm`, `keep_out` boxes |
| `variables` | list of design-variable groups (below) |
| `optimizer` | `differential_evolution` / `nelder_mead` / `random_search`, `max_evals`, `population`, `workers`, `seed` |

Variable groups (`type`):

- `duct_ring` — LEDs mounted on a cylinder (`duct`: center, axis, radius). `placement: "arc"` keeps an `n_rows × n_cols` lattice and optimises arc centre/span, axial position, row pitch; `"free"` gives every LED its own angle/height. Optional shared or per-LED axial/tangential tilt, `beam_angle_range`, `optimize_enabled` (on/off per LED), `mirror_xz` for left/right symmetry.
- `panel_pose` — ±translation/rotation of an existing custom group of the base config.
- `led_states` — on/off per LED of a group (or the base rig).
- `beam_angle` — shared viewing angle of a group.
- `beam_tilts` — per-LED beam tilt inside a dynamic group.

Output goes to `exports/optim/<name>/` (git-ignored): `best_config.json` (load it in the UI
via *Load Configuration* or save it to `layouts/` from the Optimize tab), `best2_config.json`,
`initial_config.json`, `history.csv`, `summary.json` and `report.pdf`.
Ray sampling is random Monte Carlo; the optimiser re-checks every new best with a
fresh sample and the report re-evaluates the top designs with 4× the rays before
ranking them. Use ~1 cm cells inside the FOV (grid 60–80 with `"wall_size": "auto"`)
and ≥ 1000 rays per pixel for the search.

## Notes

- GPU acceleration is auto-detected; `simulation/gpu_backend.py` runs a one-off
  CPU-vs-GPU self-test and silently falls back to CPU if the driver returns
  inconsistent results. *Advanced → GPU → Purge GPU memory* releases device buffers.
- `interactive_lighting.py` is a 4-line legacy entry point kept for `LightingSim.bat`
  and `build_exe.py`.
