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
| `raytracing/` | Intersection kernels: absorber boxes, STL meshes (trimesh BVH), GPU (CuPy/Taichi) |
| `simulation/` | The engine: `compute_wall_intensity`, `compute_room_intensity`, ray emission, room geometry, `WallSettings`/`RoomSettings`/`EmissionSettings` |
| `scene/` | Saved config (`configs/*.json`) → LEDs, absorbers, STL occluder (`build_scene_from_config`) |
| `analysis/` | `uniformity_metrics` (U0, U1, CV, ΔEV) and legend HTML |
| `pipeline.py` | Headless facade: `simulate_wall(cfg, settings)` / `simulate_room(...)` → grid + metrics |
| `ui/app.py` | The Viser GUI (`main`). Only this module imports Viser |

Everything below `ui/` is UI-free and importable without Viser.

## Headless example (basis for optimisation loops)

```python
from lighting_simulator.pipeline import simulate_wall
from lighting_simulator.scene import load_config
from lighting_simulator.simulation import WallSettings

cfg = load_config("configs/elios3.json")
settings = WallSettings(wall_dist=100, grid_size=50, wall_size=80, rays_per_pixel=4)

def objective(cfg):
    result = simulate_wall(cfg, settings, use_gpu=False)   # result.grid is lux (Z rows, Y cols)
    return -result.metrics.u0                              # maximise Emin/Eavg

# mutate cfg["custom_groups"][i]["position"] / ["rotation_*"] / ["led_positions"] and re-evaluate
```

`scripts/ui_smoke.py` drives the real GUI callbacks headlessly (load config →
wall map → room map) and is a quick end-to-end check after UI edits.

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

Output goes to `exports/optim/<name>/`: `best_config.json` (copy to `configs/` and
open it in the UI via *Load Configuration*), `history.csv`, `summary.json`.
Evaluations are deterministic (fixed Monte Carlo seeds) so the optimiser sees a
smooth objective; raise `rays_per_pixel` for less noise (≈5 ms/eval at 30×30 cells ×
30 rays, so budgets of thousands are cheap).

## Notes

- GPU acceleration is auto-detected; `simulation/gpu_backend.py` runs a one-off
  CPU-vs-GPU self-test and silently falls back to CPU if the driver returns
  inconsistent results.
- `gpu_raytrace.py` at the repo root is a compatibility shim for the old
  benchmark scripts; the implementation is `lighting_simulator/raytracing/gpu.py`.
