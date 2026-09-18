"""Smoke-test the optimisation hooks used by the UI: base_cfg override, on_eval, stop_event, GPU flag.

Usage: PYTHONPATH=. python scripts/optim_smoke.py [--gpu]
"""
import sys
import threading
import time

from lighting_simulator.optimization import OptimizerSpec, load_spec, problem_from_spec, run
from lighting_simulator.scene.builder import load_config
from lighting_simulator.simulation import gpu_backend

use_gpu = "--gpu" in sys.argv
if use_gpu:
    print("[smoke] gpu_available:", gpu_backend.gpu_available(), gpu_backend.gpu_backend_label())

spec, spec_dir = load_spec("optimization_specs/ludo_refine.json")
base = load_config("tests/data/v1_configs/ludos_panels_n4.json")
spec['wall']['rays_per_pixel'] = 50
spec['variables'] = [{"type": "panel_pose", "group_index": 0, "pos_delta": [2, 2, 2], "rot_delta": [10, 10, 10]}]
problem = problem_from_spec(spec, spec_dir, base_cfg=base, use_gpu=use_gpu)
print(f"[smoke] problem: {problem.dim} vars, use_gpu={problem.use_gpu}, name={problem.name}")

stop = threading.Event()
seen = []


def on_eval(logger, ev, x):
    seen.append((logger.n, ev.score, logger.best.score))
    if logger.n >= 12:
        stop.set()


t0 = time.time()
summary, best = run(problem, OptimizerSpec(method="random_search", max_evals=200, population=5, log_every=0),
                    output_dir="exports/optim_smoke", on_eval=on_eval, stop_event=stop)
print(f"[smoke] {len(seen)} on_eval calls, stopped={summary['stopped']}, evals={summary['evaluations']}, "
      f"{time.time() - t0:.1f}s, best={best.summary()}")
assert summary['stopped'] and summary['evaluations'] == 12, summary
assert seen[0][0] == 1
print("[smoke] OK")
