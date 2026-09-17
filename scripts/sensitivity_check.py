"""Quick check of the sensitivity pages: short DE run on the flash/VIO spec, then dump the last pages."""
import sys
import time
from pathlib import Path

from lighting_simulator.optimization import OptimizerSpec, run
from lighting_simulator.optimization.problem import load_spec, problem_from_spec

out = Path(sys.argv[1] if len(sys.argv) > 1 else "exports/sens_check")
spec, spec_dir = load_spec("optimization_specs/elios4_ducts_flash_vio.json")
spec['wall'].update({"grid_size": 24, "rays_per_pixel": 30})
spec['use_gpu'] = True
spec['vio'].update({"room_grid_size": 12})
spec['objective'].update({"tilt_room_grid_size": 16})
problem = problem_from_spec(spec, spec_dir)
t = time.time()
summary, best = run(problem, OptimizerSpec(method="differential_evolution", max_evals=90, population=10, log_every=0),
                    output_dir=out)
print("elapsed", round(time.time() - t, 1), "s; report:", summary['report'])
