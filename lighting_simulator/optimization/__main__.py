"""CLI: ``python -m lighting_simulator.optimization SPEC.json [--evals N] [--method M] [--workers W]``."""

import argparse
import multiprocessing

from .problem import load_spec, problem_from_spec
from .runner import run_spec
from lighting_simulator.scene import load_config


def main():
    parser = argparse.ArgumentParser(description="Optimise LED placement for wall uniformity.")
    parser.add_argument("spec", help="JSON problem spec (see optimization_specs/)")
    parser.add_argument("--evals", type=int, help="override optimizer.max_evals")
    parser.add_argument("--method", choices=["differential_evolution", "nelder_mead", "random_search"])
    parser.add_argument("--workers", type=int, help="parallel evaluations (DE only; -1 = all cores)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", help="override output_dir")
    parser.add_argument("--name", help="override run name (output folder)")
    parser.add_argument("--analytic", action="store_true",
                        help="score with the closed-form direct illuminance instead of ray tracing (spec key 'analytic')")
    parser.add_argument("--save-config", metavar="PATH", help="also copy best_config.json to this path (e.g. configs/opti_1.json)")
    parser.add_argument("--evaluate", metavar="CONFIG.json", action="append",
                        help="score an existing config under the spec's objective instead of optimising (repeatable)")
    args = parser.parse_args()

    spec, spec_dir = load_spec(args.spec)
    opt = spec.setdefault("optimizer", {})
    if args.evals is not None:
        opt["max_evals"] = args.evals
    if args.method:
        opt["method"] = args.method
    if args.workers is not None:
        opt["workers"] = args.workers
    if args.seed is not None:
        opt["seed"] = args.seed
    if args.output_dir:
        spec["output_dir"] = args.output_dir
    if args.name:
        spec["name"] = args.name
    if args.analytic:
        spec["analytic"] = True
    if args.evaluate:
        problem = problem_from_spec(spec, spec_dir)
        w, c = problem.wall, problem.camera
        dists = ", ".join(f"{x.wall_dist:g}" for x in problem.walls)
        print(f"[eval] wall x=[{dists}] grid={w.grid_size} rpp={w.rays_per_pixel} | "
              f"camera ({c.pos_x}, {c.pos_y}) pitch={c.pitch} fov={c.fov_h}x{c.fov_v} | metric={problem.objective.metric}")
        for path in args.evaluate:
            print(f"  {path}: {problem.evaluate_config(load_config(path)).summary()}")
        return
    summary, _best = run_spec(spec, spec_dir)
    if args.save_config:
        import shutil
        from pathlib import Path
        src = Path(summary["run_dir"]) / "best_config.json"
        shutil.copyfile(src, args.save_config)
        print(f"[optim] copied best config to {args.save_config}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
