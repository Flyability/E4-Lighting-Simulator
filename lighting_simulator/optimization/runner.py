"""Optimisation drivers (scipy) with progress logging and best-config export.

Methods
-------
- ``differential_evolution`` (default): global, handles bounds and binary variables.
- ``nelder_mead``: local refinement from the initial guess / a given x0.
- ``random_search``: uniform sampling then Gaussian refinement around the best.

Every run writes ``<output_dir>/<name>/best_config.json`` (loadable in the UI),
``history.csv`` and ``summary.json``.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .problem import Evaluation, Problem


@dataclass
class OptimizerSpec:
    method: str = "differential_evolution"
    max_evals: int = 300
    seed: int = 0
    population: int = 20
    """Absolute population size (DE / random search)."""
    workers: int = 1
    """Parallel objective evaluations (DE only; -1 = all cores)."""
    polish: bool = False
    """Finish DE with a local Nelder-Mead pass."""
    log_every: int = 10
    x0: list | None = None


class RunLogger:
    def __init__(self, problem: Problem, out_dir: Path, log_every=10):
        self.problem = problem
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.best: Evaluation | None = None
        self.best_x = None
        self.n = 0
        self.n_external = 0
        """Evaluations performed in worker processes (not individually logged)."""
        self.t0 = time.perf_counter()
        self._hist = open(self.out_dir / "history.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._hist)
        self._csv.writerow(["eval", "t_s", "score", "uniformity_pct", "coverage", "e_avg", "n_active"]
                           + problem.names)

    def __call__(self, x):
        ev = self.problem.evaluate(x)
        self.n += 1
        self._csv.writerow([self.n, f"{time.perf_counter() - self.t0:.2f}", f"{ev.score:.6f}",
                            f"{ev.uniformity_pct:.3f}", f"{ev.coverage:.4f}", f"{ev.e_avg:.2f}", ev.n_active]
                           + [f"{v:.5g}" for v in np.asarray(x, float)])
        if self.best is None or ev.score < self.best.score:
            self.best, self.best_x = ev, np.array(x, dtype=float)
            self._save_best()
            print(f"  [{self.n:5d}] NEW BEST {ev.summary()}")
        elif self.log_every and self.n % self.log_every == 0:
            print(f"  [{self.n:5d}] {ev.summary()}  (best {self.best.score:.4f})")
        return ev.score

    def _save_best(self):
        cfg = self.problem.decode(self.best_x)
        cfg['name'] = self.problem.name
        cfg['description'] = f"Optimised: {self.best.summary()}"
        with open(self.out_dir / "best_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self._hist.flush()

    def close(self, extra=None):
        self._hist.close()
        summary = {
            "name": self.problem.name,
            "evaluations": max(self.n, self.n_external),
            "elapsed_s": round(time.perf_counter() - self.t0, 2),
            "best_score": self.best.score if self.best else None,
            "best": {k: v for k, v in asdict(self.best).items() if k not in ("metrics", "grid")} if self.best else None,
            "best_x": dict(zip(self.problem.names, map(float, self.best_x))) if self.best_x is not None else None,
        }
        if extra:
            summary.update(extra)
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        return summary


def _snap_integers(problem, x):
    x = np.array(x, dtype=float)
    ints = problem.integrality
    if ints.any():
        x[ints] = np.round(x[ints])
    return x


def run(problem: Problem, opt: OptimizerSpec, output_dir="exports/optim"):
    """Optimise ``problem`` and return (summary dict, best Evaluation)."""
    from scipy import optimize

    out_dir = Path(output_dir) / problem.name
    logger = RunLogger(problem, out_dir, log_every=opt.log_every)
    bounds = problem.bounds
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(opt.x0, float) if opt.x0 is not None else problem.x0, lo, hi)
    rng = np.random.default_rng(opt.seed)

    print(f"[optim] {problem.name}: {problem.dim} variables, method={opt.method}, budget={opt.max_evals} evals")
    print(f"[optim] initial guess: {problem.evaluate(x0).summary()}")

    if opt.method == "differential_evolution":
        n = problem.dim
        popsize = max(1, math.ceil(opt.population / n))
        pop_total = popsize * n
        maxiter = max(1, math.ceil(opt.max_evals / pop_total) - 1)
        init = rng.uniform(lo, hi, size=(pop_total, n))
        init[0] = x0
        generations = [0]

        if opt.workers == 1:
            objective = logger

            def stop(xk, convergence=None):
                return logger.n >= opt.max_evals
        else:
            # Workers evaluate the (picklable) Problem; the parent re-evaluates the
            # per-generation best once so logging / best_config export still work.
            objective = problem
            print(f"[optim] parallel DE: {pop_total} evals/generation, {maxiter + 1} generations")

            def stop(xk, convergence=None):
                generations[0] += 1
                logger(xk)
                logger.n_external = (generations[0] + 1) * pop_total
                return logger.n_external >= opt.max_evals

        optimize.differential_evolution(
            objective, bounds, init=init, maxiter=maxiter, popsize=popsize, seed=opt.seed,
            integrality=problem.integrality if problem.integrality.any() else None,
            polish=opt.polish, workers=opt.workers, updating="deferred" if opt.workers != 1 else "immediate",
            callback=stop, tol=0.0, atol=0.0,
        )
    elif opt.method == "nelder_mead":
        optimize.minimize(
            lambda x: logger(_snap_integers(problem, np.clip(x, lo, hi))), x0, method="Nelder-Mead",
            options={"maxfev": opt.max_evals, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True},
        )
    elif opt.method == "random_search":
        n_explore = max(1, min(opt.max_evals, opt.population))
        for i in range(n_explore):
            x = x0 if i == 0 else rng.uniform(lo, hi)
            logger(_snap_integers(problem, x))
        sigma = 0.25 * (hi - lo)
        while logger.n < opt.max_evals:
            x = np.clip(logger.best_x + rng.normal(0.0, 1.0, problem.dim) * sigma, lo, hi)
            before = logger.best.score
            logger(_snap_integers(problem, x))
            sigma *= 1.05 if logger.best.score < before else 0.97  # 1+1-ES style step adaptation
    else:
        raise ValueError(f"unknown optimizer method {opt.method!r}")

    summary = logger.close({"optimizer": asdict(opt)})
    print(f"[optim] done: {summary['evaluations']} evals in {summary['elapsed_s']}s → {logger.best.summary()}")
    print(f"[optim] best config: {out_dir / 'best_config.json'}")
    return summary, logger.best


def run_spec(spec, spec_dir=None):
    from .problem import problem_from_spec
    problem = problem_from_spec(spec, spec_dir)
    opt = OptimizerSpec(**spec.get('optimizer', {}))
    return run(problem, opt, output_dir=spec.get('output_dir', 'exports/optim'))
