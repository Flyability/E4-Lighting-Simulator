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
    confirm_best: bool = True
    """Re-evaluate any new best with fresh rays and keep the mean, so a lucky Monte-Carlo
    draw cannot be locked in by the optimiser's elitism."""
    log_every: int = 10
    x0: list | None = None


class OptimizationStopped(Exception):
    """Raised inside the objective when the caller's ``stop_event`` is set."""


class RunLogger:
    def __init__(self, problem: Problem, out_dir: Path, log_every=10, on_eval=None, stop_event=None,
                 confirm_best=True):
        """``on_eval(logger, evaluation, x)`` is called after every logged evaluation
        (from the optimiser thread); ``stop_event`` (threading.Event) aborts the run."""
        self.problem = problem
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.on_eval = on_eval
        self.stop_event = stop_event
        self.confirm_best = confirm_best
        self.n_confirmations = 0
        self.best: Evaluation | None = None
        self.best_x = None
        self.n = 0
        self.n_external = 0
        """Evaluations performed in worker processes (not individually logged)."""
        self.records = []
        """In-memory history ``(eval_no, Evaluation without grid, x)`` for the report."""
        self.t0 = time.perf_counter()
        self._hist = open(self.out_dir / "history.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._hist)
        self._csv.writerow(["eval", "t_s", "score", "uniformity_pct", "coverage", "e_avg", "n_active",
                            "n_drivers", "total_current_a"] + problem.names)

    def __call__(self, x):
        ev = self.problem.evaluate(x)
        if self.confirm_best and self.best is not None and ev.score < self.best.score:
            # Candidate beats the best: confirm with an independent ray sample.
            ev = ev.averaged_with(self.problem.evaluate(x))
            self.n_confirmations += 1
        self.n += 1
        self._csv.writerow([self.n, f"{time.perf_counter() - self.t0:.2f}", f"{ev.score:.6f}",
                            f"{ev.uniformity_pct:.3f}", f"{ev.coverage:.4f}", f"{ev.e_avg:.2f}", ev.n_active,
                            ev.n_drivers, f"{ev.total_current_a:.3f}"]
                           + [f"{v:.5g}" for v in np.asarray(x, float)])
        self.records.append((max(self.n, self.n_external), ev, np.array(x, dtype=float)))
        if self.best is None or ev.score < self.best.score:
            self.best, self.best_x = ev, np.array(x, dtype=float)
            self._save_best()
            print(f"  [{self.n:5d}] NEW BEST {ev.summary()}")
        elif self.log_every and self.n % self.log_every == 0:
            print(f"  [{self.n:5d}] {ev.summary()}  (best {self.best.score:.4f})")
        if self.on_eval is not None:
            self.on_eval(self, ev, x)
        if self.stop_event is not None and self.stop_event.is_set():
            raise OptimizationStopped()
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
            "confirmations": self.n_confirmations,
            "elapsed_s": round(time.perf_counter() - self.t0, 2),
            "best_score": self.best.score if self.best else None,
            "best": {k: v for k, v in asdict(self.best).items() if k not in ("metrics", "grid", "vio_grid")} if self.best else None,
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


def unique_run_dir(output_dir, name):
    """``<output_dir>/<name>``, or ``<name>_002``, ``_003``… if that folder already has outputs."""
    base = Path(output_dir)
    cand = base / name
    if not cand.exists() or not any(cand.iterdir()):
        return cand
    n = 2
    while (base / f"{name}_{n:03d}").exists():
        n += 1
    return base / f"{name}_{n:03d}"


def run(problem: Problem, opt: OptimizerSpec, output_dir="exports/optim", on_eval=None, stop_event=None,
        report=True):
    """Optimise ``problem`` and return (summary dict, best Evaluation).

    ``on_eval`` / ``stop_event`` are forwarded to ``RunLogger`` for live progress and
    cancellation (a stopped run still writes its summary and best config). ``report``
    renders ``report.pdf`` next to the other outputs. Existing run folders are never
    overwritten; the actual folder is ``summary['run_dir']``.
    """
    from scipy import optimize

    if problem.use_gpu and opt.workers != 1:
        # GPU contexts cannot be shared with forked/spawned evaluation workers.
        print("[optim] use_gpu=True: forcing workers=1")
        opt = OptimizerSpec(**{**asdict(opt), 'workers': 1})

    out_dir = unique_run_dir(output_dir, problem.name)
    if out_dir.name != problem.name:
        print(f"[optim] '{problem.name}' already has results → writing to {out_dir}")
    logger = RunLogger(problem, out_dir, log_every=opt.log_every, on_eval=on_eval, stop_event=stop_event,
                       confirm_best=opt.confirm_best)
    bounds = problem.bounds
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(opt.x0, float) if opt.x0 is not None else problem.x0, lo, hi)
    rng = np.random.default_rng(opt.seed)

    print(f"[optim] {problem.name}: {problem.dim} variables, method={opt.method}, budget={opt.max_evals} evals"
          + (" [GPU]" if problem.use_gpu else ""))
    stopped = False
    try:
        logger(x0)  # initial guess is eval #1 so ``best`` is always defined
        print(f"[optim] initial guess: {logger.best.summary()}")
        _run_method(problem, opt, logger, x0, lo, hi, rng, optimize)
    except OptimizationStopped:
        stopped = True
        print(f"[optim] stopped by user after {logger.n} evaluations")

    elapsed = time.perf_counter() - logger.t0
    report_path = None
    if report and logger.records:
        try:
            from .report import write_report
            report_path = write_report(problem, logger.records, opt, out_dir, x0=x0, elapsed=elapsed,
                                       stopped=stopped, n_confirm=logger.n_confirmations)
            print(f"[optim] report: {report_path}")
        except Exception:  # a report failure must not lose the optimisation result
            import traceback
            traceback.print_exc()
            print("[optim] report generation failed (see traceback above)")

    summary = logger.close({"optimizer": asdict(opt), "stopped": stopped, "run_dir": str(out_dir),
                            "report": str(report_path) if report_path else None})
    print(f"[optim] done: {summary['evaluations']} evals in {summary['elapsed_s']}s → {logger.best.summary()}")
    print(f"[optim] best config: {out_dir / 'best_config.json'}")
    return summary, logger.best


def _run_method(problem, opt, logger, x0, lo, hi, rng, optimize):
    bounds = problem.bounds
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
        ints = problem.integrality
        if ints.any():
            print("[optim] WARNING: nelder_mead is a local continuous method; integer variables "
                  f"({int(ints.sum())}: rows/cols, on-off, roles) will only change by whole steps of the initial "
                  "simplex. Prefer differential_evolution or random_search for layouts with counts or roles.")
        # SciPy's default simplex is 5 % of x0 (0.00025 when x0 == 0): variables starting at 0 (tilts,
        # offsets) or integers would never move. Use a quarter of each variable's range instead.
        n = problem.dim
        simplex = np.tile(x0, (n + 1, 1))
        for i in range(n):
            step = max(0.25 * (hi[i] - lo[i]), 1.0 if ints[i] else 0.0)
            if step <= 0:
                continue
            simplex[i + 1, i] = x0[i] + step if x0[i] + step <= hi[i] else x0[i] - step
        simplex = np.clip(simplex, lo, hi)
        optimize.minimize(
            lambda x: logger(_snap_integers(problem, np.clip(x, lo, hi))), x0, method="Nelder-Mead",
            options={"maxfev": opt.max_evals, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True,
                     "initial_simplex": simplex},
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


def run_spec(spec, spec_dir=None, **run_kwargs):
    from .problem import problem_from_spec
    problem = problem_from_spec(spec, spec_dir)
    opt = OptimizerSpec(**spec.get('optimizer', {}))
    return run(problem, opt, output_dir=spec.get('output_dir', 'exports/optim'), **run_kwargs)
