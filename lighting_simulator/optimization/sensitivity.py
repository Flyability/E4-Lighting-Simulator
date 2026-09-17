"""Post-run sensitivity analysis: which variables (and derived layout features) drive the score.

Everything is computed from the evaluations already logged during the run, so it costs
nothing extra: rank (Spearman) correlations between every input and every outcome, a
rank-linear surrogate of the score, and the co-variation of the inputs among the best designs.
Correlations are *associations over the points the optimiser happened to visit*, not causal
effects — the optimiser concentrates samples near good regions, so read them as "what the
search found worth changing", and check the partial-dependence plots for non-monotonic effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_DERIVED = {
    'n_leds': "active LEDs",
    'nn_spacing_cm': "mean nearest-neighbour LED spacing (cm)",
    'spread_cm': "LED spread: RMS distance from the LED centroid (cm)",
    'beam_off_axis_deg': "mean beam angle off the camera axis +X (°)",
    'mean_z_cm': "mean LED height Z (cm)",
    'mean_y_abs_cm': "mean lateral offset |Y| (cm)",
}


def rankdata(a):
    """Average ranks (ties share their mean rank)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def derived_features(problem, xs):
    """Layout descriptors of decoded designs: ``{name: (N,) array}`` (NaN when no LED is active)."""
    out = {k: np.full(len(xs), np.nan) for k in _DERIVED}
    for i, x in enumerate(xs):
        scene = problem.build_scene(problem.decode(x))
        leds = scene.active_leds
        out['n_leds'][i] = len(leds)
        if not leds:
            continue
        P = np.array([np.asarray(l.position, float) for l in leds])
        D = np.array([np.asarray(l.direction, float) for l in leds])
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        out['spread_cm'][i] = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
        out['beam_off_axis_deg'][i] = float(np.degrees(np.arccos(np.clip(D[:, 0], -1, 1))).mean())
        out['mean_z_cm'][i] = float(P[:, 2].mean())
        out['mean_y_abs_cm'][i] = float(np.abs(P[:, 1]).mean())
        if len(P) > 1:
            d = np.linalg.norm(P[:, None] - P[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            out['nn_spacing_cm'][i] = float(d.min(1).mean())
    return out


def outcomes(problem, evs):
    """Score components per evaluation: ``{label: (N,) array}`` (columns that are all-NaN are dropped)."""
    cols = {
        'score': [e.score for e in evs],
        'T1 U0 %': [e.uniformity_pct for e in evs],
        'T1 coverage': [e.coverage for e in evs],
        'T1 E_avg lx': [e.e_avg for e in evs],
    }
    flight = [m for m in problem.modes if not m.is_flash and m.vio_min_lux]
    if flight:
        cols['T2 VIO %'] = [(e.modes.get(flight[0].name, {}).get('vio_fraction') or np.nan) * 100 for e in evs]
    if problem.objective.tilt_fov_deg:
        cols['T3 U0 up %'] = [e.tilt.get('up', np.nan) if e.tilt else np.nan for e in evs]
        cols['T3 U0 down %'] = [e.tilt.get('down', np.nan) if e.tilt else np.nan for e in evs]
    cols['penalties Σ'] = [sum(e.penalties.values()) for e in evs]
    out = {k: np.asarray(v, float) for k, v in cols.items()}
    return {k: v for k, v in out.items() if np.isfinite(v).sum() >= 3 and np.nanstd(v) > 0}


@dataclass
class Sensitivity:
    input_names: list
    input_labels: list
    outcome_names: list
    X: np.ndarray
    """(N, n_inputs) input values (variables then derived features)."""
    Y: np.ndarray
    """(N, n_outcomes)."""
    corr: np.ndarray
    """Spearman ρ, (n_inputs, n_outcomes)."""
    surrogate_coef: np.ndarray
    """Standardised rank-regression coefficients of the score on the variables (n_vars,)."""
    surrogate_r2: float
    n_vars: int
    top_mask: np.ndarray
    """Records in the best ``top_frac`` of scores."""
    top_corr: np.ndarray
    """Spearman ρ between variables among the top designs (n_vars, n_vars)."""
    top_range_frac: np.ndarray
    """Share of each variable's bound range still spanned by the top designs (n_vars,)."""
    n_records: int = 0
    n_derived_records: int = 0
    notes: list = field(default_factory=list)

    def ranking(self):
        """Input indices sorted by |ρ(score)| descending (NaN last)."""
        s = np.abs(self.corr[:, self.outcome_names.index('score')])
        return list(np.argsort(np.where(np.isfinite(s), -s, np.inf)))

    def weak_variables(self, rho_max=0.1, range_min=0.6):
        """Variables the score barely reacts to while the best designs still use most of their range."""
        s = np.abs(self.corr[:self.n_vars, self.outcome_names.index('score')])
        c = np.abs(self.surrogate_coef)
        c = c / c.max() if np.isfinite(c).any() and np.nanmax(c) > 0 else c
        return [i for i in range(self.n_vars)
                if np.isfinite(s[i]) and s[i] < rho_max and c[i] < 0.15 and self.top_range_frac[i] > range_min]


def analyse(problem, records, top_frac=0.2, max_derived=1200, seed=0):
    """Sensitivity of the logged run. ``records`` = ``RunLogger.records`` [(eval_no, Evaluation, x)]."""
    evs = [r[1] for r in records]
    X = np.array([np.asarray(r[2], float) for r in records])
    score = np.array([e.score for e in evs], float)
    names = list(problem.names)
    n_vars = len(names)
    labels = list(names)

    # Derived layout features on a subsample (decoding every record is the only cost here)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    if len(idx) > max_derived:
        best = np.argsort(score)[:max_derived // 4]  # keep the good region well represented
        rest = rng.choice(np.setdiff1d(idx, best), max_derived - len(best), replace=False)
        idx = np.sort(np.concatenate([best, rest]))
    feats = derived_features(problem, X[idx])
    F = np.full((len(records), len(feats)), np.nan)
    F[idx] = np.column_stack([feats[k] for k in feats])
    names += list(feats)
    labels += [_DERIVED[k] for k in feats]
    XF = np.hstack([X, F])

    outs = outcomes(problem, evs)
    out_names = list(outs)
    Y = np.column_stack([outs[k] for k in out_names])
    corr = np.array([[spearman(XF[:, i], Y[:, j]) for j in range(Y.shape[1])] for i in range(XF.shape[1])])

    # Rank-linear surrogate of the score on the variables (ridge on standardised ranks)
    Rx = np.column_stack([rankdata(X[:, i]) for i in range(n_vars)])
    ry = rankdata(score)
    keep = Rx.std(0) > 0
    Z = (Rx[:, keep] - Rx[:, keep].mean(0)) / Rx[:, keep].std(0)
    zy = (ry - ry.mean()) / (ry.std() or 1.0)
    lam = 1e-3 * len(ry)
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ zy)
    coef = np.zeros(n_vars)
    coef[keep] = beta
    pred = Z @ beta
    r2 = float(1.0 - ((zy - pred) ** 2).sum() / max(1e-12, (zy ** 2).sum()))

    n_top = max(5, int(round(top_frac * len(score))))
    top_mask = np.zeros(len(score), bool)
    top_mask[np.argsort(score)[:n_top]] = True
    top_corr = np.array([[spearman(X[top_mask, i], X[top_mask, j]) for j in range(n_vars)] for i in range(n_vars)])
    lo = np.array([b[0] for b in problem.bounds], float)
    hi = np.array([b[1] for b in problem.bounds], float)
    span = np.where(hi > lo, hi - lo, 1.0)
    top_range_frac = np.ptp(X[top_mask], axis=0) / span

    notes = []
    if len(records) < 200:
        notes.append(f"Only {len(records)} evaluations: correlations below |ρ| ≈ {2/np.sqrt(len(records)):.2f} are within "
                     "noise, and the freeze candidates are tentative.")
    if r2 < 0.3:
        notes.append(f"The rank-linear surrogate explains only {r2*100:.0f} % of the score variance: interactions or "
                     "non-monotonic effects dominate — rely on the partial-dependence plots rather than the bars.")
    return Sensitivity(input_names=names, input_labels=labels, outcome_names=out_names, X=XF, Y=Y, corr=corr,
                       surrogate_coef=coef, surrogate_r2=r2, n_vars=n_vars, top_mask=top_mask, top_corr=top_corr,
                       top_range_frac=top_range_frac, n_records=len(records), n_derived_records=len(idx), notes=notes)


def binned_trend(x, y, n_bins=12):
    """(bin centres, median y per bin, 25th, 75th percentiles) over the finite pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5 or np.ptp(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    uniq = np.unique(x)
    if len(uniq) <= n_bins:  # integer / categorical variable: one bin per value
        edges_c = uniq
        groups = [y[x == u] for u in uniq]
    else:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        edges_c = 0.5 * (edges[:-1] + edges[1:])
        groups = [y[which == k] for k in range(len(edges) - 1)]
    med = np.array([np.median(g) if g.size else np.nan for g in groups])
    q1 = np.array([np.percentile(g, 25) if g.size else np.nan for g in groups])
    q3 = np.array([np.percentile(g, 75) if g.size else np.nan for g in groups])
    return np.asarray(edges_c, float), med, q1, q3
