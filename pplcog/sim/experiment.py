"""Experiment driver: evaluation matrix, steepest-slope search, paired statistics, caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from joblib import Memory, Parallel, delayed
from scipy import stats

from ..params import Actuator, Rover
from .params_sim import SimParams, SoilWR
from .simulate import RunResult, run_episode

CONDITIONS = ("B1", "B2", "B3", "A", "C")
COMPARISONS = (("A", "B2"), ("A", "B3"), ("C", "B3"), ("C", "B2"), ("B2", "B1"), ("B3", "B2"))
METRICS = ("slip_mean", "slip_max", "anchor_back_over_stroke", "z_F_mean", "z_R_mean", "travel_eff", "cot",
           "cog_err_rms", "NF_min")


@dataclass
class Summary:
    condition: str
    theta_deg: float
    soil: int
    completed: bool
    fail_reason: str | None
    fail_time: float
    d: float
    cot: float
    E_wheel: float
    E_act: float
    E_rail: float
    Delta_max: float | None
    dx_max_abs: float | None
    cycles: list[dict]
    rec: dict | None = None


def summarize(r: RunResult, soil: int, keep_rec: bool, theta_deg: float) -> Summary:
    return Summary(r.meta["condition"], float(theta_deg), soil, r.completed, r.fail_reason,
                   float(r.fail_time), r.d, float(r.cot), r.E_wheel, r.E_act, r.E_rail, r.meta.get("Delta_max"),
                   r.meta.get("dx_max_abs"), r.cycles, r.rec if keep_rec else None)


def seed_for(soil: int, theta_deg: float) -> int:
    """Common random numbers: the sensor-noise seed depends on soil and slope only."""
    return 1000 * soil + int(round(theta_deg * 10))


def one_run(rv: Rover, act: Actuator, p: SimParams, soils: list[SoilWR], soil: int, theta_deg: float,
            condition: str, artefact=None, n_cycles: int | None = None, keep_rec: bool = False) -> Summary:
    r = run_episode(rv, act, p, soils[soil], np.radians(theta_deg), condition, artefact=artefact,
                    n_cycles=n_cycles, seed=seed_for(soil, theta_deg), keep_rec=keep_rec)
    return summarize(r, soil, keep_rec, theta_deg)


def run_matrix(rv: Rover, act: Actuator, p: SimParams, soils: list[SoilWR], test_idx: list[int],
               artefacts: dict, n_jobs: int = -1, conditions=CONDITIONS, slopes=None,
               keep_rec_for: tuple[str, float, int] | None = None, soils_key: str | None = None,
               art_key: str | None = None) -> list[Summary]:
    slopes = p.slopes_deg if slopes is None else slopes
    jobs = []
    for th in slopes:
        for cond in conditions:
            for si in test_idx:
                keep = keep_rec_for is not None and (cond, th, si) == keep_rec_for
                jobs.append(delayed(one_run)(rv, act, p, soils, si, th, cond, artefacts.get(cond), None, keep))
    return Parallel(n_jobs=n_jobs)(jobs)


def steepest_slope(rv: Rover, act: Actuator, p: SimParams, soils: list[SoilWR], soil: int, condition: str,
                   artefact=None) -> float:
    """Largest slope (deg) at which ten cycles complete, by bisection to p.steep_tol."""
    lo, hi = p.steep_lo, p.steep_hi
    ok = lambda th: one_run(rv, act, p, soils, soil, th, condition, artefact).completed
    if not ok(lo):
        return float("nan")
    if ok(hi):
        return hi
    while hi - lo > p.steep_tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def steepest_matrix(rv, act, p, soils, test_idx, artefacts, n_jobs=-1, conditions=CONDITIONS,
                    soils_key: str | None = None, art_key: str | None = None) -> dict:
    jobs, keys = [], []
    for cond in conditions:
        for si in test_idx:
            jobs.append(delayed(steepest_slope)(rv, act, p, soils, si, cond, artefacts.get(cond)))
            keys.append((cond, si))
    vals = Parallel(n_jobs=n_jobs)(jobs)
    return {k: v for k, v in zip(keys, vals)}


def run_metric(s: Summary, metric: str, skip_first: int = 1) -> float:
    """Run-level metric: mean over cycles after the first (probing) cycle. A failed run has no
    run-level metric: inf for cot and nan otherwise, so that paired statistics only use soils on
    which both runs completed (the partial cycles of a failed run are not comparable)."""
    if metric == "cot":
        return s.cot if s.completed else np.inf
    if metric == "steepest":
        raise ValueError
    if not s.completed:
        return np.nan
    cyc = s.cycles[skip_first:] if len(s.cycles) > skip_first else s.cycles
    if not cyc:
        return np.nan
    return float(np.mean([c[metric] for c in cyc]))


def long_table(summaries: list[Summary]) -> list[dict]:
    rows = []
    for s in summaries:
        for m in METRICS:
            rows.append({"slope_deg": s.theta_deg, "condition": s.condition, "soil": s.soil, "metric": m,
                         "value": run_metric(s, m), "completed": s.completed, "fail_reason": s.fail_reason})
    return rows


def paired_stats(summaries: list[Summary], metric: str, cond_a: str, cond_b: str, theta_deg: float,
                 n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison a - b across soils at one slope."""
    A = {s.soil: run_metric(s, metric) for s in summaries if s.condition == cond_a and s.theta_deg == theta_deg}
    B = {s.soil: run_metric(s, metric) for s in summaries if s.condition == cond_b and s.theta_deg == theta_deg}
    soils = sorted(set(A) & set(B))
    a = np.array([A[k] for k in soils]); b = np.array([B[k] for k in soils])
    finite = np.isfinite(a) & np.isfinite(b)
    d = a[finite] - b[finite]
    n = int(d.size)
    out = {"metric": metric, "a": cond_a, "b": cond_b, "slope_deg": theta_deg, "n": n,
           "n_total": len(soils), "mean_a": float(np.mean(a[finite])) if n else np.nan,
           "mean_b": float(np.mean(b[finite])) if n else np.nan}
    if n < 2 or np.allclose(d, 0):
        out.update({"mean_diff": float(np.mean(d)) if n else np.nan, "t_p": np.nan, "wilcoxon_p": np.nan,
                    "median_diff": float(np.median(d)) if n else np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                    "d_z": np.nan, "n_80pct_power": np.nan})
        return out
    t = stats.ttest_rel(a[finite], b[finite])
    try:
        w = stats.wilcoxon(d, method="exact")
        wp = float(w.pvalue)
    except ValueError:
        wp = np.nan
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(d, size=(n_boot, n), replace=True), axis=1)
    sd = float(np.std(d, ddof=1))
    d_z = float(np.mean(d) / sd) if sd > 0 else np.inf
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    n80 = float(np.ceil(((z_a + z_b) / d_z) ** 2 + z_a ** 2 / 2)) if np.isfinite(d_z) and d_z != 0 else np.inf
    out.update({"mean_diff": float(np.mean(d)), "t_p": float(t.pvalue), "wilcoxon_p": wp,
                "median_diff": float(np.median(d)), "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)), "d_z": d_z, "n_80pct_power": n80})
    return out


def memory(out_dir: Path) -> Memory:
    return Memory(str(out_dir / "cache"), verbose=0)


def bound_check(rv: Rover, act: Actuator, p: SimParams, soils: list[SoilWR], soil: int, theta_deg: float,
                artefact) -> dict:
    """Stage C against its paired zero-offset run: realised deviation of N_F over the Eq. (5) bound."""
    from ..statics import NF_deviation_bound
    c = one_run(rv, act, p, soils, soil, theta_deg, "C", artefact, None, True)
    b = one_run(rv, act, p, soils, soil, theta_deg, "C", None, None, True)
    n = min(len(c.rec["t"]), len(b.rec["t"]))
    Dm = c.Delta_max or 0.0
    if n == 0 or Dm <= 0:
        return {"soil": soil, "slope_deg": theta_deg, "bound_ratio": np.nan, "bound_ratio_with_inertia": np.nan,
                "dx_ratio": np.nan, "xi_ratio": np.nan}
    theta = np.radians(theta_deg)
    bound = NF_deviation_bound(theta, c.rec["L"][:n], rv, Dm)
    # Eq. (5) plus the inertial allowance of the mass actuator that the manuscript bounds separately
    inertial = 2.0 * rv.m_m * act.a_max * rv.h_rail / c.rec["L"][:n]
    dev = np.abs(c.rec["N_F"][:n] - b.rec["N_F"][:n])
    xi_dev = np.abs(c.rec["xi_m"][:n] - b.rec["xi_m"][:n])
    return {"soil": soil, "slope_deg": theta_deg, "bound_ratio": float(np.max(dev / bound)),
            "bound_ratio_with_inertia": float(np.max(dev / (bound + inertial))),
            "dx_ratio": float((c.dx_max_abs or 0.0) / Dm), "xi_ratio": float(np.max(xi_dev) / Dm)}


def bound_matrix(rv, act, p, soils, test_idx, artefact, n_jobs=-1, slopes=None, soils_key: str | None = None,
                 art_key: str | None = None) -> list[dict]:
    slopes = p.slopes_deg if slopes is None else slopes
    return Parallel(n_jobs=n_jobs)(delayed(bound_check)(rv, act, p, soils, si, th, artefact)
                                   for th in slopes for si in test_idx)
