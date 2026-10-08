"""Entry point of the simulation study: training, evaluation matrix, steepest slope, bound check,
figures, tables, key numbers, and the draft LaTeX snippet."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from ..params import Actuator, Rover
from . import interpret, plots_sim, report_sim
from .experiment import (COMPARISONS, CONDITIONS, METRICS, bound_matrix, long_table, memory, one_run,
                         paired_stats, run_matrix, steepest_matrix)
from .learning import fit_stage_a, fit_stage_c, soil_population, soils_key, strata, traction_index
from .params_sim import SimParams, SoilWR
from .wheel_soil import WheelTable

CODE_DIR = Path(__file__).resolve().parents[2]
OUT = CODE_DIR / "out"
DOCS = CODE_DIR / "docs"
PAPER_FIGS = CODE_DIR.parent / "paper" / "figures"


def load_context(n_jobs: int = -1, quick: bool = False, rail_start: float | None = None,
                 no_cache: bool = False) -> dict:
    """Parameters, soil population, split, cache wrapper, keys, and the fitted Stage A and C artefacts.

    Every call reproduces the cache keys of the main study, so the fits are served from the cache.
    Optional experiments use this to stay consistent with the main study."""
    import inspect
    OUT.mkdir(exist_ok=True); (OUT / "models").mkdir(exist_ok=True)
    rv, act, p = Rover(), Actuator(), SimParams()
    if rail_start is not None:
        rv = replace(rv, rail_start=rail_start)
    tag = "" if rail_start is None else f"_rail{int(round(rail_start * 1000))}"
    if quick:
        p = replace(p, n_soils=20, n_train=12, n_rep=3, slopes_deg=(0.0, 10.0, 20.0), grid_A=5, grid_C=3,
                    train_cycles=2, n_cycles=4, steep_tol=2.0)
        tag += "_quick"
    mem = memory(OUT)
    if no_cache:
        mem.clear(warn=False)
    # The soil population is hashed through a deterministic fingerprint (dataclass hashing is not stable
    # across processes), the worker count does not affect results, and fitted model objects hash
    # differently before and after pickling, so the evaluation caches are keyed on the training inputs
    # that determine them (art_key) and ignore the objects themselves.
    ignorable = {"soils", "n_jobs", "artefacts", "artefact"}
    cached = lambda f: mem.cache(f, ignore=[a for a in inspect.signature(f).parameters if a in ignorable])

    t_start = time.time()
    soils = soil_population(p)
    skey = soils_key(soils)
    idx = np.array([traction_index(s, rv, p) for s in soils])
    labels, cuts = strata(idx, p.n_classes)
    train_idx = list(range(p.n_train))
    test_idx = list(range(p.n_train, p.n_train + p.n_rep))
    test_all = list(range(p.n_train, p.n_soils))

    print("fitting Stage A ...", flush=True)
    A = cached(fit_stage_a)(rv, act, p, soils, labels, train_idx, test_all, cuts, n_jobs, soils_key=skey)
    print(f"  cv acc {A.cv_accuracy:.2f}, test acc {A.test_accuracy:.2f}", flush=True)
    print("fitting Stage C ...", flush=True)
    C = cached(fit_stage_c)(rv, act, p, soils, train_idx, n_jobs, soils_key=skey)
    print(f"  GP cv rmse {C.cv_rmse * 1e3:.1f} mm, target std {C.target_std * 1e3:.1f} mm", flush=True)
    joblib.dump(A, OUT / "models" / f"stageA{tag}.joblib"); joblib.dump(C, OUT / "models" / f"stageC{tag}.joblib")
    # The evaluation caches depend on what the learning components learned, so the key also carries the
    # Stage A table and the Stage C training rows (deterministic fingerprints, unlike the fitted objects).
    a_fp = tuple(sorted((k, round(float(v), 9)) for k, v in A.table.items()))
    c_fp = joblib.hash((np.asarray(C.X_train).tobytes(), np.asarray(C.y_train).tobytes()))
    art_key = joblib.hash((skey, rv, act, p, tuple(train_idx), tuple(test_all), a_fp, c_fp))
    return dict(rv=rv, act=act, p=p, soils=soils, labels=labels, cuts=cuts, idx=idx, train_idx=train_idx,
                test_idx=test_idx, test_all=test_all, tag=tag, mem=mem, cached=cached, skey=skey,
                art_key=art_key, A=A, C=C, t_start=t_start, n_jobs=n_jobs)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced study for development")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--rail-start", type=float, default=None, help="secondary rail placement (m)")
    args = ap.parse_args(argv)

    ctx = load_context(args.n_jobs, args.quick, args.rail_start, args.no_cache)
    rv, act, p, soils, labels, cuts, idx = (ctx[k] for k in ("rv", "act", "p", "soils", "labels", "cuts", "idx"))
    train_idx, test_idx, test_all, tag = (ctx[k] for k in ("train_idx", "test_idx", "test_all", "tag"))
    cached, skey, art_key, A, C, t_start = (ctx[k] for k in ("cached", "skey", "art_key", "A", "C", "t_start"))
    artefacts = {"A": A, "C": C}

    print("evaluation matrix ...", flush=True)
    S = cached(run_matrix)(rv, act, p, soils, test_idx, artefacts, args.n_jobs, CONDITIONS, None, None, soils_key=skey, art_key=art_key)
    S_all = cached(run_matrix)(rv, act, p, soils, test_all, artefacts, args.n_jobs, CONDITIONS, None, None, soils_key=skey, art_key=art_key)
    print("steepest slope ...", flush=True)
    steep = cached(steepest_matrix)(rv, act, p, soils, test_idx, artefacts, args.n_jobs, CONDITIONS, soils_key=skey, art_key=art_key)
    print("bound check ...", flush=True)
    bounds = cached(bound_matrix)(rv, act, p, soils, test_idx, C, args.n_jobs, None, soils_key=skey, art_key=art_key)
    for b in bounds:
        for s in S:
            if s.condition == "C" and s.soil == b["soil"] and s.theta_deg == b["slope_deg"]:
                s.bound_ratio = b["bound_ratio"]
                s.bound_ratio_i = b["bound_ratio_with_inertia"]

    # statistics
    stats_rows = []
    for th in p.slopes_deg:
        for a, b in COMPARISONS:
            for m in ("slip_mean", "cot", "travel_eff"):
                r = paired_stats(S, m, a, b, th); r["set"] = "protocol"; stats_rows.append(r)
                r = paired_stats(S_all, m, a, b, th); r["set"] = "all_test"; stats_rows.append(r)

    # traces for the figure: Stage C and B2 at the reference slope on the first test soil
    th_trace = 10.0 if 10.0 in p.slopes_deg else p.slopes_deg[min(1, len(p.slopes_deg) - 1)]
    traces = {cond: one_run(rv, act, p, soils, test_idx[0], th_trace, cond, artefacts.get(cond), None, True)
              for cond in ("B2", "C")}

    # figures
    plots_sim.setup_style()
    tab = WheelTable.build(SoilWR(), rv, p)
    plots_sim.plot_single_wheel(tab, rv, OUT / f"figS1{tag}")
    plots_sim.plot_traces(traces["C"], rv, act.T_phase, OUT / f"fig6a{tag}")
    plots_sim.plot_metrics_vs_slope(S_all, p.slopes_deg, OUT / f"fig6b{tag}")
    plots_sim.plot_steepest(steep, test_idx, OUT / f"fig6c{tag}")
    plots_sim.plot_offset_envelope(S, p.slopes_deg, rv, OUT / f"fig6d{tag}")
    pd_gp = interpret.gp_partial_dependences(C)
    rule = interpret.rule_numbers(pd_gp, rv)
    plots_sim.plot_gp_rule(pd_gp, rv, OUT / f"fig6e{tag}")

    # tables, numbers, snippet
    k = report_sim.key_numbers_sim(rv, act, p, S, S_all, steep, bounds, stats_rows, A, C, labels, cuts, idx,
                                   test_idx, time.time() - t_start)
    k["stageC_rule"] = rule
    report_sim.write_csv(long_table(S_all), OUT / f"results_sim{tag}.csv")
    report_sim.write_json(k, OUT / f"results_sim{tag}.json")
    (OUT / f"params_sim{tag}.md").write_text(report_sim.param_table_sim_md(SoilWR(), p))
    (OUT / f"params_sim{tag}.tex").write_text(report_sim.param_table_sim_tex(SoilWR(), p))
    if not tag:
        DOCS.mkdir(exist_ok=True)
        (DOCS / "parameters_simulation.md").write_text(report_sim.param_doc_sim_md(SoilWR(), p))
    (OUT / f"simulation_snippet{tag}.tex").write_text(report_sim.snippet_sim_tex(k, p, rv))
    if PAPER_FIGS.is_dir() and not tag:
        for f in ("fig6a", "fig6b", "fig6c", "fig6d", "fig6e", "figS1"):
            shutil.copy(OUT / f"{f}.pdf", PAPER_FIGS / f"{f}.pdf")
    print(json.dumps(k["criteria"], indent=1))
    print(f"done in {(time.time() - t_start) / 60:.1f} min; outputs in {OUT}")


if __name__ == "__main__":
    main()
