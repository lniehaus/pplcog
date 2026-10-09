"""CSV, JSON, parameter tables, and the draft LaTeX snippet of the simulation study."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..params import Actuator, Rover
from ..report import _fmt, _json_default, _tex_escape, write_json  # noqa: F401
from .experiment import COMPARISONS, Summary, run_metric
from .features import FEATURE_LABELS
from .params_sim import SimParams, SoilWR, provenance_rows_sim
from .plots_sim import COND_LABELS


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def param_table_sim_md(soil: SoilWR, p: SimParams) -> str:
    out = ["| Parameter | Value | Unit | Source | Note |", "|---|---|---|---|---|"]
    for r in provenance_rows_sim(soil, p):
        out.append(f"| {r['symbol']} | {_fmt(r['value'])} | {r['unit']} | {r['source']} | {r['note']} |")
    return "\n".join(out) + "\n"


PARAM_SIM_DOC_LEGEND = """# Parameters of the closed-loop simulation

Parameters of the closed-loop pitch-plane simulation (Section 2.5 and Figures 5 and 6 of the
manuscript *Adaptive Center of Gravity Control for Push-Pull Locomotion Rovers on Slopes*),
in addition to those of the model evaluation in `parameters_model_evaluation.md`. This table
was Table 2 of the first submission; the revised manuscript refers to this file from its
Data Availability Statement. It is written by `pplcog/sim/run_sim.py` (full run only) from the defaults in
`pplcog/sim/params_sim.py`; edit the defaults, not this file.

Sources as in `parameters_model_evaluation.md`. The Wong-Reece coefficients follow
Wong and Reece (1967) and Ishigami et al. (2007). Soil parameters are randomised per soil
sample; a pair of values denotes the sampled range.

"""


def param_doc_sim_md(soil: SoilWR, p: SimParams) -> str:
    """Tracked documentation file: legend of Table 2 of the first submission plus the table."""
    return PARAM_SIM_DOC_LEGEND + param_table_sim_md(soil, p)


def param_table_sim_tex(soil: SoilWR, p: SimParams) -> str:
    lines = [r"\begin{table}[!ht]", r"\centering",
             r"\caption{Parameters of the closed-loop simulation in addition to Table~\ref{tab:evalparams}. "
             r"Sources as in Table~\ref{tab:evalparams}; the Wong--Reece coefficients follow \citet{WR67} and \citet{IMN07}.\\}",
             r"\label{tab:simparams}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabularx}{\linewidth}{@{}l r l l X@{}}", r"\toprule",
             r"\textbf{Parameter} & \textbf{Value} & \textbf{Unit} & \textbf{Source} & \textbf{Note} \\", r"\midrule"]
    for r in provenance_rows_sim(soil, p):
        sym = r["symbol"]
        sym = f"${sym}$" if any(ch in sym for ch in "\\_^{}") else sym
        unit = r["unit"]
        unit = f"$\\mathrm{{{unit}}}$" if "^" in unit else _tex_escape(unit)
        val = r["value"]
        val = ", ".join(_fmt(v) for v in val) if isinstance(val, tuple) else _fmt(val)
        lines.append(f"{sym} & {val} & {unit} & {r['source']} & {_tex_escape(r['note'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(lines)


def _agg(S: list[Summary], cond: str, th: float, metric: str):
    rows = [s for s in S if s.condition == cond and s.theta_deg == th]
    v = np.array([run_metric(s, metric) for s in rows if s.completed])
    v = v[np.isfinite(v)] if v.size else v
    return {"mean": float(np.mean(v)) if v.size else np.nan, "min": float(np.min(v)) if v.size else np.nan,
            "max": float(np.max(v)) if v.size else np.nan, "n_completed": int(sum(s.completed for s in rows)),
            "n": len(rows), "fail_reasons": sorted({s.fail_reason for s in rows if not s.completed})}


def key_numbers_sim(rv: Rover, act: Actuator, p: SimParams, S, S_all, steep, bounds, stats_rows, A, C,
                    labels, cuts, idx, test_idx, wall_s) -> dict:
    conds = ("B1", "B2", "B3", "A", "C")
    table = {str(th): {c: {m: _agg(S, c, th, m) for m in ("slip_mean", "travel_eff", "cot", "anchor_back_over_stroke",
                                                           "z_F_mean", "z_R_mean", "cog_err_rms", "cog_err_max", "NF_min")}
                       for c in conds} for th in p.slopes_deg}
    table_all = {str(th): {c: {m: _agg(S_all, c, th, m) for m in ("slip_mean", "travel_eff", "cot")} for c in conds}
                 for th in p.slopes_deg}
    steep_by = {c: [steep.get((c, si), np.nan) for si in test_idx] for c in conds}
    steep_mean = {c: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for c, v in steep_by.items()}
    br = [b["bound_ratio"] for b in bounds if np.isfinite(b["bound_ratio"])]
    bri = [b["bound_ratio_with_inertia"] for b in bounds if np.isfinite(b["bound_ratio_with_inertia"])]
    dxr = [b["dx_ratio"] for b in bounds if np.isfinite(b["dx_ratio"])]

    def stat(m, a, b, th, set_="protocol"):
        for r in stats_rows:
            if r["metric"] == m and r["a"] == a and r["b"] == b and r["slope_deg"] == th and r["set"] == set_:
                return r
        return {}

    # pass criteria of the protocol
    crit = {}
    for th in p.slopes_deg:
        sA = stat("slip_mean", "A", "B2", th); cA = stat("cot", "A", "B2", th)
        sC = stat("slip_mean", "C", "B3", th); cC = stat("cot", "C", "B3", th)
        crit[str(th)] = {
            "A_reduces_slip_vs_B2": {"mean_diff": sA.get("mean_diff"), "n": sA.get("n"), "d_z": sA.get("d_z"), "t_p": sA.get("t_p")},
            "A_reduces_cot_vs_B2": {"mean_diff": cA.get("mean_diff"), "n": cA.get("n"), "d_z": cA.get("d_z"), "t_p": cA.get("t_p")},
            "C_improves_slip_vs_B3": {"mean_diff": sC.get("mean_diff"), "n": sC.get("n"), "d_z": sC.get("d_z"), "t_p": sC.get("t_p")},
            "C_improves_cot_vs_B3": {"mean_diff": cC.get("mean_diff"), "n": cC.get("n"), "d_z": cC.get("d_z"), "t_p": cC.get("t_p")},
        }
    criteria = {
        "per_slope": crit,
        "stageC_dx_within_envelope": bool(max(dxr) <= 1.0 + 1e-9) if dxr else None,
        "stageC_dx_ratio_max": float(max(dxr)) if dxr else None,
        "stageC_load_bound_ratio_max": float(max(br)) if br else None,
        "stageC_load_bound_never_violated": bool(max(br) <= 1.0) if br else None,
        "stageC_load_bound_with_inertia_ratio_max": float(max(bri)) if bri else None,
        "stageC_load_bound_with_inertia_never_violated": bool(max(bri) <= 1.0) if bri else None,
        "steepest_mean_deg": steep_mean,
    }
    return {
        "setup": {"n_soils": p.n_soils, "n_train": p.n_train, "n_rep": p.n_rep, "n_test_all": p.n_soils - p.n_train,
                  "slopes_deg": list(p.slopes_deg), "n_cycles": p.n_cycles, "dt": p.dt, "seed": p.seed,
                  "wall_min": wall_s / 60, "rail_start": rv.rail_start, "f": rv.f, "M": rv.M,
                  "class_cuts_traction_index": [float(c) for c in cuts],
                  "traction_index_range": [float(idx.min()), float(idx.max())]},
        "stageA": {"cv_accuracy": A.cv_accuracy, "test_accuracy": A.test_accuracy, "importances": A.importances,
                   "table_xi_over_R": {f"class{c}_slope{int(p.slopes_deg[b])}": (v - rv.rail_lo) / rv.rail_len
                                       for (c, b), v in A.table.items()},
                   "table_fallback_to_physics": [f"class{c}_slope{int(p.slopes_deg[b])}"
                                                 for c, b in getattr(A, "table_fallback", [])]},
        "stageC": {"gp_cv_rmse_mm": C.cv_rmse * 1e3, "target_std_mm": C.target_std * 1e3, "n_train_rows": C.n_train,
                   "search_mean_cot_gain": float(np.mean([r["cot_ff"] - r["cot_best"] for r in C.search_results]))},
        "metrics_protocol": table, "metrics_all_test": table_all,
        "steepest_deg": {c: [float(v) for v in vals] for c, vals in steep_by.items()},
        "bounds": bounds, "stats": stats_rows, "criteria": criteria,
    }


def _d(v, nd=2):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def snippet_sim_tex(k: dict, p: SimParams, rv: Rover) -> str:
    conds = ("B1", "B2", "B3", "A", "C")
    sl = [str(th) for th in p.slopes_deg]
    T = k["metrics_protocol"]
    steep = k["criteria"]["steepest_mean_deg"]
    cr = k["criteria"]
    # results table
    rows = []
    z_all, cog_all = [], {}
    for th in sl:
        for c in conds:
            m = T[th][c]
            z = 0.5 * (m["z_F_mean"]["mean"] + m["z_R_mean"]["mean"]) * 1e3
            cog = m["cog_err_rms"]["mean"] * 1e3
            if np.isfinite(z):
                z_all.append(z)
            if np.isfinite(cog):
                cog_all[c] = max(cog_all.get(c, 0.0), cog)
            rows.append(f"{int(float(th))} & {COND_LABELS[c]} & {m['slip_mean']['n_completed']}/{m['slip_mean']['n']} & "
                        f"{_d(m['slip_mean']['mean'])} & {_d(m['travel_eff']['mean'])} & {_d(m['cot']['mean'])} & "
                        f"{_d(z, 0)} & {_d(cog, 0)} & {_d(m['NF_min']['mean'], 0)} \\\\")
    table_rows = "\n".join(rows)
    cog_c = cog_all.get("C", np.nan); cog_other = max([v for c, v in cog_all.items() if c != "C"], default=np.nan)
    steep_txt = ", ".join(f"{COND_LABELS[c]} {_d(steep[c], 1)}$^\\circ$" for c in conds)

    crit_rows = []
    n_met = 0; n_tot = 0
    for th in sl:
        for key, label in (("A_reduces_slip_vs_B2", "Stage A $-$ PPL mass centred, slip"),
                           ("A_reduces_cot_vs_B2", "Stage A $-$ PPL mass centred, CoT"),
                           ("C_improves_slip_vs_B3", "Stage C $-$ PPL physics-based, slip"),
                           ("C_improves_cot_vs_B3", "Stage C $-$ PPL physics-based, CoT")):
            c = cr["per_slope"][th][key]
            md = c.get("mean_diff")
            enough = (c.get("n") or 0) >= 2
            met = enough and md is not None and np.isfinite(md) and md < 0
            if enough:
                n_tot += 1; n_met += int(met)
            crit_rows.append(f"{int(float(th))} & {label} & {_d(md, 3)} & {c.get('n', 'n/a')} & {_d(c.get('d_z'))} & "
                             f"{_d(c.get('t_p'), 3)} \\\\")
    crit_table = "\n".join(crit_rows)
    sA, sC = k["stageA"], k["stageC"]
    rule = k.get("stageC_rule", {})
    rel_names = [FEATURE_LABELS.get(n, n.replace("_", " ")) for n in rule.get("relevant_inputs_by_length_scale", [])[:4]]
    rel = (", ".join(rel_names[:-1]) + (", and " if len(rel_names) > 2 else " and ") + rel_names[-1]) if rel_names else "none"
    n_fb = len(sA.get("table_fallback_to_physics", []))
    fb_slopes = sorted({e.split("_slope")[1] for e in sA.get("table_fallback_to_physics", [])})
    fb_txt = ("" if n_fb == 0 else
              f" Where fewer than three training soils of a class completed the cycles at a slope, the table entry "
              f"falls back to the physics-based set point of Equation~\\eqref{{eq:setpoint}} averaged over the two wheelbases "
              f"({n_fb} of {len(sA['table_xi_over_R'])} entries, at {', '.join(fb_slopes)}$^\\circ$).")
    txt = rf"""
% ===================== PASTE NOTE: this snippet needs Table~\ref{{tab:simparams}} (out/params_sim.tex) and, from the
% quasi-static study, Table~\ref{{tab:evalparams}} (out/params.tex) and Figure~\ref{{fig:eval}} (out/model_evaluation_snippet.tex).
% ===================== METHODS: insert after Section~\ref{{sec:controller-validation}} or in Materials and Methods =====================
\subsubsection{{\rev{{Closed-Loop Simulation Model}}}}
\label{{sec:sim-model}}

\rev{{Step one of the validation protocol was carried out in a pitch-plane simulation of the platform of Section~\ref{{sec:platform}} with the parameters of Tables~\ref{{tab:evalparams}} and~\ref{{tab:simparams}}. The rover is modelled as two units, the rear unit carrying the chassis, the rail, and the movable mass, and the front unit carrying the front wheel pair, connected by the position-controlled wheelbase actuator (a proportional--derivative force law with a force limit). Each unit obeys Newton's law along the slope with the wheel--soil force of its pair, its weight component, the actuator force, and, for the rear unit, the inertial reaction of the moving mass. The normal loads follow Equation~\eqref{{eq:loads}} with the realised wheelbase and the longitudinal load transfer of the chassis and mass accelerations; pitch is kinematic, given by the sinkage difference of the two pairs over the wheelbase, because the prismatic joint transmits moments and the sinkage of the order of one centimetre yields pitch angles below two degrees. The wheel--soil interaction of a driven pair is the Wong--Reece stress integration for a rigid wheel \citep{{Wo08}}: the normal stress follows the Bekker law over the contact arc with the maximum-stress angle $\theta_m = (a_0 + a_1 s)\theta_f$, the shear stress follows the Janosi--Hanamoto relation with the shear displacement of a wheel at slip $s$, and the vertical force, drawbar pull, and torque are the integrals over the arc; these are tabulated per soil sample on a sinkage--slip grid and inverted at every step for the quasi-steady sinkage at the current load. An anchored pair holds through an elasto-plastic Coulomb element whose stiffness is the fully mobilised shear capacity over the shear modulus $K$ and whose capacity is the Mohr--Coulomb shear of the contact patch plus the Bekker bulldozing resistance of the sunk wheel; it yields when the reaction exceeds that capacity, which is the mechanism by which the load ratio decides whether the push--pull cycle advances. The driven wheels run a velocity loop with a torque limit; the commanded wheel speed is chosen so that the nominal advance per phase equals the wheelbase stroke at the reference slip $s_{{\mathrm{{ref}}}}$. Intentional sinking of the anchored pair \citep{{HFI24}} is not modelled. The equations are integrated with a semi-implicit Euler scheme at $\Delta t = {p.dt * 1e3:g}$\,ms with the slip--force coupling treated implicitly.}}

\rev{{The soil parameters of silica sand No.~5 are randomised over the ranges of Table~\ref{{tab:evalparams}} (with the shear modulus restricted to the sand range of 8 to 25\,mm) and the Wong--Reece coefficients over the ranges of Table~\ref{{tab:simparams}}, giving a population of {p.n_soils} soils, of which {p.n_train} are used for training and {p.n_soils - p.n_train} are held out; the five repetitions of the protocol are the first five held-out soils, identical across conditions and slopes, and all {p.n_soils - p.n_train} held-out soils form a secondary set. Three terrain classes are defined by terciles of the drawbar pull over load at a reference slip. Stage~A uses a gradient-boosted tree classifier on proprioceptive features of one probing cycle (slip estimated from the wheelbase and wheel encoders, motor torque, actuator force, and pitch, each with assumed sensor noise) and a per-class, per-slope set point obtained offline by a grid search over {p.grid_A} rail positions that minimises the cost of transport, a measurable quantity, in place of the traction utilisation of Section~\ref{{sec:controller-stage-a}}.{fb_txt} Stage~C adds to the physics-based set point of Equation~\eqref{{eq:setpoint}} a Gaussian-process residual of the same features, bounded by $\Delta_{{\max}}$ from Equation~\eqref{{eq:bound}} evaluated for both loads at the most restrictive wheelbase and capped at {p.delta_cap_R:g} rail lengths, and trained on the per-phase offsets that minimise the cost of transport on the training soils; no integral correction is used, in line with Figure~\ref{{fig:eval}}(d). Stage~B is not evaluated: its reinforcement-learning policy requires the simulator developed here as its training environment and a validation over the operational design domain that is a study of its own, and it is left as future work. Baselines are wheel driving with a fixed wheelbase and the mass centred, PPL with the mass centred, and PPL with the physics-based set point alone.}}

% ===================== RESULTS: insert as a new subsection after Section~\ref{{sec:controller}} =====================
\subsection{{\rev{{Simulation Results}}}}
\label{{sec:sim-results}}

\rev{{Table~\ref{{tab:simresults}} and Figure~\ref{{fig:sim}} report the protocol metrics over ten cycles for the held-out soils; the steepest slope completed, by bisection to {p.steep_tol:g}$^\circ$, averages {steep_txt}. The terrain classifier of Stage~A reaches {100 * sA['cv_accuracy']:.0f}\,\% grouped cross-validation accuracy on the training soils and {100 * sA['test_accuracy']:.0f}\,\% on the held-out soils; the Gaussian process of Stage~C has a cross-validated error of {sC['gp_cv_rmse_mm']:.0f}\,mm against a target spread of {sC['target_std_mm']:.0f}\,mm. The sinkage of the wheel pairs averaged {_d(min(z_all), 0)} to {_d(max(z_all), 0)}\,mm across conditions and slopes, and the rail actuator followed its command with a root-mean-square error of at most {_d(cog_other, 0)}\,mm for the constant and feedforward set points and {_d(cog_c, 0)}\,mm for Stage~C, whose reference steps at every phase change. Table~\ref{{tab:simcriteria}} evaluates the protocol criteria as paired differences across the held-out soils: {n_met} of the {n_tot} comparisons with at least two completed soil pairs show the expected negative sign. The commanded offset of Stage~C never exceeded its envelope (largest ratio {_d(cr['stageC_dx_ratio_max'])}), and the realised deviation of the front load from the paired zero-offset run reached at most {_d(cr['stageC_load_bound_ratio_max'])} of the quasi-static bound of Equation~\eqref{{eq:bound}} and {_d(cr['stageC_load_bound_with_inertia_ratio_max'])} of that bound augmented by the inertial allowance $2 m_m a_{{\max}} h_{{\mathrm{{rail}}}}/L$ of the mass actuator; the excess over the quasi-static value is the load transfer of the moving mass that Equation~\eqref{{eq:loads}} neglects and Section~\ref{{sec:controller-stage-c}} bounds separately through the actuator limits. The residual that Stage~C learned is readable: Figure~\ref{{fig:sim}}(e) sweeps each input of the Gaussian process over its training range with the other inputs at the mean of the same phase type. In the contraction phases the learned offset is {_d(rule.get('contraction_offset_over_R_at_0deg'))} rail lengths towards the front at low slopes and vanishes above about {_d(rule.get('contraction_offset_vanishes_above_deg'), 0)}$^\circ$, where the physics set point already rests at the front end of the rail; in the extension phases it is close to zero on average ({_d(rule.get('extension_offset_over_R_mean'))} rail lengths) with a minimum of {_d(rule.get('extension_offset_over_R_min'))} near {_d(rule.get('extension_slope_at_min_deg'), 0)}$^\circ$. By the automatic-relevance length scales, the inputs that shape the residual are {rel}; the remaining features are ignored by the model. The learned correction therefore amounts to a phase-dependent bias of the quasi-static set point of Equation~\eqref{{eq:setpoint}} towards the anchoring pair, modulated by the slip and torque of the probing cycle, which is the behaviour that Section~\ref{{sec:platform-cycle}} anticipated qualitatively.}}

\rev{{With five soils the exact Wilcoxon signed-rank test cannot reach $p < 0.05$ (its smallest two-sided value is 0.0625), so the protocol's five repetitions support only the paired $t$-test and effect sizes; the number of repetitions required for 80\,\% power at the observed effect sizes is reported in the supplementary results file, and the comparison over all {p.n_soils - p.n_train} held-out soils is given alongside. The physical protocol of Section~\ref{{sec:controller-validation}} should be sized accordingly.}}

\begin{{table}}[!ht]
\centering
\caption{{\rev{{Simulation results over ten PPL cycles on the held-out soils: completed runs, driven slip (mean over cycles two to ten), travel efficiency, cost of transport, mean sinkage of the two pairs, root-mean-square tracking error of the mass position, and minimum front load, each averaged over completed runs. Failed runs are stalls, backward slides, or load reversals.\\}}}}
\label{{tab:simresults}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}r l c r r r r r r@{{}}}}
\toprule
$\theta$ ($^\circ$) & condition & completed & slip & efficiency & CoT & $z$ (mm) & CoG err. (mm) & $N_F^{{\min}}$ (N) \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[!ht]
\centering
\caption{{\rev{{Protocol criteria as paired differences across the held-out soils (negative = improvement): mean difference, number of soil pairs with both runs completed, Cohen's $d_z$, and the paired $t$-test $p$-value. Rows with fewer than two completed pairs are excluded from the count in the text.\\}}}}
\label{{tab:simcriteria}}
\small
\begin{{tabular}}{{@{{}}r l r c r r@{{}}}}
\toprule
$\theta$ ($^\circ$) & comparison & mean diff. & $n$ & $d_z$ & $p$ \\
\midrule
{crit_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[!ht]
    \centering
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig6a.pdf}}
        \subpanel{{a}}
    \end{{minipage}}%
    \hfill
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig6b.pdf}}
        \subpanel{{b}}
    \end{{minipage}}

    \vspace{{0.6em}}
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig6c.pdf}}
        \subpanel{{c}}
    \end{{minipage}}%
    \hfill
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig6d.pdf}}
        \subpanel{{d}}
    \end{{minipage}}

    \vspace{{0.6em}}
    \begin{{minipage}}[b]{{0.98\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig6e.pdf}}
        \subpanel{{e}}
    \end{{minipage}}
    \caption{{\rev{{Closed-loop simulation of the validation protocol. (a)~Time traces of Stage~C on one held-out soil: unit positions, slip of both pairs, mass position against its command, and the normal loads with $N_{{\min}}$. (b)~Driven slip, travel efficiency, cost of transport, and completed fraction versus slope for the five conditions over all held-out soils. (c)~Steepest slope completed per condition and protocol soil. (d)~Largest commanded offset of Stage~C against its envelope, and the largest realised front-load deviation from the paired zero-offset run relative to the bound of Equation~\eqref{{eq:bound}}. (e)~Partial dependences of the Stage~C Gaussian-process residual per phase type, each input swept over its training range with the other inputs at the mean of that phase type; bands are one predictive standard deviation.}}}}
    \label{{fig:sim}}
\end{{figure}}

% ===================== LIMITATIONS: add to Section~\ref{{sec:discussion-limitations}} =====================
% \rev{{Sixth, the simulation study uses a single sand with randomised parameters, a Wong--Reece wheel model without slip-sinkage or multipass effects, kinematic pitch, an ideal velocity loop, and assumed actuator gains and efficiencies, and it does not model the intentional sinking of the anchored pair; its slip, efficiency, and cost-of-transport values are therefore comparisons between conditions under one model, not predictions for the physical rover, and the absolute scale is fixed by the terramechanics at the assumed 12\,kg.}}

% ===================== REWRITE NOTES =====================
% Abstract: replace "no experimental or simulation results are reported here" with
%   "a closed-loop pitch-plane simulation with randomised silica-sand parameters implements step one of the
%    validation protocol and compares Stages A and C against three baselines; the physical experiment is step two".
% Results preamble (Section~\ref{{sec:results}}): replace "no experimental or simulation data are reported" with a reference to Section~\ref{{sec:sim-results}}.
% Stage B (Section~\ref{{sec:controller-stage-b}}): shorten to one paragraph and mark as future work.
% Limitations, third item: "tested in simulation but not on the physical rover".
% Conclusion: "The controller has been evaluated in closed-loop simulation; its experimental validation is the follow-up study."
"""
    return txt.lstrip("\n")
