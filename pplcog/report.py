"""Parameter table (Markdown and LaTeX), key numbers (JSON), and a draft LaTeX snippet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .params import Actuator, Rover, Soil, Study, provenance_rows


def _fmt(v) -> str:
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1e4 or abs(v) < 1e-3:
            return f"{v:.3g}"
        return f"{v:.4g}"
    return str(v)


def _tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("+-", r"$\pm$")


def param_table_md(rv: Rover, soil: Soil, act: Actuator, st: Study) -> str:
    rows = provenance_rows(rv, soil, act, st)
    out = ["| Parameter | Value | Unit | Source | Note |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['symbol']} | {_fmt(r['value'])} | {r['unit']} | {r['source']} | {r['note']} |")
    return "\n".join(out) + "\n"


PARAM_DOC_LEGEND = """# Parameters of the model evaluation

Every parameter of the quasi-static model evaluation (Section 3.1.2 and Figure 4 of the
manuscript *Adaptive Center of Gravity Control for Push-Pull Locomotion Rovers on Slopes*)
with its value, unit, source, and note. This table was Table 3 of the first submission; the revised manuscript
refers to this file from its Data Availability Statement. It is written by `pplcog/run_all.py` from
the defaults in `pplcog/params.py`; edit the defaults, not this file.

Sources:

- *photo* = read from the labelled scale of Figure 1(a) of the manuscript;
- *sibling* = published rovers of the same laboratory (Higa, Fujiwara and Iizuka 2024, HFI24;
  Sugimoto, Fujiwara and Iizuka 2025, SFI25);
- *literature* = silica sand No. 5 properties (SFI25), Bekker constants for dry sand
  (Wong 2008, *Theory of Ground Vehicles*, Wo08), and the rigid-wheel model of
  Wong and Reece (1967, WR67);
- *assumed* = not available for the present build;
- *manuscript* = design choice stated in the text;
- *study* = design of this evaluation.

All photo and assumed values are estimates with an uncertainty of about +-20 %.

"""


def param_doc_md(rv: Rover, soil: Soil, act: Actuator, st: Study) -> str:
    """Tracked documentation file: legend of Table 3 of the first submission plus the table."""
    return PARAM_DOC_LEGEND + param_table_md(rv, soil, act, st)


def param_table_tex(rv: Rover, soil: Soil, act: Actuator, st: Study) -> str:
    rows = provenance_rows(rv, soil, act, st)
    lines = [
        r"\begin{table}[!ht]", r"\centering",
        r"\caption{Parameters of the model evaluation. Sources: photo = read from the labelled scale of "
        r"Figure~\ref{fig:testbed}(a); sibling = published rovers of the same laboratory \citep{HFI24,SFI25}; "
        r"literature = silica sand No.~5 properties \citep{SFI25}, Bekker constants for dry sand \citep{Wo08}, and the rigid-wheel model of \citet{WR67}; "
        r"assumed = not available for the present build; manuscript = design choice stated in the text; "
        r"study = design of this evaluation. All photo and assumed values are estimates with an uncertainty "
        r"of about $\pm 20\,\%$.\\}",
        r"\label{tab:evalparams}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{@{}l r l l X@{}}", r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} & \textbf{Unit} & \textbf{Source} & \textbf{Note} \\", r"\midrule",
    ]
    for r in rows:
        sym = r["symbol"]
        sym = f"${sym}$" if any(ch in sym for ch in "\\_^{}") else sym
        unit = r["unit"]
        unit = f"$\\mathrm{{{unit}}}$" if "^" in unit else _tex_escape(unit)
        lines.append(f"{sym} & {_fmt(r['value'])} & {unit} & {r['source']} & {_tex_escape(r['note'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(type(o))


def write_json(numbers: dict, path: Path) -> None:
    path.write_text(json.dumps(numbers, indent=1, default=_json_default, allow_nan=True) + "\n")


def snippet_tex(k: dict, st: Study) -> str:
    """Draft subsubsection for the manuscript. Written to out/, never inserted automatically."""
    f_list = [str(f) for f in st.f_list]
    lim = k["slope_limit_deg"]
    hi = {f: lim[f]["balanced_hi"] for f in f_list}
    lo = {f: lim[f]["balanced_lo"] for f in f_list}
    sens_h = k["slope_limit_sens_h_rail_over_L"]
    sens_r = k["slope_limit_sens_rail_start_m"]
    env = k["envelope_over_R_at_theta_ref"]
    envr = k["envelope_rail_limited_over_R_at_theta_ref"]
    geom = k["geom_shift_p95_over_R_at_theta_ref"]
    sim = k["sim"]
    d = lambda v: f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
    txt = rf"""
% ===================== PASTE NOTE: insert after Section~\ref{{sec:controller-model}}. The parameter table is
% docs/parameters_model_evaluation.md (out/params.tex is its LaTeX twin). The simulation snippet (out/simulation_snippet.tex)
% refers to Figure~\ref{{fig:eval}}.
\subsubsection{{\rev{{Numerical Evaluation of the Quasi-Static Model}}}}
\label{{sec:controller-evaluation}}

\rev{{The quasi-static model of Section~\ref{{sec:controller-model}} and the envelope rule of Equation~\eqref{{eq:bound}} were evaluated numerically for the platform of Section~\ref{{sec:platform}}. Because the masses, the actuator limits, and the phase duration of the current build are not yet measured, the evaluation is parametric: all lengths are normalised by the wheelbase, all loads by the rover weight, and the movable mass fraction $f = m_m/M$ is swept over {', '.join(f_list[:-1])}, and {f_list[-1]}. The remaining geometry is taken from the labelled scale of Figure~\ref{{fig:testbed}}(a), from the published sibling rovers of the same laboratory \citep{{HFI24,SFI25}}, and from stated assumptions; the parameter table of the code repository (Data Availability Statement) lists every value with its source. The wheel--soil relations use the published properties of silica sand No.~5 \citep{{SFI25}} and, for the Bekker constants that are not available for this sand, the dry-sand values of \citet{{Wo08}}. The evaluation is a model study with estimated parameters and does not replace the experimental protocol of Section~\ref{{sec:controller-validation}}. The code that produces Figure~\ref{{fig:eval}} is available on GitHub (Data Availability Statement).}}

\rev{{Three modelling decisions that Equations~\eqref{{eq:cog}} to~\eqref{{eq:loads}} leave open were fixed as follows. First, the chassis and the rail are attached to the rear wheel pair and the front pair slides, so $x_b$ and $x_m$ are constant in the rear-contact frame while $L$ varies with the phase; the chassis mass is split into a body at fixed position and two wheel-pair masses at the contact points, so $x_b(\varphi)$ follows from the wheelbase. With the rail attached to the front pair instead, the sign of $\partial N_F/\partial L$ at fixed mass position reverses, so the statement after Equation~\eqref{{eq:loads}} that a wheelbase extension shifts load towards the down-slope pair holds for the rear-mounted rail only. Second, the rail is parallel to the chassis axis, so the CoG height $h$ is independent of $x_m$; it depends on $f$ through the rail height and lies between {k['h_cog_over_L'][f_list[0]]:.2f}\,$L$ and {k['h_cog_over_L'][f_list[-1]]:.2f}\,$L$ for the values considered. Third, both $N_F \ge N_{{\min}}$ and $N_R \ge N_{{\min}}$ are enforced, whereas Section~\ref{{sec:controller-model}} states only the former.}}

\rev{{Figure~\ref{{fig:eval}}(a) shows the load ratio $\rho$ over the rail coordinate and the slope angle for the contracted and the extended wheelbase. Figure~\ref{{fig:eval}}(b) gives the range of slopes over which the target $\rho^\ast = {st.rho_star:g}$ is reachable within the rail with both loads above $N_{{\min}}$. The range is bounded by the extended wheelbase and reaches {d(hi[f_list[0]])}$^\circ$ for $f = {f_list[0]}$ and {d(hi[f_list[-1]])}$^\circ$ for $f = {f_list[-1]}$; at the contracted wheelbase the limits are {d(lim[f_list[0]]['Lmin']['hi'])}$^\circ$ and {d(lim[f_list[-1]]['Lmin']['hi'])}$^\circ$. The limit is sensitive to the rail height, moving by about {d(sens_h['0.5'][f_list[-1]] - sens_h['0.9'][f_list[-1]])}$^\circ$ between $h_{{\mathrm{{rail}}}} = 0.5\,L$ and $0.9\,L$ at $f = {f_list[-1]}$, and to the rail position along the chassis, from {d(sens_r['0.05'][f_list[-1]])}$^\circ$ with the rail starting at the rear axle to {d(sens_r['0.182'][f_list[-1]])}$^\circ$ with the rail ending at the front axle. Above these slopes the mass rests at the up-slope end of the rail and the controller degenerates to a fixed set point; exact load balancing is therefore a low-slope objective on this platform, and the steeper slopes that PPL rovers are known to climb \citep{{FHI25}} must be handled by the anchoring mechanism rather than by the movable mass alone.}}

\rev{{Figure~\ref{{fig:eval}}(c) compares three quantities per mass fraction: the envelope $\Delta_{{\max}}$ permitted by Equation~\eqref{{eq:bound}}, the envelope actually realisable within the rail around the nominal set point, and the 95th percentile of the set-point shift caused by an uncertainty of $\pm 20\,\%$ in the rail height, $\pm 0.03\,L$ in the chassis CoG, and a factor of two in the wheel-pair mass. At $\theta = {st.theta_ref_deg:g}^\circ$ the load-based rule permits between {min(env.values()):.1f} and {max(env.values()):.1f} rail lengths, more than the rail itself, so it is never the binding constraint; the rail is, and once the nominal set point saturates at the rail end (Figure~\ref{{fig:eval}}(b)) no room is left for the learned offset at all. The geometric uncertainty shifts the set point by up to {geom[f_list[0]]:.2f} rail lengths for $f = {f_list[0]}$ and {geom[f_list[-1]]:.2f} for $f = {f_list[-1]}$, which the rail room absorbs at low slopes but not once the nominal set point saturates. A Monte Carlo sweep of the soil parameters over their published ranges, by contrast, does not move the set point that minimises the traction utilisation: with identical wheel pairs, that optimum coincides with $\rho = 1$ independently of $k_c$, $k_\phi$, $n$, $c$, $\phi_s$, and $K$ (spread below $10^{{-5}}$ rail lengths). In the quasi-static model the soil therefore decides whether a slope can be climbed, not where the mass should be placed, and the learning component of Stages~B and~C must address geometric and payload uncertainty and the dynamic and anchoring effects that the model omits.}}

\rev{{Figure~\ref{{fig:eval}}(d) simulates the actuator loop of Equation~\eqref{{eq:actuator}} over {st.n_cycles} PPL cycles at $\theta = {sim['theta_deg']:.1f}^\circ$ and $f = {sim['f']}$ with a worst-case learned offset of $\pm\Delta_{{\max}}$ that changes sign every phase, for three controller variants. With the feedforward set point alone, the deviation of $N_F$ from its nominal trajectory stays within the bound of Equation~\eqref{{eq:bound}} (ratio {sim['viol_ratio']['ff']:.2f}), as the input-to-state argument predicts. When the proportional--integral correction on the load ratio of Section~\ref{{sec:controller-stage-c}} is added, the bound is exceeded transiently after every sign change (ratio {sim['viol_ratio']['ff_pi_after']:.2f} when the offset is added after the correction and {sim['viol_ratio']['ff_pi_blind']:.2f} when the correction is made blind to it), and the correction works against the learned offset, realising {100 * sim['offset_effect_last_cycle']['ff_pi_after']:.0f}\,\% of it in the last cycle compared with {100 * sim['offset_effect_last_cycle']['ff']:.0f}\,\% without correction; a constant offset is cancelled by the integrator almost entirely ({100 * sim['offset_effect_const_last_cycle']['ff_pi_after']:.0f}\,\% realised). The bound of Equation~\eqref{{eq:bound}} is therefore a property of the feedforward-plus-actuator loop, and the Stage~C design must either omit the integral correction or apply it to a residual from which the learned offset has been removed. The inertial moment of the moving mass neglected in Equation~\eqref{{eq:loads}} amounts to at most {100 * k['neglected_inertial_moment_over_MgL']:.0f}\,\% of $M g L$ at the assumed acceleration limit.}}

\begin{{figure}}[!ht]
    \centering
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig5a.pdf}}
        \subpanel{{a}}
    \end{{minipage}}%
    \hfill
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig5b.pdf}}
        \subpanel{{b}}
    \end{{minipage}}

    \vspace{{0.6em}}
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig5c.pdf}}
        \subpanel{{c}}
    \end{{minipage}}%
    \hfill
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{figures/fig5d.pdf}}
        \subpanel{{d}}
    \end{{minipage}}
    \caption{{\rev{{Numerical evaluation of the quasi-static model with the estimated parameters of the parameter table of the code repository. (a)~Load ratio $\rho = N_F/N_R$ over the rail coordinate and the slope angle for the contracted and the extended wheelbase, $f = {st.f_ref}$; hatched cells violate $N_{{F}}, N_{{R}} \ge N_{{\min}}$, the line marks $\rho = \rho^\ast$. (b)~Rail coordinate that yields $\rho^\ast$ versus slope angle per mass fraction, with the rail limits; the slope range over which the target is reachable is bounded by the extended wheelbase. (c)~Envelope permitted by Equation~\eqref{{eq:bound}}, envelope realisable within the rail, and 95th percentile of the set-point shift under geometric uncertainty, all in rail lengths. (d)~Actuator loop over {st.n_cycles} PPL cycles at $\theta = {sim['theta_deg']:.1f}^\circ$ with a worst-case learned offset: mass position, front load deviation from the nominal trajectory, and the bound of Equation~\eqref{{eq:bound}} for the feedforward controller and for the two proportional--integral variants. No soil or rover measurement enters this figure; it is a model evaluation, not a validation.}}}}
    \label{{fig:eval}}
\end{{figure}}

% --- Sentence for Section~\ref{{sec:discussion-limitations}} ---
% \rev{{Fifth, the numerical evaluation of Section~\ref{{sec:controller-evaluation}} uses estimated geometry and an assumed mass distribution, so its slope limits and envelopes are parametric statements about the model rather than properties of the built rover; they will be recomputed once the platform parameters are measured.}}
"""
    return txt.lstrip("\n")
