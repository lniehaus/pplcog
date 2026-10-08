"""Figures of the simulation study: single-wheel check, time traces, metrics, steepest slope, envelope."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..plotting import GRID, INK, INK2, PANEL_W, PHASE_BG, _save, setup_style
from ..params import Rover
from .experiment import Summary, run_metric
from .wheel_soil import WheelTable

COND_COLOURS = {"B1": "#52514e", "B2": "#2a78d6", "B3": "#eb6834", "A": "#1baf7a", "C": "#eda100"}
COND_LABELS = {"B1": "wheel driving", "B2": "PPL, mass centred", "B3": "PPL, physics-based set point",
               "A": "PPL, Stage A", "C": "PPL, Stage C"}
COND_MARK = {"B1": "x", "B2": "o", "B3": "s", "A": "^", "C": "D"}
# legend above the axes, so that it never covers the data
ABOVE = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handlelength=1.4, columnspacing=0.8, borderaxespad=0.0)


def plot_single_wheel(tab: WheelTable, rv: Rover, stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(2 * PANEL_W, 2.0))
    W = rv.M * rv.g / 4
    s = np.linspace(-0.5, 1.0, 61)
    out = np.array([tab.lookup(W, si) for si in s])
    axes[0].plot(s, out[:, 1], color="#2a78d6"); axes[0].axhline(0, color=INK2, lw=0.6)
    axes[0].set_xlabel("slip s"); axes[0].set_ylabel("drawbar pull (N)")
    axes[1].plot(s, out[:, 2], color="#eb6834"); axes[1].axhline(0, color=INK2, lw=0.6)
    axes[1].set_xlabel("slip s"); axes[1].set_ylabel("torque (N m)")
    Ws = np.linspace(5, 80, 40)
    axes[2].plot(Ws, [tab.sinkage(w, 0.15) * 1e3 for w in Ws], color="#1baf7a")
    axes[2].axvline(W, color=INK2, lw=0.6, ls=":")
    axes[2].set_xlabel("wheel load (N)"); axes[2].set_ylabel("sinkage (mm)")
    for ax in axes:
        ax.set_title("")
    fig.suptitle("single wheel, nominal silica sand parameters, 12 kg rover", fontsize=7, color=INK2)
    _save(fig, stem)


def plot_traces(s: Summary, rv: Rover, T_phase: float, stem: Path, n_show_cycles: int = 4) -> None:
    """Four stacked traces; the first n_show_cycles of the run are shown (four phases each)."""
    R = s.rec
    t = R["t"]
    fig, axes = plt.subplots(4, 1, figsize=(PANEL_W, 4.9), sharex=True)
    for ax in axes:
        for k in range(int(t[-1] // T_phase) + 1):
            if k % 2 == 0:
                ax.axvspan(k * T_phase, (k + 1) * T_phase, color=PHASE_BG, zorder=0, lw=0)
    ax = axes[0]
    ax.plot(t, R["x_R"], color="#2a78d6", label="rear unit")
    ax.plot(t, R["x_F"], color="#eb6834", label="front unit")
    ax.set_ylabel("position (m)"); ax.legend(**ABOVE)
    ax = axes[1]
    ax.plot(t, np.clip(R["s_R"], -1, 1), color="#2a78d6", lw=0.8, label="rear slip")
    ax.plot(t, np.clip(R["s_F"], -1, 1), color="#eb6834", lw=0.8, label="front slip")
    ax.set_ylabel("slip"); ax.set_ylim(-1.05, 1.05); ax.legend(**ABOVE)
    ax = axes[2]
    u = (R["xi_m"] - rv.rail_lo) / rv.rail_len
    ur = (R["xi_ref"] - rv.rail_lo) / rv.rail_len
    ax.plot(t, ur, color=INK2, lw=0.8, ls=":", label="commanded")
    ax.plot(t, u, color="#1baf7a", label="mass position")
    ax.set_ylabel(r"$\xi_m / R$"); ax.set_ylim(-0.05, 1.05); ax.legend(**ABOVE)
    ax = axes[3]
    ax.plot(t, R["N_F"], color="#eb6834", label="$N_F$")
    ax.plot(t, R["N_R"], color="#2a78d6", label="$N_R$")
    ax.axhline(rv.N_min, color=INK2, lw=0.6, ls="--")
    ax.set_ylabel("load (N)"); ax.set_xlabel("time (s)"); ax.legend(**ABOVE)
    ax.set_xlim(0.0, n_show_cycles * 4 * T_phase)
    fig.suptitle(f"{COND_LABELS[s.condition]}, θ = {s.theta_deg:g}°, test soil {s.soil}", fontsize=7, color=INK2)
    _save(fig, stem)


def plot_metrics_vs_slope(summaries: list[Summary], slopes, stem: Path, conditions=("B1", "B2", "B3", "A", "C")) -> None:
    metrics = [("slip_mean", "driven slip (mean)"), ("travel_eff", "travel efficiency"),
               ("cot", "cost of transport"), ("completed", "runs completed")]
    fig, axes = plt.subplots(2, 2, figsize=(PANEL_W, 3.9), sharex=True)
    for ax, (m, lab) in zip(axes.ravel(), metrics):
        for cond in conditions:
            ys, lo, hi = [], [], []
            for th in slopes:
                rows = [s for s in summaries if s.condition == cond and s.theta_deg == th]
                if m == "completed":
                    v = np.array([float(s.completed) for s in rows])
                else:
                    v = np.array([run_metric(s, m) for s in rows if s.completed])
                v = v[np.isfinite(v)] if v.size else v
                if v.size == 0:
                    ys.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                ys.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
            ys, lo, hi = map(np.array, (ys, lo, hi))
            ax.plot(slopes, ys, color=COND_COLOURS[cond], marker=COND_MARK[cond], ms=3.5, lw=1.2, label=COND_LABELS[cond])
            ax.fill_between(slopes, lo, hi, color=COND_COLOURS[cond], alpha=0.12, lw=0)
        ax.set_ylabel(lab)
        if m == "cot":
            ax.set_yscale("log")
    axes[1, 0].set_xlabel(r"slope $\theta$ (deg)"); axes[1, 1].set_xlabel(r"slope $\theta$ (deg)")
    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=6, handlelength=1.6, columnspacing=1.2,
                     title="mean over held-out soils, band = min to max, completed runs only;\n"
                           "a series ends at the last slope with a completed run", title_fontsize=6)
    leg.get_title().set_color(INK2)
    _save(fig, stem)


def plot_steepest(steep: dict, soils_idx, stem: Path, conditions=("B1", "B2", "B3", "A", "C")) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W, 2.2))
    x = np.arange(len(conditions))
    for i, cond in enumerate(conditions):
        v = np.array([steep.get((cond, si), np.nan) for si in soils_idx])
        ax.scatter(np.full(v.size, i) + np.linspace(-0.18, 0.18, v.size), v, color=COND_COLOURS[cond], s=14,
                   zorder=3, edgecolor="white", lw=0.5)
        if np.isfinite(v).any():
            ax.hlines(np.nanmean(v), i - 0.3, i + 0.3, color=COND_COLOURS[cond], lw=2)
    short = {"B1": "wheel\ndriving", "B2": "PPL\ncentred", "B3": "PPL\nphysics-based", "A": "PPL\nStage A", "C": "PPL\nStage C"}
    ax.set_xticks(x); ax.set_xticklabels([short[c] for c in conditions], fontsize=6.5)
    ax.set_ylabel("steepest completed slope (deg)")
    ax.set_ylim(0, 36)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))  # strip for the footnote under the tick labels
    fig.text(0.01, 0.005, "points: protocol soils; bar: mean", fontsize=6, color=INK2)
    _save(fig, stem)


def plot_offset_envelope(summaries: list[Summary], slopes, rv: Rover, stem: Path) -> None:
    """Stage C: commanded offset and envelope per slope, and the realised load-bound ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(PANEL_W, 2.2))
    ax = axes[0]
    for th in slopes:
        rows = [s for s in summaries if s.condition == "C" and s.theta_deg == th]
        Dm = np.array([s.Delta_max for s in rows if s.Delta_max is not None]) / rv.rail_len
        dx = np.array([s.dx_max_abs for s in rows if s.dx_max_abs is not None]) / rv.rail_len
        ax.scatter(np.full(dx.size, th) + np.linspace(-1, 1, dx.size), dx, color="#eda100", s=12, zorder=3, label="max |Δx|" if th == slopes[0] else None)
        ax.plot([th - 1.5, th + 1.5], [np.mean(Dm)] * 2, color=INK, lw=1.5, label=r"$\Delta_{\max}$" if th == slopes[0] else None)
    ax.set_xlabel(r"slope $\theta$ (deg)"); ax.set_ylabel("offset / R"); ax.legend(**ABOVE)
    ax.set_ylim(0, None)
    ax = axes[1]
    for th in slopes:
        rows = [s for s in summaries if s.condition == "C" and s.theta_deg == th]
        ratio = np.array([s.bound_ratio for s in rows if getattr(s, "bound_ratio", None) is not None])
        ratio_i = np.array([s.bound_ratio_i for s in rows if getattr(s, "bound_ratio_i", None) is not None])
        if ratio.size:
            ax.scatter(np.full(ratio.size, th) + np.linspace(-1, 1, ratio.size), ratio, color="#eda100", s=12, zorder=3,
                       label="Eq. (5)" if th == slopes[0] else None)
        if ratio_i.size:
            ax.scatter(np.full(ratio_i.size, th) + np.linspace(-1, 1, ratio_i.size), ratio_i, color="#2a78d6", s=12,
                       zorder=3, marker="s", label="Eq. (5) + inertial" if th == slopes[0] else None)
    ax.legend(**ABOVE)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel(r"slope $\theta$ (deg)"); ax.set_ylabel("load deviation / bound, max")
    ax.set_ylim(0, None)
    _save(fig, stem)


GP_LABELS = {"slope_deg": "slope (deg)", "slip_est_mean": "slip est. mean", "slip_est_max": "slip est. p95",
             "torque_mean": "torque mean (N m)", "torque_peak": "torque p95 (N m)",
             "force_mean": "actuator force mean (N)", "force_peak": "actuator force p95 (N)",
             "pitch_odd": "pitch, contraction (rad)", "pitch_even": "pitch, extension (rad)"}


def plot_gp_rule(pd: dict, rv: Rover, stem: Path) -> None:
    """Partial dependences of the Stage C Gaussian process per phase type, with predictive std."""
    names = pd["names"]
    fig, axes = plt.subplots(2, 5, figsize=(2 * PANEL_W, 3.1), sharey=True)
    R = rv.rail_len
    for ax, name in zip(axes.ravel()[: len(names)], names):
        g = pd["grid"][name]
        for par, ls, col in ((1, "-", "#eb6834"), (0, "--", "#2a78d6")):
            m, sd = pd["mean"][name][par] / R, pd["std"][name][par] / R
            ax.fill_between(g, m - sd, m + sd, color=col, alpha=0.12, lw=0)
            ax.plot(g, m, color=col, ls=ls, lw=1.5, label="contraction phases" if par == 1 else "extension phases")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xlabel(GP_LABELS.get(name, name), fontsize=6.5)
        ax.tick_params(labelsize=6)
    axes.ravel()[-1].axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, fontsize=6.5, loc="center", title="band: predictive std", title_fontsize=6)
    axes[0, 0].set_ylabel(r"learned offset $\Delta x / R$"); axes[1, 0].set_ylabel(r"learned offset $\Delta x / R$")
    fig.suptitle("Stage C residual: each input swept over its training range, other inputs at the mean of that phase type",
                 fontsize=6.5, color=INK2)
    _save(fig, stem)
