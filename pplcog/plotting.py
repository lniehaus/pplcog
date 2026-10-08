"""Figure panels (a) to (d), written as vector PDF and raster PNG.

Colour follows the entity: each mass fraction keeps its colour in every panel where it
appears; controller variants have their own fixed colours in panel (d).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# validated categorical palette (light surface), fixed order
F_COLOURS = {0.2: "#2a78d6", 0.3: "#eb6834", 0.4: "#1baf7a", 0.5: "#eda100"}
CTRL_COLOURS = {"ff": "#2a78d6", "ff_pi_after": "#eb6834", "ff_pi_blind": "#1baf7a"}
CTRL_LABELS = {"ff": "feedforward", "ff_pi_after": "feedforward + PI",
               "ff_pi_blind": "feedforward + PI blind to offset"}
GRID = "#e6e5e1"
INK = "#0b0b0b"
INK2 = "#52514e"
PHASE_BG = "#f0efec"
OFFSET_FILL = "#cfcdc8"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#f0efec", "#e34948"])

PANEL_W = 3.35  # inches, about 85 mm
# legend above the axes, so that it never covers the data
ABOVE = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, handlelength=1.4,
             columnspacing=0.8, borderaxespad=0.0)


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "legend.frameon": False, "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    })


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def plot_a(a: dict, stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(PANEL_W, 2.5), sharey=True)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    for ax, tag, title in zip(axes, ("Lmin", "Lmax"), ("contracted, $L_{\\min}$", "extended, $L_{\\max}$")):
        d = a[tag]
        z = np.ma.masked_where(d["infeasible"], d["log10rho"])
        m = ax.pcolormesh(a["u"], a["theta_deg"], z, cmap=DIVERGING, norm=norm, shading="nearest", rasterized=True)
        ax.contourf(a["u"], a["theta_deg"], d["infeasible"].astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["////"], zorder=2)
        ax.contour(a["u"], a["theta_deg"], d["infeasible"].astype(float), levels=[0.5], colors=INK2, linewidths=0.6)
        ok = (d["u_star"] >= 0) & (d["u_star"] <= 1)
        ax.plot(d["u_star"][ok], a["theta_deg"][ok], color=INK, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel(r"mass position $\xi_m / R$")
        ax.set_xlim(0, 1); ax.set_ylim(a["theta_deg"][0], a["theta_deg"][-1])
        ax.grid(False)
    axes[0].set_ylabel(r"slope $\theta$ (deg)")
    cb = fig.colorbar(m, ax=axes, shrink=0.9, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(r"$\log_{10}\rho$,  $\rho = N_F/N_R$")
    cb.outline.set_visible(False)
    handles = [Line2D([], [], color=INK, lw=1.2, label=r"$\rho = \rho^\ast$"),
               Patch(facecolor="none", edgecolor=INK2, hatch="////", lw=0.6, label=r"$N_F$ or $N_R < N_{\min}$")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, handlelength=1.8, columnspacing=1.5)
    _save(fig, stem)


def plot_b(b: dict, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W, 2.5))
    th = b["theta_deg"]
    ax.axhspan(0, 1, color=PHASE_BG, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6); ax.axhline(1, color=INK2, lw=0.6)
    for f in b["f_list"]:
        c = F_COLOURS[f]
        cur = b["curves"][f]
        ax.plot(th, cur["Lmax"], color=c, label=f"f = {f}")
        ax.plot(th, cur["Lmin"], color=c, ls="--", lw=1.0)
        hi = b["limits"][f]["balanced_hi"]
        if np.isfinite(hi):
            ax.plot([hi], [1.0], marker="o", ms=4, color=c, mec="white", mew=0.8, zorder=4)
            up = list(b["f_list"]).index(f) % 2 == 0
            ax.annotate(f"{hi:.0f}°", (hi, 1.0), xytext=(0, 5 if up else -10), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK2)
    ax.plot([], [], color=INK2, ls="-", lw=1.0, label=r"extended $L_{\max}$")
    ax.plot([], [], color=INK2, ls="--", lw=1.0, label=r"contracted $L_{\min}$")
    ax.set_xlabel(r"slope $\theta$ (deg)")
    ax.set_ylabel(r"set point $\xi_m^\ast / R$ for $\rho^\ast$")
    ax.set_xlim(th[0], th[-1]); ax.set_ylim(-0.6, 2.2)
    ax.text(th[-1], 0.5, "rail", ha="right", va="center", fontsize=6.5, color=INK2)
    ax.legend(**ABOVE)
    _save(fig, stem)


def plot_c(c: dict, stem: Path, theta_ref: float) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W, 2.5))
    th = c["theta_deg"]
    for f in c["f_list"]:
        col = F_COLOURS[f]
        ax.plot(th, c["env_rail"][f], color=col, label=f"f = {f}")
        ax.plot(th, c["geom"][f][2], color=col, ls="--", lw=1.0)
    ax.plot([], [], color=INK2, ls="-", lw=1.0, label="realisable in rail")
    ax.plot([], [], color=INK2, ls="--", lw=1.0, label="95 % set-point shift")
    i = int(np.argmin(np.abs(th - theta_ref)))
    lo = min(float(c["env"][f][i]) for f in c["f_list"]); hi = max(float(c["env"][f][i]) for f in c["f_list"])
    ax.text(0.02, 0.97, f"Eq. (5) alone permits\n{lo:.1f} to {hi:.1f} R at {theta_ref:g}° (off scale)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel(r"slope $\theta$ (deg)")
    ax.set_ylabel(r"$\Delta_{\max} / R$")
    ax.set_xlim(th[0], th[-1]); ax.set_ylim(0, 0.6)
    ax.legend(**ABOVE)
    _save(fig, stem)


def plot_d(d: dict, stem: Path, rv, T_phase: float, n_show_cycles: int = 2) -> None:
    """Three stacked traces; the first n_show_cycles of the simulated cycles are shown (four phases each)."""
    fig, axes = plt.subplots(3, 1, figsize=(PANEL_W, 3.4), sharex=True)
    runs = d["runs"]
    ref = runs["ff"]["off"]
    t = ref.t
    # phase background
    for ax in axes:
        for k in range(int(t[-1] // T_phase) + 1):
            if k % 2 == 0:
                ax.axvspan(k * T_phase, (k + 1) * T_phase, color=PHASE_BG, zorder=0, lw=0)
    # (1) mass position
    ax = axes[0]
    u = lambda x: (x - rv.rail_lo) / rv.rail_len
    ax.plot(t, u(ref.x_ref), color=INK2, lw=0.8, ls=":", label="commanded (feedforward)")
    for ctrl, r in runs.items():
        ax.plot(t, u(r["off"].x_m), color=CTRL_COLOURS[ctrl], label=CTRL_LABELS[ctrl])
    ax.plot(t, u(runs["ff"]["nom"].x_m), color=INK, lw=0.8, ls="--", label="nominal, no offset")
    ax.set_ylabel(r"$\xi_m / R$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, handlelength=1.4,
              columnspacing=0.8, borderaxespad=0.0)
    # (2) learned offset and PI action
    ax = axes[1]
    off = ref.offset / rv.rail_len
    ax.fill_between(t, off, 0.0, color=OFFSET_FILL, lw=0, zorder=1, label=r"learned offset $\Delta x$")
    ax.axhline(0, color=INK2, lw=0.5, zorder=1)
    for ctrl, ls in (("ff_pi_after", "-"), ("ff_pi_blind", "--")):
        ax.plot(t, runs[ctrl]["off"].u_pi / rv.rail_len, color=CTRL_COLOURS[ctrl], lw=1.2, ls=ls, zorder=3,
                label="PI action" + (", blind to offset" if ctrl == "ff_pi_blind" else ""))
    lim = 1.15 * float(np.max(np.abs(off)))
    ax.set_ylim(-lim, lim)
    ax.set_ylabel(r"$\Delta x$, $u_{\rm PI}$ / R")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, handlelength=1.4,
              columnspacing=0.8, borderaxespad=0.0)
    # (3) deviation over bound
    ax = axes[2]
    ax.axhspan(-1, 1, color="#dfe9f7", zorder=0, lw=0)
    ax.axhline(1, color=INK2, lw=0.6); ax.axhline(-1, color=INK2, lw=0.6)
    for ctrl, r in runs.items():
        dev = r["off"].deviation(r["nom"]) / r["off"].bound(rv, np.radians(d["theta_deg"]))
        ax.plot(t, dev, color=CTRL_COLOURS[ctrl], label=CTRL_LABELS[ctrl])
    ax.set_ylabel(r"$N_F - N_F^{\rm nom}$ / bound")
    ax.set_xlabel("time (s)")
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlim(0.0, n_show_cycles * 4 * T_phase)
    ax.text(0.99, 0.04, "band: Eq. (5); shaded: odd phases", transform=ax.transAxes, ha="right",
            fontsize=6.5, color=INK2)
    _save(fig, stem)
