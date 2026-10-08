"""Entry point: run the sweeps, draw the panels, write tables, numbers, and the LaTeX snippet."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from . import plotting, report, sweeps
from .params import Actuator, Rover, Soil, Study

CODE_DIR = Path(__file__).resolve().parents[1]
OUT = CODE_DIR / "out"
DOCS = CODE_DIR / "docs"
PAPER_FIGS = CODE_DIR.parent / "paper" / "figures"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rv, soil, act, st = Rover(), Soil(), Actuator(), Study()
    rng = np.random.default_rng(st.seed)

    a = sweeps.fig_a_data(rv, st)
    b = sweeps.fig_b_data(rv, st)
    c = sweeps.fig_c_data(rv, soil, st, rng)
    d = sweeps.fig_d_data(rv, act, st)
    k = sweeps.key_numbers(rv, act, st, a, b, c, d)

    plotting.setup_style()
    plotting.plot_a(a, OUT / "fig5a")
    plotting.plot_b(b, OUT / "fig5b")
    plotting.plot_c(c, OUT / "fig5c", st.theta_ref_deg)
    plotting.plot_d(d, OUT / "fig5d", rv.with_f(st.f_ref), act.T_phase)

    (OUT / "params.md").write_text(report.param_table_md(rv, soil, act, st))
    (OUT / "params.tex").write_text(report.param_table_tex(rv, soil, act, st))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "parameters_model_evaluation.md").write_text(report.param_doc_md(rv, soil, act, st))
    report.write_json(k, OUT / "results.json")
    (OUT / "model_evaluation_snippet.tex").write_text(report.snippet_tex(k, st))

    if PAPER_FIGS.is_dir():
        for p in ("fig5a", "fig5b", "fig5c", "fig5d"):
            shutil.copy(OUT / f"{p}.pdf", PAPER_FIGS / f"{p}.pdf")

    lim = k["slope_limit_deg"]
    print("balanced slope range (deg) per f:",
          {f: (round(v["balanced_lo"], 1), round(v["balanced_hi"], 1)) for f, v in lim.items()})
    print("sim:", {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in k["sim"].items()
                   if kk in ("theta_deg", "Delta_max_over_R")},
          "viol", {kk: round(vv, 2) for kk, vv in k["sim"]["viol_ratio"].items()})
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
