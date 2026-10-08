"""Per-cycle proprioceptive features (with measurement noise) and per-cycle metrics."""

from __future__ import annotations

import numpy as np

from ..params import Rover
from .params_sim import SimParams

FEATURE_NAMES = ["slip_est_mean", "slip_est_max", "torque_mean", "torque_peak",
                 "force_mean", "force_peak", "pitch_odd", "pitch_even"]

# prose labels for the manuscript text (inputs of the learning components)
FEATURE_LABELS = {"parity": "the phase type", "slope_deg": "the slope angle",
                  "slip_est_mean": "the mean slip estimate", "slip_est_max": "the peak slip estimate",
                  "torque_mean": "the mean motor torque", "torque_peak": "the peak motor torque",
                  "force_mean": "the mean actuator force", "force_peak": "the peak actuator force",
                  "pitch_odd": "the pitch in contraction", "pitch_even": "the pitch in extension"}


def cycle_features(rec: dict, k0: int, k1: int, rv: Rover, p: SimParams, rng: np.random.Generator) -> dict:
    """Features of the cycle covering record indices [k0, k1) with sensor noise."""
    sl = slice(k0, k1)
    drv = rec["driver"][sl]             # 1 front, 0 rear, 2 both
    om_F, om_R = rec["om_F"][sl], rec["om_R"][sl]
    v_F, v_R = rec["v_F"][sl], rec["v_R"][sl]
    # wheelbase encoder gives the driven unit's advance when the anchor holds
    Ld = np.abs(v_F - v_R)
    om = np.where(drv == 1, om_F, om_R)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_est = np.where(om > 1e-6, 1.0 - Ld / (rv.r * np.maximum(om, 1e-6)), 0.0)
    s_est = np.clip(s_est, -1.0, 1.0) + rng.normal(0.0, p.noise["slip"], s_est.size)
    T = np.where(drv == 1, rec["T_F"][sl], rec["T_R"][sl])
    T = T * (1 + rng.normal(0.0, p.noise["torque_rel"], T.size)) + rng.normal(0.0, p.noise["torque_abs"], T.size)
    F = np.abs(rec["F_act"][sl]) + rng.normal(0.0, p.noise["force"], T.size)
    psi = rec["psi"][sl] + rng.normal(0.0, p.noise["pitch"], T.size)
    odd = (rec["phase"][sl] % 2) == 1
    return {
        "slip_est_mean": float(np.mean(s_est)), "slip_est_max": float(np.percentile(s_est, 95)),
        "torque_mean": float(np.mean(T)), "torque_peak": float(np.percentile(T, 95)),
        "force_mean": float(np.mean(F)), "force_peak": float(np.percentile(F, 95)),
        "pitch_odd": float(np.mean(psi[odd])) if odd.any() else 0.0,
        "pitch_even": float(np.mean(psi[~odd])) if (~odd).any() else 0.0,
    }


def cycle_metrics(rec: dict, k0: int, k1: int, E0: float, E1: float, d_nom: float, rv: Rover) -> dict:
    sl = slice(k0, k1)
    drv = rec["driver"][sl]
    s_drv = np.where(drv == 1, rec["s_F"][sl], np.where(drv == 0, rec["s_R"][sl],
                     0.5 * (rec["s_F"][sl] + rec["s_R"][sl])))
    adv = float(rec["x_R"][k1 - 1] - rec["x_R"][k0])
    anchor = rec["anchor"][sl]      # 1 front, 0 rear, -1 none
    slide = np.where(anchor == 1, rec["slide_F"][sl], np.where(anchor == 0, rec["slide_R"][sl], 0.0))
    # backward displacement of the anchored unit
    v_anc = np.where(anchor == 1, rec["v_F"][sl], np.where(anchor == 0, rec["v_R"][sl], 0.0))
    dt = rec["t"][1] - rec["t"][0]
    back = float(np.sum(np.clip(-v_anc, 0.0, None)) * dt)
    E = E1 - E0
    return {
        "advance": adv, "travel_eff": adv / d_nom if d_nom > 0 else 0.0,
        "slip_mean": float(np.mean(s_drv)), "slip_max": float(np.max(s_drv)),
        "slide_frac_time": float(np.mean(slide)), "anchor_back_over_stroke": back / rv.stroke,
        "z_F_mean": float(np.mean(rec["z_F"][sl])), "z_R_mean": float(np.mean(rec["z_R"][sl])),
        "energy": E, "cot": E / (rv.M * rv.g * adv) if adv > 1e-6 else np.inf,
        "cog_err_rms": float(np.sqrt(np.mean((rec["xi_m"][sl] - rec["xi_ref"][sl]) ** 2))),
        "cog_err_max": float(np.max(np.abs(rec["xi_m"][sl] - rec["xi_ref"][sl]))),
        "NF_min": float(np.min(rec["N_F"][sl])), "NR_min": float(np.min(rec["N_R"][sl])),
    }
