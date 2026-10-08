"""Data for the four panels of the model-evaluation figure and the key numbers.

Every function returns a plain dict of numpy arrays and floats so that it can be
serialised and tested without matplotlib.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .actuator import simulate, steady_state_offset_effect
from .frame import h_cog, rail_x_bounds, x_to_xi, xi_to_x
from .params import Actuator, Rover, Soil, Study
from .statics import (delta_max, delta_max_both, loads_from_xm, load_ratio, rail_room, slope_limit,
                      slope_range, theta_tip_back, xm_star, dNF_dL_sign)
from .terramechanics import sample_soils, utilisation, xm_opt


def fig_a_data(rv: Rover, st: Study) -> dict:
    """log10 load ratio over (mass position on the rail, slope) for L_min and L_max."""
    rv = rv.with_f(st.f_ref)
    u = np.linspace(0.0, 1.0, 121)                     # rail coordinate xi_m / R
    th = np.radians(st.theta_deg)
    U, TH = np.meshgrid(u, th, indexing="xy")
    out = {"u": u, "theta_deg": st.theta_deg, "f": st.f_ref}
    for tag, L in (("Lmin", rv.L_min), ("Lmax", rv.L_max)):
        xi = rv.rail_lo + U * rv.rail_len
        x_m = xi_to_x(xi, L, rv)
        N_F, N_R = loads_from_xm(x_m, L, TH, rv)
        rho = load_ratio(N_F, N_R)
        infeasible = (N_F < rv.N_min) | (N_R < rv.N_min)
        with np.errstate(divide="ignore"):
            out[tag] = {"log10rho": np.log10(np.clip(rho, 1e-3, 1e3)), "infeasible": infeasible,
                        "u_star": (x_to_xi(xm_star(st.rho_star, L, th, rv), L, rv) - rv.rail_lo) / rv.rail_len,
                        "theta_tip_deg": np.degrees(theta_tip_back(xi_to_x(rv.rail_lo + u * rv.rail_len, L, rv), L, rv)),
                        "L": L}
    return out


def fig_b_data(rv: Rover, st: Study) -> dict:
    """Slope limit per mass fraction and the set-point curves; sensitivities to h_rail and rail placement."""
    th = np.radians(st.theta_deg)
    out = {"theta_deg": st.theta_deg, "f_list": st.f_list, "curves": {}, "limits": {},
           "sens_h": {}, "sens_rail": {}}
    for f in st.f_list:
        r = rv.with_f(f)
        lim = {}
        for tag, L in (("Lmin", r.L_min), ("Lmax", r.L_max)):
            lo, hi = slope_range(r, L, st.rho_star)
            lim[tag] = {"lo": np.degrees(lo), "hi": np.degrees(hi)}
        lim["balanced_lo"] = float(np.nanmax([lim["Lmin"]["lo"], lim["Lmax"]["lo"]]))
        lim["balanced_hi"] = float(np.nanmin([lim["Lmin"]["hi"], lim["Lmax"]["hi"]]))
        out["limits"][f] = lim
        cur = {}
        for tag, L in (("Lmin", r.L_min), ("Lmax", r.L_max)):
            cur[tag] = (x_to_xi(xm_star(st.rho_star, L, th, r), L, r) - r.rail_lo) / r.rail_len
        out["curves"][f] = cur
    for hL in st.h_rail_over_L:
        out["sens_h"][hL] = {f: np.degrees(np.nanmin([slope_limit(replace(rv.with_f(f), h_rail=hL * rv.L_max), L, st.rho_star)
                                                      for L in (rv.L_min, rv.L_max)])) for f in st.f_list}
    for rs in st.rail_start_sweep:
        out["sens_rail"][rs] = {f: np.degrees(np.nanmin([slope_limit(replace(rv.with_f(f), rail_start=rs), L, st.rho_star)
                                                         for L in (rv.L_min, rv.L_max)])) for f in st.f_list}
    return out


def _geom_samples(rv: Rover, st: Study, rng: np.random.Generator) -> list[Rover]:
    draws = {k: rng.uniform(lo, hi, st.n_mc) for k, (lo, hi) in st.geom_ranges.items()}
    return [replace(rv, **{k: float(v[i]) for k, v in draws.items()}) for i in range(st.n_mc)]


def fig_c_data(rv: Rover, soil: Soil, st: Study, rng: np.random.Generator) -> dict:
    """Envelope from Eq. (5) versus slope per f, and the set-point shift under geometric uncertainty.

    Also the (degenerate) spread of the utilisation optimum under soil uncertainty, reported as a number."""
    th = np.radians(st.theta_mc_deg)
    out = {"theta_deg": st.theta_mc_deg, "f_list": st.f_list, "env": {}, "env_both": {}, "env_rail": {},
           "geom": {}, "soil_spread_R": {}, "mu_nominal": {}}
    soils = sample_soils(soil, rng, st.n_mc)
    for f in st.f_list:
        r = rv.with_f(f)
        env = np.full(th.size, np.nan); env2 = np.full(th.size, np.nan); env3 = np.full(th.size, np.nan)
        q = np.full((3, th.size), np.nan)
        for i, t in enumerate(th):
            # the envelope is evaluated at the most restrictive wheelbase of the cycle
            e_vals, e2_vals, e3_vals = [], [], []
            for L in (r.L_min, r.L_max):
                lo, hi = rail_x_bounds(L, r)
                xm = float(np.clip(xm_star(st.rho_star, L, t, r), lo, hi))
                N_F_nom, _ = loads_from_xm(xm, L, t, r)
                e_vals.append(float(delta_max(t, L, r, N_F_nom)))
                e2_vals.append(float(delta_max_both(t, L, r, N_F_nom)))
                e3_vals.append(min(e_vals[-1], float(rail_room(xm, L, r))))
            env[i], env2[i], env3[i] = min(e_vals), min(e2_vals), min(e3_vals)
            # set-point shift under geometric uncertainty, at L_max (largest h tan(theta) lever)
            L = r.L_max
            x_nom = x_to_xi(xm_star(st.rho_star, L, t, r), L, r)
            shifts = np.array([x_to_xi(xm_star(st.rho_star, L, t, g), L, g) - x_nom for g in _geom_samples(r, st, rng)])
            q[:, i] = np.percentile(np.abs(shifts), [50, 90, 95])
        out["env"][f] = env / r.rail_len
        out["env_both"][f] = env2 / r.rail_len
        out["env_rail"][f] = env3 / r.rail_len
        out["geom"][f] = q / r.rail_len
    # soil sensitivity of the utilisation optimum, f_ref, 5 degrees, phase 1 (expected ~0)
    r = rv.with_f(st.f_ref)
    t5 = np.radians(5.0)
    xo = np.array([xm_opt(r.L_max, t5, 1, s, r) for s in soils[: min(60, st.n_mc)]])
    out["soil_spread_R"] = {"p5_p95": float(np.nanpercentile(xo, 95) - np.nanpercentile(xo, 5)) / r.rail_len,
                            "std": float(np.nanstd(xo)) / r.rail_len,
                            "theta_deg": 5.0, "n": int(np.isfinite(xo).sum())}
    for td in (0.0, 10.0, 20.0):
        t = np.radians(td)
        lo, hi = rail_x_bounds(r.L_max, r)
        xm = float(np.clip(xm_star(st.rho_star, r.L_max, t, r), lo, hi))
        mu = utilisation(xm, r.L_max, t, 1, soil, r)
        out["mu_nominal"][td] = float(np.maximum(*mu))
    return out


def sim_theta(rv: Rover, st: Study) -> float:
    """Slope for the time simulation: st.theta_ref_deg if inside the balanced range of f_ref,
    otherwise the midpoint of that range (rad)."""
    r = rv.with_f(st.f_ref)
    lo = max(slope_range(r, L, st.rho_star)[0] for L in (r.L_min, r.L_max))
    hi = min(slope_range(r, L, st.rho_star)[1] for L in (r.L_min, r.L_max))
    t = np.radians(st.theta_ref_deg)
    if np.isnan(lo) or np.isnan(hi):
        return t
    return t if lo <= t <= hi else 0.5 * (lo + hi)


def fig_d_data(rv: Rover, act: Actuator, st: Study) -> dict:
    """Closed-loop time simulation inside the balanced slope range for the three controller variants."""
    r = rv.with_f(st.f_ref)
    theta = sim_theta(rv, st)
    out = {"theta_deg": float(np.degrees(theta)), "f": st.f_ref, "runs": {}, "viol": {}, "offset_effect": {}}
    for ctrl in ("ff", "ff_pi_after", "ff_pi_blind"):
        nom = simulate(r, act, st, theta, ctrl, offset=False)
        off = simulate(r, act, st, theta, ctrl, offset=True, Delta_max=nom.Delta_max)
        out["runs"][ctrl] = {"nom": nom, "off": off}
        out["viol"][ctrl] = off.viol_ratio(nom, r, theta)
        out["offset_effect"][ctrl] = steady_state_offset_effect(off, nom, act)
        const = simulate(r, act, st, theta, ctrl, offset=True, offset_kind="const", Delta_max=nom.Delta_max)
        out.setdefault("offset_effect_const", {})[ctrl] = steady_state_offset_effect(const, nom, act)
    out["Delta_max_R"] = out["runs"]["ff"]["nom"].Delta_max / r.rail_len
    return out


def key_numbers(rv: Rover, act: Actuator, st: Study, a: dict, b: dict, c: dict, d: dict) -> dict:
    r = rv.with_f(st.f_ref)
    i_ref = int(np.argmin(np.abs(st.theta_mc_deg - st.theta_ref_deg)))
    return {
        "nominal": {"M_kg": rv.M, "L_min_m": rv.L_min, "L_max_m": rv.L_max, "rail_m": rv.rail_len,
                    "rail_start_m": rv.rail_start, "h_rail_over_L": rv.h_rail / rv.L_max,
                    "rail_on": rv.rail_on, "rho_star": st.rho_star, "N_min_over_Mg": rv.N_min_frac},
        "h_cog_over_L": {str(f): h_cog(rv.with_f(f)) / rv.L_max for f in st.f_list},
        "slope_limit_deg": {str(f): b["limits"][f] for f in st.f_list},
        "slope_limit_sens_h_rail_over_L": {str(k): {str(f): v for f, v in d_.items()} for k, d_ in b["sens_h"].items()},
        "slope_limit_sens_rail_start_m": {str(k): {str(f): v for f, v in d_.items()} for k, d_ in b["sens_rail"].items()},
        "dNF_dL_sign": {ro: dNF_dL_sign(replace(rv, rail_on=ro)) for ro in ("rear", "front")},
        "envelope_over_R_at_theta_ref": {str(f): float(c["env"][f][i_ref]) for f in st.f_list},
        "envelope_rail_limited_over_R_at_theta_ref": {str(f): float(c["env_rail"][f][i_ref]) for f in st.f_list},
        "envelope_both_loads_over_R_at_theta_ref": {str(f): float(c["env_both"][f][i_ref]) for f in st.f_list},
        "geom_shift_p95_over_R_at_theta_ref": {str(f): float(c["geom"][f][2, i_ref]) for f in st.f_list},
        "soil_spread_of_utilisation_optimum_over_R": c["soil_spread_R"],
        "mu_nominal_illustrative": {str(k): v for k, v in c["mu_nominal"].items()},
        "sim": {"theta_deg": d["theta_deg"], "f": d["f"], "Delta_max_over_R": d["Delta_max_R"],
                "viol_ratio": d["viol"], "offset_effect_last_cycle": d["offset_effect"],
                "offset_effect_const_last_cycle": d["offset_effect_const"]},
        "neglected_inertial_moment_over_MgL": r.m_m * act.a_max * r.h_rail / (r.M * r.g * r.L_max),
        "seed": st.seed, "n_mc": st.n_mc,
    }
