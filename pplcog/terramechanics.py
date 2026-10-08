"""Wheel-soil relations used to locate the physics-optimal mass position.

Rigid-wheel Bekker sinkage and compaction resistance (Wong 2008, ch. 2), an arc-length
contact patch, a Mohr-Coulomb thrust limit developed with slip according to the
Janosi-Hanamoto relation, and the traction utilisation defined in the manuscript.
These relations need absolute scale (sqrt(D), k_c / b), so their results are
illustrative for the nominal 12 kg rover, not dimensionless.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from .frame import anchor, rail_x_bounds
from .params import Rover, Soil
from .statics import loads_from_xm


def sinkage(W, soil: Soil, rv: Rover):
    """Bekker static sinkage of a rigid wheel of diameter D and width b under load W (per wheel)."""
    W = np.clip(np.asarray(W, dtype=float), 0.0, None)
    D = 2 * rv.r
    kk = soil.k_c + rv.b * soil.k_phi
    n = soil.n
    z0 = (3 * W / ((3 - n) * kk * np.sqrt(D))) ** (2 / (2 * n + 1))
    return np.minimum(z0, 0.95 * rv.r)


def compaction_resistance(W, soil: Soil, rv: Rover):
    """Bekker compaction resistance of a rigid wheel under load W (per wheel)."""
    W = np.clip(np.asarray(W, dtype=float), 0.0, None)
    D = 2 * rv.r
    kk = soil.k_c + rv.b * soil.k_phi
    n = soil.n
    e = (2 * n + 2) / (2 * n + 1)
    return (3 * W) ** e / ((3 - n) ** e * (n + 1) * kk ** (1 / (2 * n + 1)) * D ** ((n + 1) / (2 * n + 1)))


def contact_patch(z0, rv: Rover):
    """Arc-length contact patch: entry angle, patch length, patch area (per wheel)."""
    z0 = np.asarray(z0, dtype=float)
    theta_c = np.arccos(np.clip(1.0 - z0 / rv.r, -1.0, 1.0))
    ell = rv.r * theta_c
    return theta_c, ell, rv.b * ell


def thrust_max(W, soil: Soil, rv: Rover, slip: float | None = None):
    """Maximal thrust of one wheel at the reference slip: (A c + W tan phi) * Janosi-Hanamoto factor."""
    slip = soil.slip_ref if slip is None else slip
    W = np.clip(np.asarray(W, dtype=float), 0.0, None)
    z0 = sinkage(W, soil, rv)
    _, ell, A = contact_patch(z0, rv)
    jd = slip * ell
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(jd > 0, 1.0 - soil.K / jd * (1.0 - np.exp(-jd / soil.K)), 0.0)
    return (A * soil.c + W * np.tan(soil.phi_s)) * factor


def utilisation(x_m, L, theta, phase: int, soil: Soil, rv: Rover, xdd_m=0.0,
                count_anchor_resistance: bool = True):
    """Traction utilisation (mu_driver, mu_anchor) as defined in the manuscript.

    H_req = M g sin(theta) + R_F + R_R + m_m xdd_m, supplied by the driving pair and
    reacted by the anchoring pair.  Loads and thrusts are per pair (two wheels).
    """
    N_F, N_R = loads_from_xm(x_m, L, theta, rv)
    N = {"front": N_F, "rear": N_R}
    R = {k: 2 * compaction_resistance(v / 2, soil, rv) for k, v in N.items()}
    H = {k: 2 * thrust_max(v / 2, soil, rv) for k, v in N.items()}
    a = anchor(phase)
    d = "rear" if a == "front" else "front"
    resist = R[d] + (R[a] if count_anchor_resistance else 0.0)
    H_req = rv.M * rv.g * np.sin(theta) + resist + rv.m_m * np.asarray(xdd_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_d = np.where(H[d] > 0, H_req / H[d], np.inf)
        mu_a = np.where(H[a] > 0, H_req / H[a], np.inf)
    return mu_d, mu_a


def xm_opt(L, theta, phase: int, soil: Soil, rv: Rover) -> float:
    """Mass position on the rail that minimises max(mu_driver, mu_anchor); nan if infeasible."""
    lo, hi = rail_x_bounds(L, rv)
    lo, hi = float(lo), float(hi)
    grid = np.linspace(lo, hi, 97)
    N_F, N_R = loads_from_xm(grid, L, theta, rv)
    ok = (N_F >= rv.N_min) & (N_R >= rv.N_min)
    if not ok.any():
        return float("nan")
    glo, ghi = grid[ok].min(), grid[ok].max()

    def cost(x):
        mu_d, mu_a = utilisation(x, L, theta, phase, soil, rv)
        return float(np.maximum(mu_d, mu_a))

    res = minimize_scalar(cost, bounds=(glo, ghi), method="bounded", options={"xatol": 1e-5})
    return float(res.x)


def sample_soils(soil: Soil, rng: np.random.Generator, n: int) -> list[Soil]:
    """Independent draws of the uncertain soil parameters over their ranges."""
    out = []
    draws = {}
    for name, (lo, hi) in soil.ranges.items():
        if name in soil.log_uniform:
            draws[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        else:
            draws[name] = rng.uniform(lo, hi, n)
    for i in range(n):
        out.append(replace(soil, **{k: float(v[i]) for k, v in draws.items()}))
    return out
