"""Equations (1) to (3) and (5) of the manuscript: quasi-static pitch-plane statics.

All functions broadcast over numpy arrays.  Angles are in radians.  x_m is the
movable-mass position in the manuscript's x frame (from the rear contact).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .frame import h_cog, rail_x_bounds, x_body, xi_to_x, x_to_xi
from .params import Rover


def cog_x(x_m, L, rv: Rover):
    """Eq. (1): longitudinal CoG position."""
    x_m = np.asarray(x_m, dtype=float)
    return (rv.m_b * x_body(L, rv) + rv.m_m * x_m) / rv.M


def normal_loads(x_c, L, theta, rv: Rover, h: float | None = None):
    """Eq. (2): normal loads on the front and rear pair. Negative values mean tipping."""
    h = h_cog(rv) if h is None else h
    x_c = np.asarray(x_c, dtype=float)
    L = np.asarray(L, dtype=float)
    theta = np.asarray(theta, dtype=float)
    W = rv.M * rv.g
    N_F = W / L * (x_c * np.cos(theta) - h * np.sin(theta))
    N_R = W * np.cos(theta) - N_F
    return N_F, N_R


def load_ratio(N_F, N_R):
    N_F = np.asarray(N_F, dtype=float)
    N_R = np.asarray(N_R, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(N_R > 0, N_F / N_R, np.inf)


def load_fraction(N_F, theta, rv: Rover):
    """N_F / (M g cos theta); linear in x_m, used as the PI variable."""
    return np.asarray(N_F) / (rv.M * rv.g * np.cos(theta))


def loads_from_xm(x_m, L, theta, rv: Rover, h: float | None = None):
    return normal_loads(cog_x(x_m, L, rv), L, theta, rv, h)


def xc_star(rho_star, L, theta, rv: Rover, h: float | None = None):
    """Eq. (3a): CoG position that yields the target load ratio."""
    h = h_cog(rv) if h is None else h
    L = np.asarray(L, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return rho_star / (1.0 + rho_star) * L + h * np.tan(theta)


def xm_star(rho_star, L, theta, rv: Rover, h: float | None = None):
    """Eq. (3b): mass position for the target load ratio, in the x frame (may lie off the rail)."""
    xc = xc_star(rho_star, L, theta, rv, h)
    return (rv.M * xc - rv.m_b * x_body(L, rv)) / rv.m_m


def feasible(x_m, L, theta, rv: Rover):
    """Feasibility flags: on the rail, N_F >= N_min, N_R >= N_min."""
    lo, hi = rail_x_bounds(L, rv)
    x_m = np.asarray(x_m, dtype=float)
    N_F, N_R = loads_from_xm(x_m, L, theta, rv)
    in_rail = (x_m >= lo - 1e-12) & (x_m <= hi + 1e-12)
    return {"in_rail": in_rail, "NF_ok": N_F >= rv.N_min, "NR_ok": N_R >= rv.N_min,
            "ok": in_rail & (N_F >= rv.N_min) & (N_R >= rv.N_min)}


def theta_tip_back(x_m, L, rv: Rover):
    """Slope at which N_F reaches zero (backward tipping), rad."""
    return np.arctan2(cog_x(x_m, L, rv), h_cog(rv))


def dNF_dL_sign(rv: Rover, theta: float = 0.0, x_m: float | None = None) -> int:
    """Sign of dN_F/dL at fixed mass position in the rail frame (review item a)."""
    if x_m is None:
        x_m = xi_to_x(0.5 * (rv.rail_lo + rv.rail_hi), rv.L_max, rv)
    xi = x_to_xi(x_m, rv.L_max, rv)
    L1, L2 = rv.L_min, rv.L_max
    N1 = loads_from_xm(xi_to_x(xi, L1, rv), L1, theta, rv)[0]
    N2 = loads_from_xm(xi_to_x(xi, L2, rv), L2, theta, rv)[0]
    return int(np.sign(N2 - N1))


def _balance_ok(rv: Rover, L: float, rho_star: float, theta: float, h: float | None) -> bool:
    lo, hi = rail_x_bounds(L, rv)
    xm = float(xm_star(rho_star, L, theta, rv, h))
    if not (lo - 1e-12 <= xm <= hi + 1e-12):
        return False
    N_F, N_R = loads_from_xm(xm, L, theta, rv, h)
    return bool(N_F >= rv.N_min and N_R >= rv.N_min)


def slope_range(rv: Rover, L: float, rho_star: float, theta_lo: float = np.radians(-30.0),
                theta_hi: float = np.radians(60.0), h: float | None = None) -> tuple[float, float]:
    """Interval of slopes (rad) over which the target load ratio is reachable on the rail with
    both loads above N_min.  x_c* is monotone in theta, so the feasible set is one interval.
    Returns (nan, nan) if it is empty."""
    th = np.linspace(theta_lo, theta_hi, 901)
    flags = np.array([_balance_ok(rv, L, rho_star, t, h) for t in th])
    if not flags.any():
        return float("nan"), float("nan")
    idx = np.flatnonzero(flags)
    a, b = idx[0], idx[-1]
    g = lambda t: 1.0 if _balance_ok(rv, L, rho_star, t, h) else -1.0
    lo = th[a] if a == 0 else brentq(g, th[a - 1], th[a], xtol=1e-6)
    hi = th[b] if b == th.size - 1 else brentq(g, th[b], th[b + 1], xtol=1e-6)
    return float(lo), float(hi)


def slope_limit(rv: Rover, L: float, rho_star: float, h: float | None = None) -> float:
    """Upper end of ``slope_range`` (rad); nan if the target is never reachable."""
    return slope_range(rv, L, rho_star, h=h)[1]


def rail_room(x_m, L, rv: Rover):
    """Distance from a set point to the nearer rail end (the physically realisable envelope)."""
    lo, hi = rail_x_bounds(L, rv)
    x_m = np.asarray(x_m, dtype=float)
    return np.clip(np.minimum(x_m - lo, hi - x_m), 0.0, None)


def delta_max(theta, L, rv: Rover, N_F_nom):
    """Eq. (5) design rule: largest envelope that keeps N_F >= N_min under any bounded offset."""
    theta = np.asarray(theta, dtype=float)
    val = np.asarray(L) * (np.asarray(N_F_nom) - rv.N_min) / (rv.m_m * rv.g * np.cos(theta))
    return np.clip(val, 0.0, None)


def delta_max_both(theta, L, rv: Rover, N_F_nom):
    """Envelope that keeps both N_F and N_R above N_min (the manuscript states only N_F)."""
    N_R_nom = rv.M * rv.g * np.cos(theta) - np.asarray(N_F_nom)
    return np.minimum(delta_max(theta, L, rv, N_F_nom), delta_max(theta, L, rv, N_R_nom))


def NF_deviation_bound(theta, L, rv: Rover, Delta):
    """Left-hand side bound of Eq. (5): |N_F - N_F_nom| <= m_m g cos(theta) Delta / L."""
    return rv.m_m * rv.g * np.cos(theta) * np.asarray(Delta) / np.asarray(L)
