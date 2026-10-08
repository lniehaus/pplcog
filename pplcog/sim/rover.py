"""Pitch-plane dynamics of the two-unit PPL rover with a movable mass.

Frame: x along the slope, uphill positive; the front unit is uphill.  Units: rear unit R
(rear pair + chassis + rail), front unit F (front pair), movable mass on the rail of R
(rail_on='rear').  Sinkage is quasi-steady from the wheel table; pitch is kinematic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..frame import h_cog, xi_to_x
from ..params import Rover
from ..statics import cog_x, normal_loads
from .params_sim import SimParams, SoilWR
from .wheel_soil import WheelTable, bulldozing_resistance


@dataclass
class RoverState:
    x_R: float = 0.0
    v_R: float = 0.0
    x_F: float = 0.0
    v_F: float = 0.0
    a_R: float = 0.0
    a_Rf: float = 0.0      # low-pass filtered chassis acceleration used for load transfer
    p_F: float = 0.0       # anchor plastic reference, front
    p_R: float = 0.0       # anchor plastic reference, rear
    z_F: float = 0.0
    z_R: float = 0.0


@dataclass
class StepOut:
    N_F: float; N_R: float; H_F: float; H_R: float; F_act: float
    s_F: float; s_R: float; om_F: float; om_R: float; T_F: float; T_R: float
    z_F: float; z_R: float; slide_F: float; slide_R: float; psi: float
    P_wheel: float; P_act: float; P_rail: float


def anchor_capacity(W_half: float, tab: WheelTable, soil: SoilWR, rv: Rover) -> tuple[float, float]:
    """Holding capacity of an anchored pair (two wheels) and its sinkage."""
    z = tab.sinkage(max(W_half, 0.0), 0.0)
    A = rv.b * rv.r * np.arccos(np.clip(1.0 - z / rv.r, -1.0, 1.0))
    cap = 2.0 * (A * soil.c + max(W_half, 0.0) * np.tan(soil.phi_s) + bulldozing_resistance(z, soil, rv))
    return float(cap), z


def slip(v: float, om: float, r: float, v_eps: float) -> float:
    rw = r * om
    if rw >= v:
        return min((rw - v) / max(rw, v_eps), 1.0)
    return max((rw - v) / max(v, v_eps), -1.0)


def step(st: RoverState, xi_m: float, xi_dd: float, theta: float, L_ref: float, Ld_ref: float,
         anchor: str | None, om_ref: float, tab: WheelTable, soil: SoilWR, rv: Rover,
         p: SimParams, dt: float) -> StepOut:
    """Advance the rover by one step (semi-implicit Euler, linearly implicit soil damping)."""
    g = rv.g
    m_F = rv.m_pair
    m_Rt = rv.m_pair + rv.m_body + rv.m_m
    L = st.x_F - st.x_R
    Ld = st.v_F - st.v_R
    h = h_cog(rv)
    # normal loads with longitudinal load transfer
    xc = float(cog_x(xi_to_x(xi_m, L, rv), L, rv))
    N_F, N_R = normal_loads(xc, L, theta, rv)
    N_F = float(N_F) - (rv.m_m * xi_dd * rv.h_rail + rv.M * st.a_Rf * h) / L
    N_R = rv.M * g * np.cos(theta) - N_F
    NF_c, NR_c = max(N_F, 0.0), max(N_R, 0.0)
    # wheelbase actuator
    F_act = float(np.clip(p.Kp_L * (L_ref - L) + p.Kd_L * (Ld_ref - Ld), -p.F_act_max, p.F_act_max))

    out = {}
    forces = {}
    for unit, Nc, v, x_u, pref in (("front", NF_c, st.v_F, st.x_F, st.p_F), ("rear", NR_c, st.v_R, st.x_R, st.p_R)):
        W_half = Nc / 2
        if anchor == unit:
            cap, z = anchor_capacity(W_half, tab, soil, rv)
            m_u = m_F if unit == "front" else m_Rt
            k_s = cap / soil.K
            c_s = 2 * p.zeta_anchor * np.sqrt(k_s * m_u)
            # implicit spring-damper: H = -k (x + v' dt - pref) - c v'
            H0 = -k_s * (x_u - pref)
            dHdv = -(k_s * dt + c_s)
            forces[unit] = (H0, dHdv, cap, k_s)
            out[unit] = dict(s=slip(v, 0.0, rv.r, p.v_eps), om=0.0, T=0.0, z=z)
        else:
            om = om_ref
            s = slip(v, om, rv.r, p.v_eps)
            z, DP, T, dDP = tab.lookup(W_half, s)
            saturated = False
            if T > p.T_max and s > 0:
                # motor at its torque limit: thrust fixed by T_max, wheel speed follows the unit
                s = tab.slip_at_torque(W_half, p.T_max, s)
                z, DP, T, dDP = tab.lookup(W_half, s)
                T = min(T, p.T_max)
                om = max(v, p.v_eps) / (rv.r * (1.0 - min(s, 0.999)))
                saturated = True
            rw = max(rv.r * om, p.v_eps)
            if saturated:
                dHdv = 0.0
            else:
                dHdv = -2.0 * dDP / rw if rw >= v else -2.0 * dDP * rw / max(v, p.v_eps) ** 2
            forces[unit] = (2.0 * DP, min(dHdv, 0.0), None, None)
            out[unit] = dict(s=s, om=om, T=T, z=z)

    # velocity updates (each unit separately; coupling through F_act is explicit)
    def solve(m, v, F_ext, H0, dHdv):
        return (v + dt / m * (F_ext + H0)) / (1.0 - dt / m * dHdv)

    # front
    H0, dHdv, cap, k_s = forces["front"]
    vF_new = solve(m_F, st.v_F, -m_F * g * np.sin(theta) + F_act, H0, dHdv)
    H_F = H0 + dHdv * vF_new
    slide_F = 0.0
    if cap is not None and abs(H_F) > cap:
        H_F = float(np.sign(H_F) * cap)
        vF_new = st.v_F + dt / m_F * (-m_F * g * np.sin(theta) + F_act + H_F)
        slide_F = 1.0
    # rear
    H0, dHdv, cap, k_s = forces["rear"]
    vR_new = solve(m_Rt, st.v_R, -m_Rt * g * np.sin(theta) - rv.m_m * xi_dd - F_act, H0, dHdv)
    H_R = H0 + dHdv * vR_new
    slide_R = 0.0
    if cap is not None and abs(H_R) > cap:
        H_R = float(np.sign(H_R) * cap)
        vR_new = st.v_R + dt / m_Rt * (-m_Rt * g * np.sin(theta) - rv.m_m * xi_dd - F_act + H_R)
        slide_R = 1.0

    st.a_R = (vR_new - st.v_R) / dt
    # vertical compliance of chassis and soil is not modelled; filter the impulsive part of a_R
    st.a_Rf += dt / p.tau_a * (st.a_R - st.a_Rf)
    st.v_F, st.v_R = vF_new, vR_new
    st.x_F += vF_new * dt
    st.x_R += vR_new * dt
    # plastic anchor references
    if anchor == "front":
        if slide_F:
            st.p_F = st.x_F + H_F / forces["front"][3]
    else:
        st.p_F = st.x_F
    if anchor == "rear":
        if slide_R:
            st.p_R = st.x_R + H_R / forces["rear"][3]
    else:
        st.p_R = st.x_R
    st.z_F, st.z_R = out["front"]["z"], out["rear"]["z"]
    psi = float(np.arctan2(st.z_R - st.z_F, L))
    # mechanical power (positive work only)
    P_wheel = 2.0 * (max(out["front"]["T"] * out["front"]["om"], 0.0) + max(out["rear"]["T"] * out["rear"]["om"], 0.0))
    P_act = max(F_act * (vF_new - vR_new), 0.0)
    return StepOut(N_F, N_R, H_F, H_R, F_act, out["front"]["s"], out["rear"]["s"], out["front"]["om"],
                   out["rear"]["om"], out["front"]["T"], out["rear"]["T"], st.z_F, st.z_R, slide_F, slide_R,
                   psi, P_wheel, P_act, 0.0)
