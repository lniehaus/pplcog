"""Equation (4) actuator loop with rate and acceleration limits, an optional PI
correction on the load fraction, and a bounded learned offset (review item c).

Controller variants
* ``ff``            reference = feedforward set point + offset
* ``ff_pi_after``   reference = feedforward + PI + offset; the PI measures the plant
                    and therefore sees the offset's effect and works against it
* ``ff_pi_blind``   reference = feedforward + PI + offset; the PI error is taken
                    against the offset-shifted target, so it does not react to the offset

The plant used for the PI measurement carries a deliberate mismatch (body CoG shifted
by ``Actuator.mismatch`` times L) so that the PI has a real correction to make.
Loads are always evaluated with the true (mismatched) plant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .frame import anchor, phase_timeline, rail_x_bounds, wheelbase, xi_to_x, x_to_xi
from .params import Actuator, Rover, Study
from .statics import NF_deviation_bound, delta_max, load_fraction, loads_from_xm, rail_room, xm_star

Controller = Literal["ff", "ff_pi_after", "ff_pi_blind"]


@dataclass
class SimResult:
    t: np.ndarray
    phase: np.ndarray
    L: np.ndarray
    x_ref: np.ndarray
    x_m: np.ndarray
    v: np.ndarray
    offset: np.ndarray
    u_pi: np.ndarray
    N_F: np.ndarray
    N_R: np.ndarray
    Delta_max: float
    controller: str

    def deviation(self, nominal: "SimResult") -> np.ndarray:
        return self.N_F - nominal.N_F

    def bound(self, rv: Rover, theta: float) -> np.ndarray:
        return NF_deviation_bound(theta, self.L, rv, self.Delta_max)

    def viol_ratio(self, nominal: "SimResult", rv: Rover, theta: float) -> float:
        b = self.bound(rv, theta)
        return float(np.max(np.abs(self.deviation(nominal)) / b))


def offset_schedule(phase: np.ndarray, Delta: float, kind: str = "square") -> np.ndarray:
    """Worst-case learned offset: +-Delta with the sign flipping every phase, or constant +Delta."""
    if kind == "square":
        return Delta * np.where(phase % 2 == 1, 1.0, -1.0)
    if kind == "const":
        return np.full(phase.shape, Delta)
    raise ValueError(kind)


def envelope_for_cycle(rv: Rover, theta: float, rho_star: float) -> float:
    """Realisable envelope: Eq. (5) and the rail room, at the most restrictive wheelbase of the cycle."""
    vals = []
    for L in (rv.L_min, rv.L_max):
        xm = float(np.clip(xm_star(rho_star, L, theta, rv), *rail_x_bounds(L, rv)))
        N_F_nom, _ = loads_from_xm(xm, L, theta, rv)
        vals.append(min(float(delta_max(theta, L, rv, N_F_nom)), float(rail_room(xm, L, rv))))
    return min(vals)


def simulate(rv: Rover, act: Actuator, st: Study, theta: float, controller: Controller = "ff",
             offset: bool = True, offset_kind: str = "square", Delta_max: float | None = None,
             rho_star: float | None = None) -> SimResult:
    rho_star = st.rho_star if rho_star is None else rho_star
    Delta = envelope_for_cycle(rv, theta, rho_star) if Delta_max is None else Delta_max
    plant = replace(rv, xi_body=rv.xi_body + act.mismatch * rv.L_max)  # true plant
    t, phase, s = phase_timeline(act.T_phase, act.dt, st.n_cycles)
    n = t.size
    L = wheelbase(phase, s, rv)
    off = offset_schedule(phase, Delta, offset_kind) if offset else np.zeros(n)
    # feedforward set point in the rail frame, clipped to the rail
    xi_ff = x_to_xi(xm_star(rho_star, L, theta, rv), L, rv)
    xi_ff = np.clip(xi_ff, rv.rail_lo, rv.rail_hi)
    lam_star = rho_star / (1.0 + rho_star)  # target load fraction N_F / (M g cos theta)

    x_ref = np.empty(n); x_m = np.empty(n); v = np.empty(n); u_pi = np.empty(n)
    N_F = np.empty(n); N_R = np.empty(n)
    xi = float(xi_ff[0]); vel = 0.0; integ = 0.0
    use_pi = controller != "ff"
    # sensitivity of the load fraction to the mass position, for the blind variant
    dlam_dx = rv.m_m / (rv.M * L)  # per metre of x (rail frame sign handled below)
    sign = 1.0 if rv.rail_on == "rear" else -1.0
    for k in range(n):
        x_now = float(xi_to_x(xi, L[k], rv))
        NF_k, NR_k = loads_from_xm(x_now, L[k], theta, plant)
        lam = float(load_fraction(NF_k, theta, plant))
        if use_pi:
            target = lam_star
            if controller == "ff_pi_blind":
                target = lam_star + sign * float(dlam_dx[k]) * off[k]
            e = target - lam
            integ += e * act.dt
            u = sign * (act.Kp * e + act.Ki * integ)
            # anti-windup: keep the total reference on the rail
            lo, hi = rv.rail_lo - xi_ff[k] - off[k], rv.rail_hi - xi_ff[k] - off[k]
            if u > hi or u < lo:
                u = float(np.clip(u, lo, hi))
                integ = (u / sign - act.Kp * e) / act.Ki if act.Ki > 0 else integ
        else:
            u = 0.0
        ref = float(np.clip(xi_ff[k] + u + off[k], rv.rail_lo, rv.rail_hi))
        # first-order tracking with rate and acceleration limits
        v_des = (ref - xi) / act.tau
        a = float(np.clip((v_des - vel) / act.dt, -act.a_max, act.a_max))
        vel = float(np.clip(vel + a * act.dt, -act.v_max, act.v_max))
        xi = float(np.clip(xi + vel * act.dt, rv.rail_lo, rv.rail_hi))
        x_ref[k] = ref; x_m[k] = xi; v[k] = vel; u_pi[k] = u; N_F[k] = NF_k; N_R[k] = NR_k
    return SimResult(t, phase, L, x_ref, x_m, v, off, u_pi, N_F, N_R, Delta, controller)


def steady_state_offset_effect(with_off: SimResult, without: SimResult, act: Actuator) -> float:
    """Ratio of the achieved mass displacement to the commanded offset over the last cycle.

    1 means the offset is fully realised, 0 means the PI cancelled it."""
    k = with_off.t >= with_off.t[-1] - 4 * act.T_phase
    achieved = np.abs(with_off.x_m[k] - without.x_m[k])
    return float(np.mean(achieved) / np.mean(np.abs(with_off.offset[k])))
