"""One closed-loop episode: locomotion, CoG controller, rail actuator, rover dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..params import Actuator, Rover
from .cog_controllers import BoundedResidual, Feedforward, FixedAt, FixedCentre, RailActuator, StaticSetPoint
from .features import cycle_features, cycle_metrics
from .locomotion import PhaseMachine
from .params_sim import SimParams, SoilWR
from .rover import RoverState, step
from .wheel_soil import WheelTable

Condition = Literal["B1", "B2", "B3", "A", "C", "FIX"]
CONDITION_LABELS = {"B1": "wheel driving", "B2": "PPL, mass centred", "B3": "PPL, physics-based set point",
                    "A": "PPL, Stage A", "C": "PPL, Stage C"}

_REC_KEYS = ["t", "phase", "cycle", "driver", "anchor", "x_R", "v_R", "x_F", "v_F", "L", "L_ref", "xi_m",
             "xi_ref", "dx", "N_F", "N_R", "z_F", "z_R", "s_F", "s_R", "om_F", "om_R", "T_F", "T_R",
             "F_act", "H_F", "H_R", "slide_F", "slide_R", "psi", "E"]


@dataclass
class RunResult:
    rec: dict
    cycles: list[dict]
    E_wheel: float
    E_act: float
    E_rail: float
    d: float
    completed: bool
    fail_reason: str | None
    fail_time: float
    meta: dict = field(default_factory=dict)

    @property
    def cot(self) -> float:
        E = self.E_wheel + self.E_act + self.E_rail
        M, g = self.meta["M"], self.meta["g"]
        return E / (M * g * self.d) if self.d > 1e-6 else np.inf


def make_controller(condition: Condition, rv: Rover, theta: float, p: SimParams, artefact=None):
    if condition in ("B1", "B2"):
        return FixedCentre(rv)
    if condition == "B3":
        return Feedforward(rv)
    if condition == "A":
        return StaticSetPoint(rv, theta, artefact)
    if condition == "C":
        return BoundedResidual(rv, theta, p, artefact)
    if condition == "FIX":          # training helper: hold a given rail position from the start
        return FixedAt(float(artefact))
    raise ValueError(condition)


def run_episode(rv: Rover, act: Actuator, p: SimParams, soil: SoilWR, theta: float, condition: Condition,
                artefact=None, n_cycles: int | None = None, seed: int = 0, tab: WheelTable | None = None,
                keep_rec: bool = True) -> RunResult:
    n_cycles = p.n_cycles if n_cycles is None else n_cycles
    tab = WheelTable.build(soil, rv, p) if tab is None else tab
    rng = np.random.default_rng(seed)
    mode = "wheel" if condition == "B1" else "ppl"
    pm = PhaseMachine(mode, rv, act, p)
    ctrl = make_controller(condition, rv, theta, p, artefact)
    dt = p.dt
    n = int(round(4 * act.T_phase * n_cycles / dt))
    rec = {k: np.zeros(n) for k in _REC_KEYS}
    st = RoverState(x_R=0.0, x_F=rv.L_max)
    ps0 = pm.state(0.0)
    rail = RailActuator(rv, act, ctrl.xi_ref(ps0, rv.L_max, theta))
    pm._last_phase = None
    E_w = E_a = E_r = 0.0
    E_cycle_start = 0.0
    k_cycle_start = 0
    d_nom = 2 * rv.stroke if mode == "ppl" else pm.v_nom * 4 * act.T_phase
    cycles: list[dict] = []
    fail = None; fail_t = np.nan
    slip_hi_steps = 0
    slide_steps = 0
    steps_per_phase = int(round(act.T_phase / dt))
    xi_prev_vel = 0.0
    for k in range(n):
        t = k * dt
        ps = pm.state(t)
        L = st.x_F - st.x_R
        xi_ref = ctrl.xi_ref(ps, L, theta)
        xi, xi_v, xi_a = rail.update(xi_ref, dt)
        om_ref = pm.omega_ref(ps)
        o = step(st, xi, xi_a, theta, ps.L_ref, ps.Ldot_ref, ps.anchor, om_ref, tab, soil, rv, p, dt)
        # rail power: force to move the mass along the slope against inertia and gravity
        F_rail = rv.m_m * (xi_a + st.a_R + rv.g * np.sin(theta))
        P_rail = max(F_rail * xi_v, 0.0)
        E_w += o.P_wheel * dt / p.eta_w
        E_a += o.P_act * dt / p.eta_a
        E_r += P_rail * dt / p.eta_r
        drv = 2 if ps.driver is None else (1 if ps.driver == "front" else 0)
        anc = -1 if ps.anchor is None else (1 if ps.anchor == "front" else 0)
        vals = (t, ps.phase, ps.cycle, drv, anc, st.x_R, st.v_R, st.x_F, st.v_F, L, ps.L_ref, xi, xi_ref,
                getattr(ctrl, "dx", {}).get(ps.phase % 2, 0.0) if condition == "C" else 0.0,
                o.N_F, o.N_R, o.z_F, o.z_R, o.s_F, o.s_R, o.om_F, o.om_R, o.T_F, o.T_R, o.F_act, o.H_F, o.H_R,
                o.slide_F, o.slide_R, o.psi, E_w + E_a + E_r)
        for key, v in zip(_REC_KEYS, vals):
            rec[key][k] = v
        # failure checks
        slide_steps = slide_steps + 1 if st.v_R < -p.v_slide else 0
        if slide_steps * dt >= p.t_slide or st.x_R < -rv.stroke:
            fail, fail_t = "slide_back", t; break
        if o.N_F <= 0 or o.N_R <= 0:
            fail, fail_t = "pitch_load", t; break
        if abs(o.psi) > p.pitch_limit:
            fail, fail_t = "pitch_angle", t; break
        s_drv = o.s_F if drv == 1 else (o.s_R if drv == 0 else max(o.s_F, o.s_R))
        slip_hi_steps = slip_hi_steps + 1 if s_drv > p.stall_slip else 0
        if slip_hi_steps >= steps_per_phase:
            fail, fail_t = "stall_slip", t; break
        # cycle boundary
        last = k == n - 1
        if (k + 1) % (4 * steps_per_phase) == 0 or last:
            k1 = k + 1
            m = cycle_metrics(rec, k_cycle_start, k1, E_cycle_start, E_w + E_a + E_r, d_nom, rv)
            f = cycle_features(rec, k_cycle_start, k1, rv, p, rng)
            m.update({"cycle": len(cycles), **{"f_" + kk: vv for kk, vv in f.items()}})
            cycles.append(m)
            ctrl.on_cycle_end(f)
            if len(cycles) >= 2 and m["advance"] < p.stall_adv_frac * d_nom:
                fail, fail_t = "stall_advance", t; break
            k_cycle_start, E_cycle_start = k1, E_w + E_a + E_r
    n_done = k + 1
    rec = {kk: v[:n_done] for kk, v in rec.items()}
    d = float(rec["x_R"][-1] - rec["x_R"][0])
    meta = {"condition": condition, "theta": theta, "seed": seed, "M": rv.M, "g": rv.g,
            "Delta_max": getattr(ctrl, "Delta_max", None), "n_cycles": n_cycles, "d_nom_cycle": d_nom}
    if condition == "C":
        meta["dx_max_abs"] = float(np.max(np.abs(rec["dx"]))) if n_done else 0.0
    return RunResult(rec if keep_rec else {kk: rec[kk] for kk in ("t", "x_R", "phase")}, cycles,
                     E_w, E_a, E_r, d, fail is None, fail, fail_t, meta)
