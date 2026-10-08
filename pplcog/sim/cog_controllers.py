"""Rail actuator and the CoG controllers used as baselines and stages.

All controllers return a rail reference xi_ref (rail frame) that the actuator tracks.
No proportional-integral correction is used anywhere (see the quasi-static study).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..frame import rail_x_bounds, x_to_xi
from ..params import Actuator, Rover
from ..statics import delta_max_both, loads_from_xm, xm_star
from .locomotion import PhaseState
from .params_sim import SimParams


class RailActuator:
    """Eq. (4) first-order tracking with rate and acceleration limits, saturated to the rail."""

    def __init__(self, rv: Rover, act: Actuator, xi0: float):
        self.rv, self.act = rv, act
        self.xi = float(np.clip(xi0, rv.rail_lo, rv.rail_hi))
        self.vel = 0.0
        self.acc = 0.0

    def update(self, ref: float, dt: float) -> tuple[float, float, float]:
        rv, act = self.rv, self.act
        ref = float(np.clip(ref, rv.rail_lo, rv.rail_hi))
        v_des = (ref - self.xi) / act.tau
        a = float(np.clip((v_des - self.vel) / dt, -act.a_max, act.a_max))
        v_new = float(np.clip(self.vel + a * dt, -act.v_max, act.v_max))
        xi_new = float(np.clip(self.xi + v_new * dt, rv.rail_lo, rv.rail_hi))
        self.acc = (v_new - self.vel) / dt
        self.vel, self.xi = v_new, xi_new
        return self.xi, self.vel, self.acc


def feedforward_xi(L: float, theta: float, rv: Rover, rho_star: float = 1.0) -> float:
    lo, hi = rail_x_bounds(L, rv)
    xm = float(np.clip(xm_star(rho_star, L, theta, rv), lo, hi))
    return float(x_to_xi(xm, L, rv))


def delta_max_for_run(rv: Rover, theta: float, p: SimParams, rho_star: float = 1.0) -> float:
    """Eq. (5) envelope (both loads) at the most restrictive wheelbase, capped at delta_cap_R * R."""
    vals = []
    for L in (rv.L_min, rv.L_max):
        lo, hi = rail_x_bounds(L, rv)
        xm = float(np.clip(xm_star(rho_star, L, theta, rv), lo, hi))
        N_F_nom, _ = loads_from_xm(xm, L, theta, rv)
        vals.append(float(delta_max_both(theta, L, rv, N_F_nom)))
    return min(min(vals), p.delta_cap_R * rv.rail_len)


class CoGController(Protocol):
    name: str
    def xi_ref(self, ps: PhaseState, L: float, theta: float) -> float: ...
    def on_cycle_end(self, feat: dict) -> None: ...


@dataclass
class FixedCentre:
    rv: Rover
    name: str = "fixed"

    def xi_ref(self, ps, L, theta):
        return 0.5 * (self.rv.rail_lo + self.rv.rail_hi)

    def on_cycle_end(self, feat):
        pass


@dataclass
class FixedAt:
    xi: float
    name: str = "fixed_at"

    def xi_ref(self, ps, L, theta):
        return self.xi

    def on_cycle_end(self, feat):
        pass


@dataclass
class ConstantOffsets:
    """Training helper for Stage C: a fixed offset per phase parity."""
    dx_odd: float
    dx_even: float

    def predict(self, feat, parity, theta):
        return self.dx_odd if parity == 1 else self.dx_even


@dataclass
class Feedforward:
    rv: Rover
    rho_star: float = 1.0
    name: str = "feedforward"

    def xi_ref(self, ps, L, theta):
        return feedforward_xi(L, theta, self.rv, self.rho_star)

    def on_cycle_end(self, feat):
        pass


class StaticSetPoint:
    """Stage A: probe one cycle at the centre, classify the terrain, then hold a class set point."""

    name = "stageA"

    def __init__(self, rv: Rover, theta: float, artefact=None):
        self.rv, self.theta, self.artefact = rv, theta, artefact
        self.centre = 0.5 * (rv.rail_lo + rv.rail_hi)
        self.xi = self.centre
        self.cls = None

    def xi_ref(self, ps, L, theta):
        return self.xi

    def on_cycle_end(self, feat):
        if self.artefact is None or self.cls is not None:
            return
        self.cls = int(self.artefact.classify(feat, self.theta))
        self.xi = float(self.artefact.set_point(self.cls, self.theta))


class BoundedResidual:
    """Stage C: feedforward plus a bounded learned offset, piecewise constant per phase type."""

    name = "stageC"

    def __init__(self, rv: Rover, theta: float, p: SimParams, artefact=None, rho_star: float = 1.0):
        self.rv, self.theta, self.artefact, self.rho_star = rv, theta, artefact, rho_star
        self.Delta_max = delta_max_for_run(rv, theta, p, rho_star)
        self.dx = {1: 0.0, 0: 0.0}   # keyed by phase parity (1 = odd/contract, 0 = even/extend)
        self.dx_log: list[float] = []

    def xi_ref(self, ps, L, theta):
        ff = feedforward_xi(L, theta, self.rv, self.rho_star)
        dx = self.dx[ps.phase % 2]
        self.dx_log.append(dx)
        return ff + dx

    def on_cycle_end(self, feat):
        if self.artefact is None or self.Delta_max <= 0:
            return
        for parity in (1, 0):
            g = float(self.artefact.predict(feat, parity, self.theta))
            self.dx[parity] = float(self.Delta_max * np.tanh(g / self.Delta_max))
