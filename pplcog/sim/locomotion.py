"""PPL phase machine, wheelbase reference profile, and wheel speed commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..frame import anchor, driver
from ..params import Actuator, Rover
from .params_sim import SimParams

Mode = Literal["ppl", "wheel"]


@dataclass
class PhaseState:
    phase: int          # 1..4
    cycle: int          # 0-based
    s: float            # progress in the phase, 0..1
    L_ref: float
    Ldot_ref: float
    anchor: str | None  # 'front' | 'rear' | None (wheel mode)
    driver: str | None  # 'front' | 'rear' | None (both driven in wheel mode)
    boundary: bool      # first step of a new phase


def phase_at(t: float, T_phase: float) -> tuple[int, int, float]:
    k = int(np.floor(t / T_phase + 1e-12))
    return 1 + k % 4, k // 4, (t - k * T_phase) / T_phase


def wheelbase_profile(phase: int, s: float, rv: Rover, T_phase: float, p: SimParams) -> tuple[float, float]:
    """Cosine wheelbase profile and its rate for the phase; odd contracts, even extends."""
    frac = 0.5 * (1.0 - np.cos(np.pi * s))
    dfrac = 0.5 * np.pi * np.sin(np.pi * s) / T_phase
    if phase % 2 == 1:
        L = rv.L_max - rv.stroke * frac
        Ld = -rv.stroke * dfrac
    else:
        L = rv.L_min + rv.stroke * frac
        Ld = rv.stroke * dfrac
    Ld = float(np.clip(Ld, -p.vL_max, p.vL_max))
    return float(L), Ld


class PhaseMachine:
    def __init__(self, mode: Mode, rv: Rover, act: Actuator, p: SimParams):
        self.mode, self.rv, self.act, self.p = mode, rv, act, p
        self._last_phase = None
        self.v_nom = 2 * rv.stroke / (4 * act.T_phase)   # PPL-equivalent mean speed for wheel mode

    def state(self, t: float) -> PhaseState:
        phase, cycle, s = phase_at(t, self.act.T_phase)
        boundary = phase != self._last_phase
        self._last_phase = phase
        if self.mode == "ppl":
            L_ref, Ld_ref = wheelbase_profile(phase, s, self.rv, self.act.T_phase, self.p)
            return PhaseState(phase, cycle, s, L_ref, Ld_ref, anchor(phase), driver(phase), boundary)
        return PhaseState(phase, cycle, s, self.rv.L_max, 0.0, None, None, boundary)

    def omega_ref(self, ps: PhaseState) -> float:
        """Driven wheel angular speed so that the nominal advance equals the stroke per phase."""
        if self.mode == "ppl":
            v_nom = abs(ps.Ldot_ref)
        else:
            v_nom = self.v_nom
        return v_nom / (self.rv.r * (1.0 - self.p.s_ref))
