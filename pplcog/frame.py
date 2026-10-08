"""Reference frames, PPL phases, and geometry that resolve review items (a) and (b).

Two longitudinal coordinates are used.

* x: the manuscript's frame, measured from the rear contact point up the chassis
  axis; the front contact is at x = L.
* xi: a body-fixed coordinate measured from the contact point of the wheel pair to
  which the chassis and the rail are fixed (``Rover.rail_on``).  The rail limits and
  the body CoG are constant in xi.

For ``rail_on == 'rear'`` the two coincide, x = xi.  For ``rail_on == 'front'`` the
chassis moves with the front pair and x = L - xi.  The height h of the total CoG does
not depend on the mass position because the rail is parallel to the chassis axis.

Phase convention (Fig. 2 of the manuscript): odd phases contract the wheelbase from
L_max to L_min with the front pair anchored and the rear pair driving; even phases
extend it from L_min to L_max with the rear pair anchored and the front pair driving.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .params import Rover

Pair = Literal["front", "rear"]


def wheelbase(phase: int | np.ndarray, s: float | np.ndarray, rv: Rover) -> np.ndarray:
    """Wheelbase at progress s in [0, 1] through the given phase (1..4)."""
    phase = np.asarray(phase)
    s = np.asarray(s, dtype=float)
    contract = (phase % 2) == 1
    return np.where(contract, rv.L_max - s * rv.stroke, rv.L_min + s * rv.stroke)


def anchor(phase: int) -> Pair:
    return "front" if phase % 2 == 1 else "rear"


def driver(phase: int) -> Pair:
    return "rear" if phase % 2 == 1 else "front"


def xi_to_x(xi, L, rv: Rover):
    xi = np.asarray(xi, dtype=float)
    L = np.asarray(L, dtype=float)
    return xi if rv.rail_on == "rear" else L - xi


def x_to_xi(x, L, rv: Rover):
    x = np.asarray(x, dtype=float)
    L = np.asarray(L, dtype=float)
    return x if rv.rail_on == "rear" else L - x


def x_body(L, rv: Rover):
    """CoG of the chassis without the movable mass (m_b in Eq. 1), in the x frame.

    The chassis is a body of mass m_body at xi_body (rail fixed) plus one wheel pair
    at x = 0 and one at x = L, so this position depends on the wheelbase.
    """
    L = np.asarray(L, dtype=float)
    return (rv.m_pair * L + rv.m_body * xi_to_x(rv.xi_body, L, rv)) / rv.m_b


def h_cog(rv: Rover) -> float:
    """Height of the total CoG; a function of f, not of the mass position."""
    return (rv.m_body * rv.h_body + 2 * rv.m_pair * rv.r + rv.m_m * rv.h_rail) / rv.M


def rail_x_bounds(L, rv: Rover) -> tuple[np.ndarray, np.ndarray]:
    """Rail limits mapped to the x frame, as (low, high)."""
    a = xi_to_x(rv.rail_lo, L, rv)
    b = xi_to_x(rv.rail_hi, L, rv)
    return np.minimum(a, b), np.maximum(a, b)


def phase_timeline(T_phase: float, dt: float, n_cycles: int):
    """Time vector with the phase index (1..4) and the progress s within the phase."""
    t = np.arange(0.0, 4 * T_phase * n_cycles, dt)
    k = np.floor(t / T_phase).astype(int)
    phase = 1 + (k % 4)
    s = (t - k * T_phase) / T_phase
    return t, phase, s
