"""Parameters of the study with a provenance record for every value.

All quantities are SI (m, kg, s, N, Pa, rad) unless a field name says otherwise.
The source tags are: photo (Fig. 1a of the manuscript, labelled 0/96/192/288 mm rail),
sibling (published rovers HFI24 / SFI25 of the same laboratory), literature,
assumed, manuscript (a design choice stated in the paper), study (design of this evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from math import radians
from typing import Literal, Mapping

import numpy as np

Source = Literal["photo", "sibling", "literature", "assumed", "manuscript", "study"]


@dataclass(frozen=True)
class Prov:
    """Provenance of one parameter."""

    symbol: str
    unit: str
    source: Source
    note: str = ""


@dataclass(frozen=True)
class Rover:
    M: float = 12.0            # total mass
    f: float = 0.3             # movable mass fraction m_m / M (swept)
    f_pair: float = 0.12       # mass of one wheel pair / M
    L_max: float = 0.52        # extended wheelbase
    stroke: float = 0.12       # wheelbase change per phase; L_min = L_max - stroke
    r: float = 0.135           # wheel radius
    b: float = 0.10            # wheel width
    rail_on: Literal["rear", "front"] = "rear"   # pair to which chassis and rail are fixed
    rail_start: float = 0.116  # xi of rail position 0 mm from the rail_on contact; default centres the rail at L_max
    rail_len: float = 0.288    # rail travel
    xi_body: float = 0.26      # body CoG in the rail-fixed frame
    h_body: float = 0.26       # body CoG height above the contact line
    h_rail: float = 0.38       # rail height above the contact line
    N_min_frac: float = 0.10   # N_min = N_min_frac * M g
    g: float = 9.81

    @property
    def m_m(self) -> float:
        return self.f * self.M

    @property
    def m_pair(self) -> float:
        return self.f_pair * self.M

    @property
    def m_body(self) -> float:
        return self.M - self.m_m - 2 * self.m_pair

    @property
    def m_b(self) -> float:
        """Chassis mass without the movable mass (Eq. 1 of the manuscript)."""
        return self.M - self.m_m

    @property
    def L_min(self) -> float:
        return self.L_max - self.stroke

    @property
    def N_min(self) -> float:
        return self.N_min_frac * self.M * self.g

    @property
    def rail_lo(self) -> float:
        return self.rail_start

    @property
    def rail_hi(self) -> float:
        return self.rail_start + self.rail_len

    def with_f(self, f: float) -> "Rover":
        return replace(self, f=f)


@dataclass(frozen=True)
class Soil:
    k_c: float = 0.99e3        # Bekker cohesive modulus, N/m^(n+1)
    k_phi: float = 1528.43e3   # Bekker frictional modulus, N/m^(n+2)
    n: float = 1.10            # sinkage exponent
    c: float = 761.8           # cohesion, Pa
    phi_s: float = radians(22.3)  # internal friction angle
    K: float = 0.01            # shear deformation modulus, m
    slip_ref: float = 0.5      # reference slip at which H_max is evaluated (PPL drives at high slip)
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=lambda: {
        "k_c": (0.5e3, 2.0e3),
        "k_phi": (0.75e6, 3.0e6),
        "n": (0.9, 1.3),
        "c": (380.0, 1140.0),
        "phi_s": (radians(20.0), radians(30.0)),
        "K": (0.001, 0.025),
    })
    log_uniform: frozenset = frozenset({"k_c", "k_phi", "K"})


@dataclass(frozen=True)
class Actuator:
    tau: float = 0.3           # first-order tracking time constant
    v_max: float = 0.30        # rail speed limit
    a_max: float = 2.0         # rail acceleration limit
    T_phase: float = 2.0       # duration of one PPL phase
    dt: float = 2e-3           # integration step
    Kp: float = 0.10           # PI gain on the load fraction error, m per unit
    Ki: float = 0.20           # PI integral gain, m per unit per s
    mismatch: float = 0.03     # actual body CoG offset relative to nominal, fraction of L (plant mismatch for PI)


@dataclass(frozen=True)
class Study:
    f_list: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)
    f_ref: float = 0.3
    theta_deg: np.ndarray = field(default_factory=lambda: np.arange(0.0, 35.01, 0.5))
    theta_ref_deg: float = 20.0
    rho_star: float = 1.0
    h_rail_over_L: tuple[float, ...] = (0.5, 0.7, 0.9)
    seed: int = 20261001
    n_mc: int = 200
    n_cycles: int = 3
    theta_mc_deg: np.ndarray = field(default_factory=lambda: np.arange(0.0, 30.01, 2.5))
    rail_start_sweep: tuple[float, ...] = (0.05, 0.116, 0.182)   # rail zero at rear axle / centred / rail end at front axle
    # geometric uncertainty of the nominal rover that the learned residual must cover
    geom_ranges: Mapping[str, tuple[float, float]] = field(default_factory=lambda: {
        "h_rail": (0.8 * 0.38, 1.2 * 0.38),
        "xi_body": (0.26 - 0.03 * 0.52, 0.26 + 0.03 * 0.52),
        "f_pair": (0.08, 0.16),
    })


# ---------------------------------------------------------------------------
# Provenance registry: one entry per dataclass field.

PROVENANCE: dict[str, Prov] = {
    # Rover
    "Rover.M": Prov("M", "kg", "sibling", "SFI25 rover 120.5 N (12.3 kg); illustrative only"),
    "Rover.f": Prov(r"f = m_m/M", "-", "manuscript", "'substantial fraction'; swept 0.2 to 0.5"),
    "Rover.f_pair": Prov(r"m_{pair}/M", "-", "assumed", "each wheel pair with motors"),
    "Rover.L_max": Prov(r"L_{max}", "m", "photo", "wheel-centre spacing from the Fig. 1a scale, +-20 %"),
    "Rover.stroke": Prov(r"L_{max}-L_{min}", "m", "sibling", "HFI24 120 mm; SFI25 80 mm"),
    "Rover.r": Prov("r", "m", "photo", "wheel diameter 250 to 300 mm from the Fig. 1a scale"),
    "Rover.b": Prov("b", "m", "assumed", "wheel width, not visible in the side view"),
    "Rover.rail_on": Prov("rail frame", "-", "assumed", "chassis and rail fixed to this pair; both evaluated"),
    "Rover.rail_start": Prov(r"\xi_{rail,0}", "m", "assumed", "rail centred on the chassis at L_max; photo shows the rail zero near one axle, swept 0.05 to 0.182"),
    "Rover.rail_len": Prov("R", "m", "photo", "labelled 0, 96, 192, 288 mm"),
    "Rover.xi_body": Prov(r"\xi_b", "m", "assumed", "body CoG at mid-chassis at L_max"),
    "Rover.h_body": Prov("h_b", "m", "assumed", "about wheel top height"),
    "Rover.h_rail": Prov("h_{rail}", "m", "photo", "+-20 %; sensitivity sweep 0.5 to 0.9 L"),
    "Rover.N_min_frac": Prov(r"N_{min}/(Mg)", "-", "manuscript", "design choice for the stability margin"),
    "Rover.g": Prov("g", "m/s^2", "literature", ""),
    # Soil
    "Soil.k_c": Prov("k_c", "N/m^(n+1)", "literature", "Wong 2008 dry sand; fallback, not measured for silica sand No. 5"),
    "Soil.k_phi": Prov(r"k_\phi", "N/m^(n+2)", "literature", "Wong 2008 dry sand; fallback"),
    "Soil.n": Prov("n", "-", "literature", "Wong 2008 dry sand; fallback"),
    "Soil.c": Prov("c", "Pa", "literature", "silica sand No. 5, SFI25 Table 4 (unit printed as N/m3, read as Pa)"),
    "Soil.phi_s": Prov(r"\phi_s", "rad", "literature", "silica sand No. 5, 22.3 deg, SFI25 Table 4 after Matsumoto 2013"),
    "Soil.K": Prov("K", "m", "literature", "Wong 2008 range 0.001 to 0.025 m as quoted in SFI25 Table 4, log-midpoint"),
    "Soil.slip_ref": Prov("s_{ref}", "-", "assumed", "slip at which H_max is evaluated"),
    # Actuator
    "Actuator.tau": Prov(r"\tau", "s", "assumed", "stepper position loop"),
    "Actuator.v_max": Prov("v_{max}", "m/s", "assumed", "mass crosses the rail within one phase"),
    "Actuator.a_max": Prov("a_{max}", "m/s^2", "assumed", ""),
    "Actuator.T_phase": Prov("T_{phase}", "s", "assumed", ""),
    "Actuator.dt": Prov(r"\Delta t", "s", "study", "integration step"),
    "Actuator.Kp": Prov("K_p", "m", "assumed", "PI on the load fraction"),
    "Actuator.Ki": Prov("K_i", "m/s", "assumed", "PI on the load fraction"),
    "Actuator.mismatch": Prov(r"\delta x_b/L", "-", "study", "plant mismatch that gives the PI a task"),
    # Study
    "Study.rho_star": Prov(r"\rho^\ast", "-", "manuscript", "target load ratio"),
    "Study.theta_ref_deg": Prov(r"\theta_{ref}", "deg", "study", ""),
    "Study.seed": Prov("seed", "-", "study", "Monte Carlo seed"),
    "Study.n_mc": Prov("n_{MC}", "-", "study", "soil samples"),
    "Study.n_cycles": Prov("cycles", "-", "study", "simulated PPL cycles"),
    "Study.n_mc": Prov("n_{MC}", "-", "study", "soil and geometry samples"),
}


def provenance_rows(rv: Rover, soil: Soil, act: Actuator, st: Study) -> list[dict]:
    """Rows for the parameter table, values read from the live dataclass instances."""
    rows = []
    for obj, name in ((rv, "Rover"), (soil, "Soil"), (act, "Actuator"), (st, "Study")):
        for fld in fields(obj):
            key = f"{name}.{fld.name}"
            if key not in PROVENANCE:
                continue
            p = PROVENANCE[key]
            val = getattr(obj, fld.name)
            rows.append({"key": key, "symbol": p.symbol, "value": val, "unit": p.unit,
                         "source": p.source, "note": p.note})
    return rows
