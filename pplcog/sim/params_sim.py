"""Parameters of the closed-loop simulation study, with provenance for every value.

`SoilWR` composes the quasi-static `Soil` with the extra Wong-Reece parameters; the
base `Soil.ranges` are left untouched so that the quasi-static study keeps its random
stream and its results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from math import radians
from typing import Mapping

import numpy as np

from ..params import Prov, Soil

G = 9.81


@dataclass(frozen=True)
class SoilWR:
    """Wong-Reece wheel-soil parameters on top of the Bekker / Mohr-Coulomb set."""

    base: Soil = field(default_factory=Soil)
    a0: float = 0.40           # max-stress angle coefficient (theta_m = (a0 + a1 s) theta_f)
    a1: float = 0.15
    lam: float = 0.1           # rear exit sinkage ratio; near zero for dry sand without rebound
    gamma_s: float = 1300.0 * G  # soil unit weight, N/m^3 (bulk density 1.29 to 1.31 g/cm^3)
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=lambda: {
        "a0": (0.2, 0.45), "a1": (0.1, 0.35), "lam": (0.0, 0.3),
        "gamma_s": (1250.0 * G, 1350.0 * G)})
    K_range: tuple[float, float] = (0.010, 0.025)   # shear deformation modulus of sand, 10 to 25 mm (Wong 2008, sec. 2.4.3); overrides the base draw

    # convenience pass-through
    @property
    def k_c(self): return self.base.k_c
    @property
    def k_phi(self): return self.base.k_phi
    @property
    def n(self): return self.base.n
    @property
    def c(self): return self.base.c
    @property
    def phi_s(self): return self.base.phi_s
    @property
    def K(self): return self.base.K

    def sample(self, rng: np.random.Generator) -> "SoilWR":
        """One draw of the Wong-Reece extras and of K; the other base parameters are drawn elsewhere."""
        d = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.ranges.items()}
        K = float(rng.uniform(*self.K_range))
        return replace(self, base=replace(self.base, K=K), **d)


@dataclass(frozen=True)
class SimParams:
    dt: float = 1e-3
    n_cycles: int = 10
    s_ref: float = 0.15        # commanded slip of the driven pair
    Kp_L: float = 4000.0       # wheelbase actuator PD, N/m
    Kd_L: float = 120.0        # N s/m
    F_act_max: float = 120.0   # N
    vL_max: float = 0.15       # wheelbase rate limit, m/s
    T_max: float = 3.0         # motor torque limit per wheel, N m
    zeta_anchor: float = 0.7   # anchor damping ratio
    v_eps: float = 1e-3        # slip denominator floor, m/s
    tau_a: float = 0.05        # low-pass time constant of the chassis acceleration in the load transfer, s
    v_slide: float = 0.05      # backward speed that counts as sliding, m/s
    t_slide: float = 0.5       # duration of backward sliding that fails a run, s
    eta_w: float = 0.7         # wheel drive efficiency
    eta_a: float = 0.7         # wheelbase actuator efficiency
    eta_r: float = 0.7         # rail actuator efficiency
    stall_slip: float = 0.95
    stall_adv_frac: float = 0.05
    pitch_limit: float = radians(15.0)
    delta_cap_R: float = 0.5   # cap on Delta_max in rail lengths
    noise: Mapping[str, float] = field(default_factory=lambda: {
        "slip": 0.02, "torque_rel": 0.02, "torque_abs": 0.02, "force": 0.5, "pitch": radians(0.2)})
    n_quad: int = 30           # Gauss-Legendre nodes per contact segment
    n_z: int = 64
    n_s: int = 81
    z_max_frac: float = 0.6    # table sinkage limit, fraction of r
    seed: int = 20261002
    n_soils: int = 60
    n_train: int = 40
    n_rep: int = 5
    slopes_deg: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    steep_lo: float = 0.0
    steep_hi: float = 40.0
    steep_tol: float = 0.5
    n_classes: int = 3
    grid_A: int = 9            # rail positions in the Stage A grid search
    grid_C: int = 5            # offsets per phase type in the Stage C grid search
    train_cycles: int = 3
    gp_ls_min: float = 0.5     # lower bound of the Stage C RBF length scales, in standard deviations of each input


PROVENANCE_SIM: dict[str, Prov] = {
    "SoilWR.a0": Prov("a_0", "-", "literature", "Wong and Reece 1967: 0.18 (loose) to 0.43 (compact) sand; sampled 0.2 to 0.45, assumed"),
    "SoilWR.a1": Prov("a_1", "-", "literature", "Ishigami et al. 2007: 0 to 0.3; Wong and Reece 1967 report 0.32; sampled 0.1 to 0.35, assumed"),
    "SoilWR.lam": Prov(r"\lambda", "-", "assumed", "rear exit sinkage ratio"),
    "SoilWR.gamma_s": Prov(r"\gamma_s", "N/m^3", "literature", "converted from 1.29 to 1.31 g/cm3, silica sand No. 5, SFI25 Table 4"),
    "SoilWR.K_range": Prov("K", "m", "literature", "shear deformation modulus of sand, 10 to 25 mm (Wong 2008); clay about 6 mm"),
    "SimParams.dt": Prov(r"\Delta t", "s", "study", "integration step"),
    "SimParams.n_cycles": Prov("cycles", "-", "manuscript", "ten cycles per run, validation protocol"),
    "SimParams.s_ref": Prov("s_{ref}", "-", "assumed", "commanded slip of the driven pair"),
    "SimParams.Kp_L": Prov("K_{p,L}", "N/m", "assumed", "wheelbase actuator position loop"),
    "SimParams.Kd_L": Prov("K_{d,L}", "N s/m", "assumed", ""),
    "SimParams.F_act_max": Prov("F_{act,max}", "N", "assumed", "wheelbase actuator force limit"),
    "SimParams.vL_max": Prov(r"\dot L_{max}", "m/s", "assumed", "wheelbase rate limit"),
    "SimParams.T_max": Prov("T_{max}", "N m", "assumed", "motor torque limit per wheel"),
    "SimParams.zeta_anchor": Prov(r"\zeta_a", "-", "assumed", "anchor damping ratio"),
    "SimParams.tau_a": Prov(r"\tau_a", "s", "assumed", "load-transfer acceleration filter"),
    "SimParams.v_slide": Prov("v_{slide}", "m/s", "study", "backward sliding criterion"),
    "SimParams.eta_w": Prov(r"\eta_w", "-", "assumed", "drive efficiency"),
    "SimParams.eta_a": Prov(r"\eta_a", "-", "assumed", "wheelbase actuator efficiency"),
    "SimParams.eta_r": Prov(r"\eta_r", "-", "assumed", "rail actuator efficiency"),
    "SimParams.stall_slip": Prov("s_{stall}", "-", "study", "stall criterion"),
    "SimParams.pitch_limit": Prov(r"\psi_{max}", "rad", "study", "pitch failure criterion"),
    "SimParams.delta_cap_R": Prov(r"\Delta_{max}/R\ \mathrm{cap}", "-", "assumed", "design-time cap on the envelope"),
    "SimParams.seed": Prov("seed", "-", "study", ""),
    "SimParams.n_soils": Prov("n_{soil}", "-", "study", "soil population"),
    "SimParams.n_train": Prov("n_{train}", "-", "study", "training soils"),
    "SimParams.n_rep": Prov("n_{rep}", "-", "manuscript", "repetitions per condition, validation protocol"),
    "SimParams.n_classes": Prov("classes", "-", "manuscript", "terrain classes for Stage A"),
    "SimParams.gp_ls_min": Prov(r"\ell_{min}", "-", "study", "lower bound of the Stage C Gaussian-process length scales, in standard deviations of each input"),
}


def provenance_rows_sim(soil: SoilWR, p: SimParams) -> list[dict]:
    rows = []
    for obj, name in ((soil, "SoilWR"), (p, "SimParams")):
        for fld in fields(obj):
            key = f"{name}.{fld.name}"
            if key in PROVENANCE_SIM:
                pr = PROVENANCE_SIM[key]
                rows.append({"key": key, "symbol": pr.symbol, "value": getattr(obj, fld.name),
                             "unit": pr.unit, "source": pr.source, "note": pr.note})
    return rows
