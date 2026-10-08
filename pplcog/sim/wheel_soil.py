"""Wong-Reece stress integration for a driven rigid wheel, tabulated per soil.

References: Wong and Reece (1967), Ishigami et al. (2007), Wong (2008) ch. 2.
Per wheel of radius r and width b under vertical load W at slip s:

    theta_f = arccos(1 - z/r),  theta_r = -arccos(1 - lam z/r),  theta_m = (a0 + a1 s) theta_f
    sigma(theta) = (k_c/b + k_phi) r^n (cos theta - cos theta_f)^n                  theta in [theta_m, theta_f]
    sigma(theta) = (k_c/b + k_phi) r^n [cos(theta_f - (theta - theta_r)(theta_f - theta_m)/(theta_m - theta_r)) - cos theta_f]^n
    j(theta)  = r [(theta_f - theta) - (1 - s)(sin theta_f - sin theta)]            drive, s >= 0
    j(theta)  = r [(theta_f - theta) - (sin theta_f - sin theta)/(1 + s)]           skid,  s <  0
    tau = (c + sigma tan phi)(1 - exp(-j/K))
    W  = r b int (sigma cos + tau sin) dtheta
    DP = r b int (tau cos - sigma sin) dtheta
    T  = r^2 b int tau dtheta

The tables are built once per soil on a (z, s) grid and inverted per step for the sinkage
at which W equals the load.  Slip-sinkage and multipass effects are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..params import Rover
from .params_sim import SimParams, SoilWR


def bearing_factors(phi: float) -> tuple[float, float, float]:
    """Prandtl-Reissner / Vesic bearing capacity factors (N_c, N_q, N_gamma)."""
    if phi < 1e-9:
        return 2.0 + np.pi, 1.0, 0.0
    N_q = np.exp(np.pi * np.tan(phi)) * np.tan(np.pi / 4 + phi / 2) ** 2
    N_c = (N_q - 1.0) / np.tan(phi)
    N_g = 2.0 * (N_q + 1.0) * np.tan(phi)
    return float(N_c), float(N_q), float(N_g)


def bulldozing_resistance(z, soil: SoilWR, rv: Rover):
    """Bekker bulldozing resistance of a wheel of width b sunk to depth z (Wong 2008)."""
    phi = soil.phi_s
    N_c, _, N_g = bearing_factors(phi)
    if phi < 1e-9:
        K_pc, K_pg = N_c, 0.0
    else:
        K_pc = (N_c - np.tan(phi)) * np.cos(phi) ** 2
        K_pg = (2.0 * N_g / np.tan(phi) + 1.0) * np.cos(phi) ** 2
    z = np.asarray(z, dtype=float)
    return rv.b * (soil.c * z * K_pc + 0.5 * soil.gamma_s * z ** 2 * K_pg)


def _gl(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def wheel_forces_grid(z, s, soil: SoilWR, rv: Rover, n_quad: int = 30):
    """Vectorised W, DP, T on broadcast arrays z and s (per wheel)."""
    z = np.asarray(z, dtype=float)[..., None]
    s = np.asarray(s, dtype=float)[..., None]
    r, b = rv.r, rv.b
    kk = soil.k_c / b + soil.k_phi
    th_f = np.arccos(np.clip(1.0 - z / r, -1.0, 1.0))
    th_r = -np.arccos(np.clip(1.0 - soil.lam * z / r, -1.0, 1.0))
    th_m = (soil.a0 + soil.a1 * np.clip(s, 0.0, 1.0)) * th_f
    x, w = _gl(n_quad)
    # front segment [th_m, th_f] and rear segment [th_r, th_m]
    thA = 0.5 * (th_f - th_m) * x + 0.5 * (th_f + th_m); wA = 0.5 * (th_f - th_m) * w
    thB = 0.5 * (th_m - th_r) * x + 0.5 * (th_m + th_r); wB = 0.5 * (th_m - th_r) * w
    sigA = kk * r ** soil.n * np.clip(np.cos(thA) - np.cos(th_f), 0.0, None) ** soil.n
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = th_f - (thB - th_r) * (th_f - th_m) / np.where(th_m - th_r > 0, th_m - th_r, 1.0)
    sigB = kk * r ** soil.n * np.clip(np.cos(arg) - np.cos(th_f), 0.0, None) ** soil.n

    def shear(th, sig):
        drive = s >= 0
        jd = r * ((th_f - th) - (1.0 - s) * (np.sin(th_f) - np.sin(th)))
        js = r * ((th_f - th) - (np.sin(th_f) - np.sin(th)) / (1.0 + np.clip(s, -0.999, 0.0)))
        j = np.where(drive, jd, js)
        mag = (soil.c + sig * np.tan(soil.phi_s)) * (1.0 - np.exp(-np.abs(j) / soil.K))
        return np.sign(j) * mag

    tauA, tauB = shear(thA, sigA), shear(thB, sigB)
    W = r * b * (np.sum((sigA * np.cos(thA) + tauA * np.sin(thA)) * wA, -1)
                 + np.sum((sigB * np.cos(thB) + tauB * np.sin(thB)) * wB, -1))
    DP = r * b * (np.sum((tauA * np.cos(thA) - sigA * np.sin(thA)) * wA, -1)
                  + np.sum((tauB * np.cos(thB) - sigB * np.sin(thB)) * wB, -1))
    T = r * r * b * (np.sum(tauA * wA, -1) + np.sum(tauB * wB, -1))
    return W, DP, T


@dataclass
class WheelTable:
    """W, DP, T on a (z, s) grid for one soil and one wheel geometry, with lookups."""

    z: np.ndarray
    s: np.ndarray
    W: np.ndarray
    DP: np.ndarray
    T: np.ndarray
    soil: SoilWR
    rv: Rover

    @classmethod
    def build(cls, soil: SoilWR, rv: Rover, p: SimParams) -> "WheelTable":
        z = (np.linspace(0.0, 1.0, p.n_z) ** 2) * p.z_max_frac * rv.r
        s = np.linspace(-1.0, 1.0, p.n_s)
        Z, S = np.meshgrid(z, s, indexing="ij")
        W, DP, T = wheel_forces_grid(Z, S, soil, rv, p.n_quad)
        W = np.maximum.accumulate(W, axis=0)  # enforce monotonicity against quadrature noise
        return cls(z, s, W, DP, T, soil, rv)

    def _s_index(self, s: float):
        s = float(np.clip(s, -1.0, 1.0))
        k = int(np.clip(np.searchsorted(self.s, s) - 1, 0, self.s.size - 2))
        f = (s - self.s[k]) / (self.s[k + 1] - self.s[k])
        return k, f

    def sinkage(self, W_half: float, s: float) -> float:
        """Sinkage at which the vertical force equals W_half (per wheel)."""
        k, f = self._s_index(s)
        Wcol = (1 - f) * self.W[:, k] + f * self.W[:, k + 1]
        return float(np.interp(W_half, Wcol, self.z, left=0.0, right=self.z[-1]))

    def lookup(self, W_half: float, s: float):
        """(z, DP, T, dDP/ds) for a wheel at load W_half and slip s."""
        k, f = self._s_index(s)
        z = self.sinkage(W_half, s)
        def col(A, kk):
            return float(np.interp(z, self.z, A[:, kk]))
        DP = (1 - f) * col(self.DP, k) + f * col(self.DP, k + 1)
        T = (1 - f) * col(self.T, k) + f * col(self.T, k + 1)
        ds = self.s[k + 1] - self.s[k]
        dDP = (col(self.DP, k + 1) - col(self.DP, k)) / ds
        return z, DP, T, dDP

    def slip_at_torque(self, W_half: float, T_lim: float, s_hi: float) -> float:
        """Largest slip in [0, s_hi] at which the torque does not exceed T_lim.

        The torque is evaluated at the sinkage that carries W_half at each slip, not at the sinkage
        of s_hi: the sinkage grows with slip, so a single sinkage overstates the torque at low slip
        and returns a slip whose thrust is far below the torque-limited value."""
        mask = (self.s >= 0) & (self.s <= s_hi)
        ss = self.s[mask]
        TT = np.array([self.lookup(W_half, s)[2] for s in ss])
        if TT[-1] <= T_lim:
            return s_hi
        TT = np.maximum.accumulate(TT)  # monotone for the inversion
        return float(np.interp(T_lim, TT, ss))

    def traction_index(self, W_half: float, s: float = 0.3) -> float:
        """Drawbar pull over load at a reference slip; used to stratify soils."""
        _, DP, _, _ = self.lookup(W_half, s)
        return DP / W_half
