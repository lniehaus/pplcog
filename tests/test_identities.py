import numpy as np
import pytest
from dataclasses import replace

from pplcog.params import Actuator, Rover, Soil, Study
from pplcog import frame, statics as st, terramechanics as tm
from pplcog.actuator import simulate

RNG = np.random.default_rng(0)


def random_cases(n=200):
    f = RNG.uniform(0.15, 0.55, n)
    L = RNG.uniform(0.35, 0.6, n)
    th = RNG.uniform(-0.3, 0.6, n)
    rho = RNG.uniform(0.5, 2.0, n)
    return f, L, th, rho


def test_load_sum_identity():
    f, L, th, _ = random_cases()
    for fi, Li, ti in zip(f, L, th):
        rv = Rover(f=fi)
        x_m = RNG.uniform(0.0, Li)
        N_F, N_R = st.loads_from_xm(x_m, Li, ti, rv)
        np.testing.assert_allclose(N_F + N_R, rv.M * rv.g * np.cos(ti))


def test_setpoint_round_trip():
    f, L, th, rho = random_cases()
    for fi, Li, ti, ri in zip(f, L, th, rho):
        rv = Rover(f=fi)
        xm = st.xm_star(ri, Li, ti, rv)
        N_F, N_R = st.loads_from_xm(xm, Li, ti, rv)
        np.testing.assert_allclose(st.load_ratio(N_F, N_R), ri, rtol=1e-9)


def test_frame_round_trip_and_physical_consistency():
    for rail_on in ("rear", "front"):
        rv = Rover(rail_on=rail_on)
        xi = RNG.uniform(rv.rail_lo, rv.rail_hi, 50)
        L = RNG.uniform(rv.L_min, rv.L_max, 50)
        np.testing.assert_allclose(frame.x_to_xi(frame.xi_to_x(xi, L, rv), L, rv), xi)
    # the same physical placement described in both frames gives the same loads
    L, th = 0.45, 0.2
    rear = Rover(rail_on="rear")
    front = Rover(rail_on="front", xi_body=L - rear.xi_body)  # mirror the body CoG at this L
    x_m = 0.3
    NF_r = st.loads_from_xm(x_m, L, th, rear)[0]
    NF_f = st.loads_from_xm(x_m, L, th, front)[0]
    np.testing.assert_allclose(NF_r, NF_f)


def test_dimensionless_invariance():
    rv = Rover()
    lam, kap = 2.7, 0.4
    scaled = replace(rv, M=kap * rv.M, L_max=lam * rv.L_max, stroke=lam * rv.stroke, r=lam * rv.r,
                     b=lam * rv.b, rail_start=lam * rv.rail_start, rail_len=lam * rv.rail_len,
                     xi_body=lam * rv.xi_body, h_body=lam * rv.h_body, h_rail=lam * rv.h_rail)
    for th in (0.0, 0.2, 0.4):
        for xi in np.linspace(rv.rail_lo, rv.rail_hi, 7):
            for L in (rv.L_min, rv.L_max):
                N1 = st.loads_from_xm(frame.xi_to_x(xi, L, rv), L, th, rv)
                N2 = st.loads_from_xm(frame.xi_to_x(lam * xi, lam * L, scaled), lam * L, th, scaled)
                np.testing.assert_allclose(st.load_ratio(*N1), st.load_ratio(*N2))
                np.testing.assert_allclose(N1[0] / (rv.M * rv.g), N2[0] / (scaled.M * scaled.g))
    for L in (rv.L_min, rv.L_max):
        np.testing.assert_allclose(st.slope_limit(rv, L, 1.0), st.slope_limit(scaled, lam * L, 1.0), atol=1e-5)


def test_dNF_dL_sign():
    assert st.dNF_dL_sign(Rover(rail_on="rear")) == -1
    assert st.dNF_dL_sign(Rover(rail_on="front")) == 1


@pytest.mark.parametrize("rail_on", ["rear", "front"])
@pytest.mark.parametrize("kind", ["square", "const"])
def test_bound_holds_for_feedforward(rail_on, kind):
    rv = Rover(rail_on=rail_on)
    act = Actuator(mismatch=0.0)
    study = Study(n_cycles=2)
    theta = np.radians(8.0)
    nom = simulate(rv, act, study, theta, "ff", offset=False, Delta_max=0.05)
    off = simulate(rv, act, study, theta, "ff", offset=True, offset_kind=kind, Delta_max=0.05)
    assert off.viol_ratio(nom, rv, theta) <= 1.0 + 1e-6


def test_terramechanics_sanity():
    rv, soil = Rover(), Soil()
    W = np.linspace(5.0, 60.0, 12)
    H = tm.thrust_max(W, soil, rv)
    assert np.all(np.diff(H) > 0)
    z = tm.sinkage(rv.M * rv.g / 4, soil, rv)
    assert 0 < z < rv.r
    # with vanishing resistance the utilisation tends to M g sin(theta) / H_max of the driving pair
    stiff = replace(soil, k_c=1e14, k_phi=1e14)
    th = 0.2
    x_m = 0.26
    mu_d, _ = tm.utilisation(x_m, rv.L_max, th, 1, stiff, rv)
    N_F, N_R = st.loads_from_xm(x_m, rv.L_max, th, rv)
    H_d = 2 * tm.thrust_max(N_R / 2, stiff, rv)
    np.testing.assert_allclose(mu_d, rv.M * rv.g * np.sin(th) / H_d, rtol=1e-2)


def test_utilisation_optimum_equals_unit_load_ratio():
    rv, soil = Rover(), Soil()
    th = np.radians(5.0)
    for ph in (1, 2):
        xo = tm.xm_opt(rv.L_max, th, ph, soil, rv)
        np.testing.assert_allclose(xo, st.xm_star(1.0, rv.L_max, th, rv), atol=1e-3)
