import numpy as np
from dataclasses import replace

from pplcog.params import Rover, Soil
from pplcog import terramechanics as tm
from pplcog.sim.params_sim import SimParams, SoilWR
from pplcog.sim.wheel_soil import WheelTable, bearing_factors, wheel_forces_grid

RV, P = Rover(), SimParams()


def test_sinkage_inversion_matches_load():
    rng = np.random.default_rng(3)
    tab = WheelTable.build(SoilWR(), RV, P)
    for _ in range(30):
        W = rng.uniform(5.0, 80.0)
        s = rng.uniform(-0.8, 1.0)
        z = tab.sinkage(W, s)
        Wz, _, _ = wheel_forces_grid(np.array([z]), np.array([s]), SoilWR(), RV, P.n_quad)
        np.testing.assert_allclose(Wz[0], W, rtol=2e-2)


def test_shapes_and_monotonicity():
    tab = WheelTable.build(SoilWR(), RV, P)
    assert np.all(np.diff(tab.W, axis=0) >= 0)
    W = RV.M * RV.g / 4
    s = np.linspace(0.0, 1.0, 21)
    DP = np.array([tab.lookup(W, si)[1] for si in s])
    T = np.array([tab.lookup(W, si)[2] for si in s])
    assert DP[0] < 0                      # pure rolling resistance at zero slip
    assert T[0] >= 0 and T[0] < 0.2 * T[-1]
    assert np.all(np.diff(DP) > -1e-6) and np.all(np.diff(T) > -1e-6)
    # concave, saturating drawbar pull
    assert DP[5] - DP[0] > DP[-1] - DP[-6]


def test_bearing_factors_prandtl():
    N_c, N_q, N_g = bearing_factors(0.0)
    np.testing.assert_allclose(N_c, 2 + np.pi)
    assert N_q == 1.0 and N_g == 0.0


def test_magnitudes_against_bekker():
    tab = WheelTable.build(SoilWR(), RV, P)
    W = RV.M * RV.g / 4
    z0, DP0, _, _ = tab.lookup(W, 0.0)
    zb = tm.sinkage(W, Soil(), RV)
    Rc = tm.compaction_resistance(W, Soil(), RV)
    assert 0.5 < z0 / zb < 2.0
    assert 0.5 < -DP0 / Rc < 2.0
    _, DP1, _, _ = tab.lookup(W, 1.0)
    Hmax = tm.thrust_max(W, replace(Soil(), slip_ref=1.0), RV)
    assert 0.5 < DP1 / (Hmax - Rc) < 2.0
