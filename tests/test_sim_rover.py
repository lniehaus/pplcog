import numpy as np
import pytest

from pplcog.params import Actuator, Rover
from pplcog.sim.params_sim import SimParams, SoilWR
from pplcog.sim.simulate import run_episode
from pplcog.sim.wheel_soil import WheelTable

RV, ACT, P, SOIL = Rover(), Actuator(), SimParams(), SoilWR()
TAB = WheelTable.build(SOIL, RV, P)


def run(cond, theta_deg, n_cycles=3, **kw):
    return run_episode(RV, ACT, P, SOIL, np.radians(theta_deg), cond, n_cycles=n_cycles, tab=TAB, **kw)


def test_level_ground_advance_and_anchor():
    r = run("B2", 0.0)
    assert r.completed
    c = r.cycles[-1]
    # the anchor never saturates on level ground and the advance is a large fraction of 2 strokes
    assert c["slide_frac_time"] < 0.01
    assert 0.7 < c["travel_eff"] <= 1.02


def test_travel_efficiency_decreases_with_slope():
    eff = []
    for th in (0.0, 10.0, 20.0):
        r = run("B2", th)
        assert r.completed
        eff.append(r.cycles[-1]["travel_eff"])
    assert eff[0] > eff[1] > eff[2]


def test_wheel_driving_worse_than_ppl_at_20deg():
    b1 = run("B1", 20.0)
    b2 = run("B2", 20.0)
    assert (not b1.completed) or b1.d < b2.d


def test_steep_slope_fails_as_slide_not_pitch():
    r = run("B2", 30.0, n_cycles=2)
    assert not r.completed and r.fail_reason in ("slide_back", "stall_slip", "stall_advance")


def test_determinism_and_crn():
    a = run("B3", 10.0, n_cycles=2, seed=7)
    b = run("B3", 10.0, n_cycles=2, seed=7)
    np.testing.assert_array_equal(a.rec["x_R"], b.rec["x_R"])
    assert a.cycles[-1]["f_slip_est_mean"] == b.cycles[-1]["f_slip_est_mean"]


def test_energy_nonnegative_and_cot_finite():
    r = run("B3", 10.0)
    assert r.E_wheel >= 0 and r.E_act >= 0 and r.E_rail >= 0
    assert np.isfinite(r.cot) and r.cot > 0


def test_stage_c_without_artefact_equals_b3():
    a = run("B3", 10.0, n_cycles=2)
    c = run("C", 10.0, n_cycles=2)
    np.testing.assert_array_equal(a.rec["x_R"], c.rec["x_R"])
    assert c.meta["dx_max_abs"] == 0.0
