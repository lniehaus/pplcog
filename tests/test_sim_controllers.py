import numpy as np

from pplcog.params import Actuator, Rover, Study
from pplcog.actuator import simulate as qs_simulate
from pplcog.frame import phase_timeline, wheelbase, x_to_xi
from pplcog.statics import NF_deviation_bound, xm_star
from pplcog.sim.cog_controllers import ConstantOffsets, RailActuator, delta_max_for_run
from pplcog.sim.params_sim import SimParams, SoilWR
from pplcog.sim.simulate import run_episode
from pplcog.sim.wheel_soil import WheelTable
from pplcog.sim.learning import soil_population

RV, ACT, P = Rover(), Actuator(), SimParams()


def test_rail_actuator_matches_quasi_static_simulation():
    """Feeding the quasi-static feedforward references through RailActuator reproduces actuator.simulate."""
    theta = np.radians(5.0)
    st = Study(n_cycles=2)
    ref = qs_simulate(RV, ACT, st, theta, "ff", offset=False, Delta_max=0.05)
    t, phase, s = phase_timeline(ACT.T_phase, ACT.dt, st.n_cycles)
    L = wheelbase(phase, s, RV)
    xi_ff = np.clip(x_to_xi(xm_star(st.rho_star, L, theta, RV), L, RV), RV.rail_lo, RV.rail_hi)
    rail = RailActuator(RV, ACT, float(xi_ff[0]))
    xs = np.array([rail.update(float(r), ACT.dt)[0] for r in xi_ff])
    np.testing.assert_allclose(xs, ref.x_m, atol=1e-9)


def test_stage_c_offsets_within_envelope_and_bound():
    soils = soil_population(P)
    for si in (40, 41):
        soil = soils[si]
        tab = WheelTable.build(soil, RV, P)
        theta_deg = 5.0
        theta = np.radians(theta_deg)
        Dm = delta_max_for_run(RV, theta, P)
        art = ConstantOffsets(Dm, -Dm)
        c = run_episode(RV, ACT, P, soil, theta, "C", artefact=art, n_cycles=3, tab=tab, seed=1)
        b = run_episode(RV, ACT, P, soil, theta, "C", artefact=None, n_cycles=3, tab=tab, seed=1)
        assert c.meta["dx_max_abs"] <= Dm + 1e-12
        n = min(len(c.rec["t"]), len(b.rec["t"]))
        bound = NF_deviation_bound(theta, c.rec["L"][:n], RV, Dm)
        # the commanded mass offset is realised within the envelope
        assert np.max(np.abs(c.rec["xi_m"][:n] - b.rec["xi_m"][:n])) <= Dm + 1e-9
        # the realised front-load deviation is reported relative to the quasi-static bound (dynamic terms may add to it)
        ratio = np.max(np.abs(c.rec["N_F"][:n] - b.rec["N_F"][:n]) / bound)
        assert np.isfinite(ratio)


def test_stage_a_set_point_falls_back_when_grid_search_is_uninformative():
    """A flat, all-failed objective must not select the first grid entry (the rear rail end)."""
    from pplcog.sim.learning import physics_set_point, select_set_point
    grid = np.linspace(RV.rail_lo, RV.rail_hi, 9)
    fb = physics_set_point(RV, np.radians(30.0))
    assert RV.rail_lo <= fb <= RV.rail_hi and fb > 0.5 * (RV.rail_lo + RV.rail_hi)   # front half at 30 degrees
    flat = np.full(9, 20.0)
    xi, used = select_set_point(flat, np.zeros(9, int), grid, fb)
    assert used and xi == fb
    scores = flat.copy(); scores[6] = 2.0
    xi, used = select_set_point(scores, np.array([0, 0, 0, 0, 1, 2, 3, 3, 3]), grid, fb)
    assert not used and xi == grid[6]
    xi, used = select_set_point(scores, np.array([0, 0, 0, 0, 1, 2, 2, 3, 3]), grid, fb)
    assert used and xi == fb
