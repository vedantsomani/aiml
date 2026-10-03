import numpy as np

from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall, TeamConfig
from pitsense.pitwall.engineers.head import decide
from pitsense.pitwall.engineers.strategy import analysis, priors
from pitsense.pitwall.engineers.strategy.run import simulate_field
from pitsense.pitwall.engineers.strategy.sim import Draws, FieldIn
from pitsense.pitwall.types import ACTIONS
from pitsense.state import RaceState


def _run(log, until=None, **ctx):
    wall = PitWall(Context(prior=PitLossPrior(), **ctx))
    state = RaceState(log.meta)
    for e in log.events:
        if until is not None and e.t > until:
            break
        state.apply(e)
        wall.observe(state)
    return state, wall


def _call_tuple(c):
    return (c.car, c.action, c.compound, c.confidence, c.plan_a, c.plan_b, tuple((r.code, r.text, r.value) for r in c.reasons))


def test_calls_are_valid_and_identical_from_truncated_log(race_log):
    team = TeamConfig(cars=("11", "22"))
    cut = race_log.events[int(len(race_log) * 0.7)].t
    _, w_full = _run(race_log.until(cut), team=team)  # the future is not in the log at all
    state, w_cut = _run(race_log, until=cut, team=team)  # the future exists but is never applied
    full = w_full.calls(state)
    cut_calls = w_cut.calls(state)
    assert {c.car for c in cut_calls} == {"11", "22"}
    assert all(c.action in ACTIONS for c in cut_calls)
    assert [_call_tuple(c) for c in full] == [_call_tuple(c) for c in cut_calls]


def _field(C=6, total=30, A=10):
    z = np.zeros
    F = FieldIn(A=A, total=total, cars=[str(i) for i in range(C)], x0=np.arange(C) * 2.0, pace0=np.full(C, 90.0),
                deg=np.full(C, 0.08), fuel=0.03, age0=np.full(C, 8.0), stint_age=np.full(C, 8.0),
                comp0=np.ones(C, dtype=np.int64), fresh=np.tile([88.8, 89.5, 90.3], (C, 1)), fresh_ref=np.full(C, A + 3.0),
                degc=np.tile([0.12, 0.08, 0.05], (C, 1)), pp=np.tile([0.05, 0.15, 0.3], (C, 1)), must=np.zeros(C, bool),
                used=np.tile([False, True, False], (C, 1)), cliff_risk=z(C), team_off=z(C), penalty=z(C))
    F.loss = {"green": 21.0, "sc": 11.0, "vsc": 14.0, "sd": 1.0}
    F.stints = [([14, 16, 18], []), ([22, 25, 28], []), ([30, 33, 36], [])]
    F.slots, F.n_slots = np.arange(C), C
    return F


def test_simulation_is_deterministic_and_orders_the_field():
    F = _field()
    runs = [simulate_field(F, Draws(F, 64, 7)) for _ in range(2)]
    assert np.array_equal(runs[0].pos, runs[1].pos)
    assert runs[0].pos[:, 0].mean() < runs[0].pos[:, -1].mean()  # the car in front tends to finish ahead
    assert runs[0].pos.min() >= 1 and runs[0].pos.max() <= F.C


def test_same_seed_same_futures_on_later_laps():
    F1, F2 = _field(A=10), _field(A=12)
    d1, d2 = Draws(F1, 16, 3), Draws(F2, 16, 3)
    assert np.array_equal(d1.noise[:, :, 2:], d2.noise)  # absolute laps line up
    assert np.array_equal(d1.level, d2.level)


def test_candidate_plans_respect_the_two_compound_rule():
    F = _field()
    F.must[0] = True
    info = {"stops_done": {"0": 0}, "stint_left": {"0": None}}
    plans = analysis.candidates(F, 0, info, "0")
    assert plans and all(any(cj != 1 for _, cj in p) for p in plans)  # the used set is MEDIUM only


def test_head_hysteresis_keeps_last_laps_target():
    from pitsense.pitwall.engineers.strategy.analysis import CarAnalysis, PlanResult

    def pr(stops, util):
        return PlanResult(stops, util, 0.0, 0.0, (stops[0][0] - 11) if stops else None, None, util, None)

    best, near = pr(((14, "HARD"),), 3.00), pr(((16, "HARD"),), 3.03)
    a = CarAnalysis("1", True, anchor=10, plan_a=best, ranked=[best, near])
    assert decide(a, None)[4].stops == best.stops
    assert decide(a, 16)[4].stops == near.stops  # last lap's target stays while within the tolerance
    far = pr(((16, "HARD"),), 3.5)
    a2 = CarAnalysis("1", True, anchor=10, plan_a=best, ranked=[best, far])
    assert decide(a2, 16)[4].stops == best.stops


def test_priors_default_without_history():
    p = priors.Priors((), 5)
    assert p.life["MEDIUM"] == priors.LIFE["MEDIUM"] and p.sc_rate > 0 and len(p.pass_p) == 6
