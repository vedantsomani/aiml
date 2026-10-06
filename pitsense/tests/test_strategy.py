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


# --- call quality: the P18.7 forecast (live Bahrain 2026, lap 9 under an SC) -------------------------------------
def test_order_consistent_puts_the_car_where_the_timing_screen_has_it():
    from types import SimpleNamespace as NS

    F = _field(C=4)
    F.x0 = np.array([0.0, 50.0, 101.0, 90.0])  # car 3 is ahead of car 2's time on screen P3 but crossed the line 11 s later
    cars = [NS(position=1, laps=9), NS(position=2, laps=9), NS(position=3, laps=9), NS(position=4, laps=9)]
    analysis._order_consistent(F, cars)
    assert np.all(np.diff(F.x0) > 0)
    assert F.x0[2] == 101.0 and F.x0[3] > 101.0


def test_order_consistent_leaves_other_laps_alone():
    from types import SimpleNamespace as NS

    F = _field(C=3)
    F.x0 = np.array([0.0, 120.0, 30.0])
    analysis._order_consistent(F, [NS(position=1, laps=9), NS(position=2, laps=8), NS(position=3, laps=9)])
    assert list(F.x0) == [0.0, 120.0, 30.0]


def test_pace_band_bounds_outlier_stints():
    lo, hi = 110.0 * (1 - analysis.PACE_BAND[0]), 110.0 * (1 + analysis.PACE_BAND[1])
    assert min(max(137.3, lo), hi) < 120.0 and min(max(100.0, lo), hi) == lo


# --- call stability, SC reaction, field switches ----------------------------------------------------------------
def _an(anchor, stops_util, exp=5.0, ps=None):
    """Analysis whose ranked plans are [(stops, util)], best first (plan-only rule: BOX when the best plan stops now)."""
    from types import SimpleNamespace as NS

    ranked = [NS(util=u, stops=s, first_offset=(s[0][0] - anchor - 1) if s else None, exp_pos=exp) for s, u in stops_util]
    return NS(plan_a=True, car="44", anchor=anchor, ranked=ranked, diff_now_later=(-0.2, 0.05), pp=None, ps=ps, gain_sc=None,
              sc_best_comp=None)


def test_box_keeps_compound_and_decays_when_not_taken():
    ctl = {}
    now = lambda lap, comp, other: _an(lap, [(((lap + 1, comp),), 3.0), (((lap + 1, other),), 3.05), (((lap + 9, comp),), 3.1)])  # noqa: E731
    r = [decide(now(24, "SOFT", "MEDIUM"), None, ctl=ctl, stops_done=0, pos=5)]
    r.append(decide(now(25, "MEDIUM", "SOFT"), None, ctl=ctl, stops_done=0, pos=5))  # medium now marginally best: keep SOFT
    r.append(decide(now(26, "MEDIUM", "SOFT"), None, ctl=ctl, stops_done=0, pos=5))  # third lap: not taken
    assert [x[0] for x in r] == ["BOX", "BOX", "PREPARE_BOX"] or [x[0] for x in r] == ["BOX", "BOX", "STAY_OUT"]
    assert r[0][1] == r[1][1] == "SOFT" and r[2][3] == "box_not_taken"
    r4 = decide(now(27, "MEDIUM", "SOFT"), None, ctl=ctl, stops_done=0, pos=5)
    assert r4[0] != "BOX"  # not repeated while nothing changed
    r5 = decide(now(28, "MEDIUM", "SOFT"), None, ctl=ctl, stops_done=1, pos=5)  # the stop was made: reset
    assert r5[0] == "BOX" and "box_lap" in ctl and ctl["box_lap"] == 28


def test_compound_changes_when_the_plan_clearly_prefers_another():
    ctl = {}
    decide(_an(24, [(((25, "SOFT"),), 3.0), (((25, "HARD"),), 3.5)]), None, ctl=ctl, stops_done=0)
    out = decide(_an(25, [(((26, "HARD"),), 3.0), (((26, "SOFT"),), 3.5)]), None, ctl=ctl, stops_done=0)
    assert out[0] == "BOX" and out[1] == "HARD"


def test_implausible_forecast_blocks_box():
    out = decide(_an(9, [(((10, "SOFT"),), 3.0)], exp=18.7), None, ctl={}, stops_done=0, pos=15 - 9)
    assert out[0] == "PREPARE_BOX" and out[3] == "forecast_implausible"


def test_sc_deployed_in_window_boxes_now():
    a = _an(20, [(((26, "MEDIUM"),), 3.0), (((21, "HARD"),), 3.2)])
    assert decide(a, None, ctl={}, sc_phase="none", stops_done=0)[0] == "STAY_OUT"
    out = decide(a, None, ctl={}, sc_phase="sc", stops_done=0)
    assert out[0] == "BOX" and out[1] == "HARD" and out[3] == "sc_cheap_stop"
    assert decide(a, None, ctl={}, sc_phase="sc", pit_open=False, stops_done=0)[0] == "PREPARE_BOX"
    far = _an(20, [(((45, "MEDIUM"),), 3.0), (((21, "HARD"),), 3.2)])
    assert decide(far, None, ctl={}, sc_phase="sc", stops_done=0)[0] == "STAY_OUT"  # not in its window


def test_field_event_switch_to_slicks_and_inters():
    from types import SimpleNamespace as NS

    from pitsense.pitwall.engineers.strategy import wethead

    def lapr(c):
        return NS(compound=c)

    mem = NS(index=NS(by_driver={str(i): {8: lapr("INTERMEDIATE")} for i in range(10)}))
    drivers = {str(i): NS(number=str(i), running=True, laps=10, compound="SOFT" if i < 6 else "INTERMEDIATE") for i in range(10)}
    state = NS(drivers=drivers)
    view = NS(race=lambda k: {"crossover": "to_slicks", "inters_vs_slicks_s": 3.1})
    ev = wethead.field_event(mem, state, view, NS(compound="INTERMEDIATE"))
    assert ev["to_slicks"] == 6 and ev["target"] == "SLICKS"
    view2 = NS(race=lambda k: {"crossover": "none", "inters_vs_slicks_s": None})
    assert wethead.field_event(mem, NS(drivers={"1": NS(number="1", running=True, laps=10, compound="SOFT")}), view2,
                               NS(compound="INTERMEDIATE"))["target"] is None
    act, comp, rs = wethead.apply_event(ev, "STAY_OUT", None, [], {}, 10, priors.Priors((), 5), NS(total_laps=50))
    assert act == "BOX" and "6 cars switched to slicks" in rs[0].text
