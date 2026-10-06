"""Simulator realism: several neutralisations per future, red flags, dirty air, circuit priors (synthetic data)."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from pitsense.asof import RaceSummary
from pitsense.pitwall.engineers.strategy import priors
from pitsense.pitwall.engineers.strategy.run import _stop_amount, evaluate_plans, simulate_field
from pitsense.pitwall.engineers.strategy.sim import K_STOPS, NOSTOP, Draws, FieldIn, red_tyres
from pitsense.state import RaceState


def _field(C=6, total=40, A=10):
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


def _summary(i, circuit, ex, laps=50, green=(21.0, 22.0, 20.5)):
    t = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=14 * i)
    return RaceSummary(f"r{i}", 2024, circuit, t, t + timedelta(hours=2), list(green), [12.0], [15.0], laps, 0, 0,
                       extra={"strategy": dict({"stints": {}, "laps": laps, "v": 2, "pass": [[0] * 6, [0] * 6]}, **ex)})


def _starts(status):
    s = status.astype(int)
    return ((s[:, 1:] > 0) & (s[:, :-1] == 0)).sum(1) + (s[:, 0] > 0)


def test_several_neutralisations_per_future_with_green_laps_between():
    F = _field(total=60)
    F.haz = np.zeros((3, F.R))
    F.haz[1], F.haz[2] = 0.06, 0.04
    F.sc_lens, F.vsc_lens = (3, 4, 4, 5, 6), (1, 2, 2, 2, 3)
    D = Draws(F, 400, 11)
    n = _starts(D.status)
    assert (n >= 2).mean() > 0.3 and n.max() >= 3
    assert np.array_equal(D.status, Draws(F, 400, 11).status)  # seeded
    # after every neutralisation at least restart_gap green laps
    for s in D.status[:50].astype(int):
        ends = np.where((s[:-1] > 0) & (s[1:] == 0))[0]
        for e in ends:
            assert (s[e + 1:e + 3] == 0).all()
    # lengths come from the past ones
    sc_runs = []
    for s in D.status.astype(int):
        run = 0
        for x in list(s) + [0]:
            if x == 2:
                run += 1
            elif run:
                sc_runs.append(run)
                run = 0
    assert set(sc_runs) <= {1, 2, 3, 4, 5, 6}


def test_hazard_follows_the_lap_phase_and_circuit():
    late = {"sc": [[40, 4], [45, 3], [42, 5]] * 4, "vsc": [], "red": []}
    races = [_summary(i, 7, late) for i in range(10)] + [_summary(10 + i, 9, {"sc": [], "vsc": [], "red": []}) for i in range(10)]
    p7, p9 = priors.Priors(races, 7), priors.Priors(races, 9)
    h7 = p7.hazard(10, 50)
    assert h7[1, -5:].mean() > 2 * h7[1, :5].mean()  # SCs in this pool come late in the race
    assert p7.hazard(10, 50)[1].mean() > p9.hazard(10, 50)[1].mean()  # circuit 7 has them, circuit 9 never
    assert h7.shape == (3, 40) and (h7[0] > 0).all()  # red flags: small default hazard


def test_red_flag_is_a_free_tyre_change():
    F = _field(total=40, A=10)
    red = np.array([15, NOSTOP])
    stop = np.full((2, 2, K_STOPS), NOSTOP, dtype=np.int64)
    comp = np.zeros((2, 2, K_STOPS), dtype=np.int64)
    stop[0, :, 0] = 22  # plan with a stop after the red flag: it moves to the red-flag lap
    s2, c2 = red_tyres(F, red[None, :], stop, comp)
    assert s2[0, 0, 0] == 15 and s2[0, 1, 0] == 22
    assert s2[1, 0, 0] == 15 and c2[1, 0, 0] in (1, 2)  # no stop planned: a free set that reaches the flag
    assert s2[1, 1, 0] == NOSTOP
    assert float(_stop_amount(F, np.array([3]), np.array([0.0]), 0.0)[0]) == 0.0
    assert float(_stop_amount(F, np.array([0]), np.array([0.0]), 0.0)[0]) > 15


def test_red_flags_in_simulated_futures():
    F = _field(total=50, A=10)
    F.haz = np.zeros((3, F.R))
    F.haz[0] = 0.05
    D = Draws(F, 200, 5)
    has = D.red_lap < NOSTOP
    assert has.any()
    i = int(np.argmax(has))
    assert D.status[i, D.red_lap[i] - F.A - 1] == 3
    run = simulate_field(F, D)
    on_red = (run.stop == D.red_lap[:, None, None]).any(2)
    assert on_red[has].mean() > 0.5  # most cars take the free change


def test_dirty_air_costs_time_behind_a_close_car():
    F = _field(C=2, total=30, A=10)
    F.x0 = np.array([0.0, 0.5])
    F.pace0 = np.array([90.3, 90.0])  # the car behind is a little faster but cannot pass
    F.pass_p = (0.0,) * 6
    stop = np.full((1, 32, K_STOPS), NOSTOP, dtype=np.int64)
    comp = np.zeros((1, 32, K_STOPS), dtype=np.int64)
    t = []
    for air in (0.0, 0.6):
        F.air = air
        D = Draws(F, 32, 3)
        run = simulate_field(F, D)
        t.append(evaluate_plans(F, D, run, 1, stop, comp)[1].mean())
    assert t[1] > t[0]


def test_pass_chance_shift_by_circuit_and_air_learnt():
    def ex(swp_share, air):
        e = [60, 60, 60, 60, 60, 60]
        return {"sc": [], "vsc": [], "red": [], "pass2": [e, [int(x * swp_share) for x in e]], "air": [air * 50, 50]}

    races = [_summary(i, 1, ex(0.05, 0.5)) for i in range(4)] + [_summary(4 + i, 2, ex(0.4, 0.1)) for i in range(4)]
    hard, easy = priors.Priors(races, 1), priors.Priors(races, 2)
    assert all(a < b for a, b in zip(hard.pass_p, easy.pass_p))
    assert hard.pass_shift < 0 < easy.pass_shift
    assert all(b >= a for a, b in zip(hard.pass_p, hard.pass_p[1:]))
    assert hard.air > easy.air
    F = _field()
    hard.fill(F)
    assert F.haz.shape == (3, F.R) and F.air == hard.air and F.pass_p == tuple(hard.pass_p)


def test_pit_loss_by_circuit_fills_missing_values():
    races = [_summary(i, 3, {"sc": [], "vsc": []}, green=(30.0, 31.0, 29.0)) for i in range(3)]
    races += [_summary(3 + i, 4, {"sc": [], "vsc": []}, green=(19.0, 20.0, 21.0)) for i in range(3)]
    p3, p4 = priors.Priors(races, 3), priors.Priors(races, 4)
    assert p3.loss["green"] > p4.loss["green"] and p3.loss["sc"] < p3.loss["green"]
    F = _field()
    F.loss = {"green": None, "sc": 11.0, "vsc": None, "sd": 1.0}
    p3.fill(F)
    assert F.loss["sc"] == 11.0 and F.loss["green"] == p3.loss["green"] and F.loss["vsc"] is not None


def test_old_history_without_new_keys_still_works():
    old = [_summary(i, 1, {"sc": [[10, 4]], "vsc": [[20, 2]]}) for i in range(3)]
    for r in old:
        del r.extra["strategy"]["v"]
    p = priors.Priors(old, 1)
    assert len(p.pass_p) == 6 and p.hazard(5, 50).shape == (3, 45) and p.air > 0


def test_learn_on_a_synthetic_race_is_json(race_log):
    final = RaceState(race_log.meta)
    for e in race_log.events:
        final.apply(e)
    out = priors.learn(final, {"circuit_key": 1})
    for k in ("red", "pass2", "air", "v"):
        assert k in out
    json.dumps(out)
