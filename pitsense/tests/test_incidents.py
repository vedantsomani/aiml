"""Incidents engineer on a synthetic race (no timing data in the repo)."""

from __future__ import annotations

from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.pitwall.engineers.incidents import FEATURES, IncidentsEngineer, classify_rc, clip_features, probabilities
from pitsense.pitwall.engineers.mechanic_radio import MechanicRadio
from pitsense.pitwall.engineers.mechanic_telemetry import MechanicTelemetry
from pitsense.state import RaceState

from .conftest import _fmt, make_log

CARS = [str(n) for n in range(1, 9)]
N_LAPS = 12
START = 100.0


def _events(slow_car="3", slow_lap=11, extra=6.0):
    ev = []
    add = lambda t, topic, data: ev.append((t, topic, data))  # noqa: E731
    add(0.0, "SessionInfo", {"Meeting": {"Name": "Test GP"}, "Name": "Race", "Type": "Race"})
    add(0.0, "SessionStatus", {"Status": "Inactive"})
    add(0.0, "TrackStatus", {"Status": "1", "Message": "AllClear"})
    add(0.5, "DriverList", {n: {"RacingNumber": n, "Tla": f"C{n}", "TeamName": f"Team {n}"} for n in CARS})
    add(1.0, "LapCount", {"CurrentLap": 1, "TotalLaps": N_LAPS})
    add(1.0, "TimingAppData", {"Lines": {n: {"RacingNumber": n, "Stints": []} for n in CARS}})
    add(1.0, "TimingData", {"Lines": {n: {"Position": str(i + 1), "Line": i + 1, "NumberOfPitStops": 0, "NumberOfLaps": 0,
                                           "InPit": False, "PitOut": False, "Retired": False, "Stopped": False,
                                           "LastLapTime": {"Value": ""}} for i, n in enumerate(CARS)}})
    add(10.0, "TimingAppData", {"Lines": {n: {"Stints": {"0": {"Compound": "MEDIUM", "New": "true", "TyresNotChanged": "0",
                                                                "TotalLaps": 0, "StartLaps": 0}}} for n in CARS}})
    add(START, "SessionStatus", {"Status": "Started"})
    t_cross = {n: START for n in CARS}
    for lap in range(1, N_LAPS + 1):
        for i, n in enumerate(CARS):
            lt = 90.0 + 0.3 * i - 0.1 * lap + (extra if n == slow_car and lap >= slow_lap else 0.0)
            t_cross[n] += lt if lap > 1 else 95.0 + i
            upd = {"NumberOfLaps": lap, "Position": str(i + 1), "Line": i + 1}
            if lap > 1:
                upd["LastLapTime"] = {"Value": _fmt(lt)}
            add(t_cross[n], "TimingData", {"Lines": {n: upd}})
            if n == CARS[0]:
                add(t_cross[n] + 0.01, "LapCount", {"CurrentLap": lap + 1})
    return ev, t_cross


def _run(events, stop_at=None):
    log = make_log(events)
    st = RaceState(log.meta)
    wall = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=[MechanicTelemetry, MechanicRadio, IncidentsEngineer])
    for e in log.events:
        if stop_at is not None and e.t > stop_at:
            break
        st.apply(e)
        wall.observe(st)
    return st, wall


def test_classify_rc():
    assert classify_rc("TURN 4 INCIDENT INVOLVING CARS 1 (VER) AND 44 (HAM) NOTED - CAUSING A COLLISION") == ("collision", ("1", "44"))
    assert classify_rc("INCIDENT INVOLVING CAR 7 (HAD) NOTED - LEAVING THE TRACK AND GAINING AN ADVANTAGE")[0] == "incident"
    assert classify_rc("INCIDENT INVOLVING CAR 7 (HAD) NOTED - SPEEDING IN THE PIT LANE")[0] == ""
    assert classify_rc("FIA STEWARDS: INCIDENT INVOLVING CARS 1 (VER) AND 44 (HAM) REVIEWED NO FURTHER INVESTIGATION")[0] == ""
    assert classify_rc("DEBRIS ON TRACK AT TURN 3")[0] == "debris"
    assert classify_rc("CAR 4 (NOR) TIME 1:30.1 DELETED - TRACK LIMITS")[0] == ""


def test_probabilities_are_monotone_and_bounded():
    base = {k: 0.0 for k in FEATURES}
    p0 = probabilities(base)
    hi = dict(base, pace_drop=6.0, rc_collision=1.0)
    p1 = probabilities(hi)
    assert 0.0 <= p0[0] <= p1[0] <= 1.0 and 0.0 <= p0[1] <= p1[1] <= 1.0
    assert clip_features(dict(base, pace_drop=99.0))["pace_drop"] == 2.0


def test_pace_collapse_is_found_against_own_stint_and_field():
    ev, _ = _events()
    st, wall = _run(ev)
    v = wall.view(st)
    slow, ok = v.car("incidents", "3"), v.car("incidents", "5")
    assert slow["pace_drop"] > 4.0 and abs(ok["pace_drop"]) < 0.5
    assert slow["pace_drop_prev"] > 4.0
    assert slow["neigh_drop"] > 3.0
    assert slow["damage_prob"] >= ok["damage_prob"]
    for x in slow.values():
        assert x is None or isinstance(x, (int, float, str, bool))


def test_collision_message_names_the_car_and_expires():
    ev, t_cross = _events(slow_car="none")
    t_msg = t_cross["2"] - 3 * 90.0
    ev.append((t_msg, "RaceControlMessages", {"Messages": {"0": {"Category": "Other", "Message": "TURN 3 INCIDENT INVOLVING CARS 2 (BBB) AND 6 (CCC) NOTED - CAUSING A COLLISION", "Lap": 9}}}))
    st, wall = _run(ev)
    v = wall.view(st)
    assert v.car("incidents", "2")["rc_collision"] == 1.0 and v.car("incidents", "6")["rc_collision"] == 1.0
    assert v.car("incidents", "4")["rc_collision"] == 0.0
    assert "collision" in v.car("incidents", "2")["incident_reason"]


def test_as_of_now_matches_truncated_log():
    ev, t_cross = _events()
    cut = t_cross["1"] - 1.0 * 90.0
    st, wall = _run(ev, stop_at=cut)
    full_st, full_wall = _run([e for e in ev if e[0] <= cut])
    a, b = wall.view(st).car("incidents", "3"), full_wall.view(full_st).car("incidents", "3")
    assert a == b


def test_retired_car_has_no_probability():
    ev, _ = _events()
    ev.append((START + 1000.0, "TimingData", {"Lines": {"3": {"Retired": True}}}))
    st, wall = _run(ev)
    v = wall.view(st).car("incidents", "3")
    assert v["damage_prob"] == 0.0 and v["incident_reason"] == ""


def test_head_calls_box_with_reason_only_when_probability_is_high():
    from pitsense.pitwall.engineers.head import DAMAGE_BOX, damage_call

    lo = {"damage_prob": DAMAGE_BOX - 0.01, "incident_reason": "debris on track"}
    hi = {"damage_prob": 0.8, "incident_reason": "race control: collision involving the car"}
    assert damage_call(5.0, "3", lo) is None
    assert damage_call(5.0, "3", {}) is None
    assert damage_call(5.0, "3", hi, pit_open=False) is None
    c = damage_call(5.0, "3", hi)
    assert c.action == "BOX" and c.car == "3" and c.reasons[0].code == "damage" and "collision" in c.reasons[0].text
