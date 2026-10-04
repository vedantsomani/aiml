"""Mechanic engineers on synthetic telemetry and radio (no timing data in the repo)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pitsense.asof import LeakageError
from pitsense.events import Event, EventLog
from pitsense.feeds import RadioStore
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall
from pitsense.pitwall.engineers.mechanic_chief import MechanicChief, rc_car_problem
from pitsense.pitwall.engineers.mechanic_radio import MechanicRadio, classify
from pitsense.pitwall.engineers.mechanic_telemetry import MechanicTelemetry, Level
from pitsense.state import RaceState

from .conftest import make_log, synthetic_events

CARS = ["11", "22", "33"]
LAP_S = 60.0
N_LAPS = 14
T0 = 100.0
UTC0 = 1_750_000_000.0  # epoch of t = 0 for the feed's own clock
FAULT_T = T0 + 7 * LAP_S  # car 22 loses power here
ENGINES = [MechanicTelemetry, MechanicRadio, MechanicChief]


def _fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:06.3f}"


def _iso(utc: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(utc, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _healthy_speed(s: float) -> float:
    if s < 30:
        return 120 + 180 * (s / 30) ** 0.6
    if s < 34:
        return 300 - (300 - 90) * (s - 30) / 4
    return 90 + 60 * math.sin((s - 34) / 26 * math.pi)


_DIST = np.cumsum([_healthy_speed(i / 100) / 3.6 * 0.01 * 10 for i in range(6000)])  # 1/10 m


def _lap_profile(s: float, scale: float = 1.0, gearbox_fault: bool = False):
    """One 60 s lap around a 2 km loop: speed, rpm, gear, throttle, brake and the track position.

    0-30 s a long straight at full throttle (accelerating towards 300 km/h), 30-34 s braking to
    90 km/h, 34-60 s a medium-speed section with throttle in the middle.
    """
    if s < 30:
        v = 120 + 180 * (s / 30) ** 0.6
        v *= scale
        thr, brk = 100.0, 0.0
    elif s < 34:
        v = (300 - (300 - 90) * (s - 30) / 4) * (scale if scale < 1 else 1.0)
        thr, brk = 0.0, 100.0
    else:
        v = 90 + 60 * math.sin((s - 34) / 26 * math.pi)
        thr, brk = 60.0, 0.0
    gear = int(np.clip(1 + v // 40, 2, 8))
    if gearbox_fault and s < 30:
        gear = 0  # neutral while moving
    rpm = float(np.clip(v * (62.0 / (1 + 0.15 * (gear - 4)) if gear else 20.0), 4000, 12000))
    # track: distance along the lap at the healthy car's speed, in 1/10 m, so that the position feed agrees with CarData
    x = _DIST[min(int(s * 100), len(_DIST) - 1)]
    y = 0.0
    return v, rpm, gear, thr, brk, x, y


def feed_events(fault_car: str | None = None, fault: str = "power", freeze_pos: bool = False) -> list[tuple[float, str, dict]]:
    """Race timing for 3 cars plus 4 Hz CarData and Position. The fault car develops ``fault`` at FAULT_T."""
    ev: list[tuple[float, str, dict]] = []
    add = lambda t, topic, data: ev.append((t, topic, data))  # noqa: E731
    add(0.0, "SessionInfo", {"Meeting": {"Name": "Test GP"}, "Name": "Race", "Type": "Race"})
    add(0.0, "TrackStatus", {"Status": "1", "Message": "AllClear"})
    add(0.5, "DriverList", {n: {"RacingNumber": n, "Tla": f"C{n}", "TeamName": f"Team {n}"} for n in CARS})
    add(1.0, "LapCount", {"CurrentLap": 1, "TotalLaps": N_LAPS + 5})
    add(1.0, "TimingData", {"Lines": {
        n: {"Position": str(i + 1), "Line": i + 1, "GapToLeader": "", "IntervalToPositionAhead": {"Value": ""},
            "InPit": False, "PitOut": False, "NumberOfPitStops": 0, "NumberOfLaps": 0, "Retired": False, "Stopped": False,
            "LastLapTime": {"Value": ""}} for i, n in enumerate(CARS)}})
    add(T0, "SessionStatus", {"Status": "Started"})
    for lap in range(1, N_LAPS + 1):
        t_end = T0 + lap * LAP_S
        for i, n in enumerate(CARS):
            upd = {"NumberOfLaps": lap}
            if lap > 1:
                upd["LastLapTime"] = {"Value": _fmt(LAP_S + 0.1 * i)}
            add(t_end + 0.1 * i, "TimingData", {"Lines": {n: upd}})
        add(t_end + 0.5, "LapCount", {"CurrentLap": lap + 1})
    # telemetry and positions: one message per second carrying 4 entries 0.25 s apart; the entries'
    # own clock runs 2 s behind the publish time
    for sec in range(1, int(T0 + N_LAPS * LAP_S) + 2):
        t = float(sec)
        tel, pos = [], []
        for k in range(4):
            tm = t - 2.0 + 0.25 * k
            cars, pcars = {}, {}
            for ci, n in enumerate(CARS):
                s = ((tm - T0) + ci * 3.0) % LAP_S if tm >= T0 else 0.0
                scale = 1.0
                gb = False
                if n == fault_car and tm >= FAULT_T:
                    if fault == "power":
                        scale = 0.88
                    elif fault == "gearbox":
                        gb = True
                v, rpm, gear, thr, brk, x, y = _lap_profile(s, scale, gb)
                if tm < T0:
                    v = rpm = gear = thr = brk = 0.0
                cars[n] = {"Channels": {"0": rpm, "2": v, "3": gear, "4": thr, "5": brk, "45": 0}}
                if freeze_pos:
                    x, y = 1000.0, 1000.0  # a stuck position feed
                pcars[n] = {"Status": "OnTrack", "X": x + 40 * ci, "Y": y + 20 * ci, "Z": 0}
            tel.append({"Utc": _iso(UTC0 + tm), "Cars": cars})
            pos.append({"Timestamp": _iso(UTC0 + tm), "Entries": pcars})
        add(t, "CarData.z", {"Entries": tel})
        add(t + 0.01, "Position.z", {"Position": pos})
    return ev


def run(events, until: float | None = None, engineers=ENGINES, every: float = 20.0):
    log = make_log(events)
    st = RaceState(log.meta)
    wall = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=engineers)
    for e in log.events:
        if until is not None and e.t > until:
            break
        st.apply(e)
        wall.observe(st)
    return st, wall, log


# --------------------------------------------------------------------------- telemetry
def test_power_loss_flagged_on_the_faulty_car_only():
    st, wall, _ = run(feed_events("22", "power"))
    v = {n: wall.view(st).car("mechanic_telemetry", n) for n in CARS}
    assert v["22"]["power_loss"] is not None and v["22"]["power_loss"] > 0.6
    assert v["22"]["mech_issue"] == "power_loss" and v["22"]["mech_risk"] > 0.6
    assert v["22"]["power_loss_since"] is not None and FAULT_T <= v["22"]["power_loss_since"] <= FAULT_T + 120
    for n in ("11", "33"):
        assert (v[n]["power_loss"] or 0) < 0.3 and v[n]["power_loss_since"] is None
    alerts = [a for a in wall.alerts(st) if a.engineer == "mechanic_telemetry"]
    assert {a.car for a in alerts} == {"22"} and ("22", "mech_power_loss") in {(a.car, a.code) for a in alerts}
    pl = next(a for a in alerts if a.code == "mech_power_loss")
    assert pl.since == v["22"]["power_loss_since"]


def test_healthy_race_raises_nothing():
    st, wall, _ = run(feed_events(None))
    assert not [a for a in wall.alerts(st)]
    for n in CARS:
        assert (wall.view(st).car("mechanic_telemetry", n)["mech_risk"] or 0) < 0.3


def test_frozen_position_feed_is_not_a_slow_car():
    """Positions that do not agree with CarData speed (2026 Hungary) must not turn into field comparisons."""
    st, wall, _ = run(feed_events(None, freeze_pos=True))
    assert wall.alerts(st) == []
    for n in CARS:
        v = wall.view(st).car("mechanic_telemetry", n)
        assert (v["slow_car"] or 0) < 0.3 and (v["power_loss"] is None or v["power_loss"] < 0.3)


def test_neutral_while_moving_is_a_gearbox_flag():
    log = make_log(feed_events("33", "gearbox"))
    st = RaceState(log.meta)
    wall = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=[MechanicTelemetry])
    best, best_other, flagged = 0.0, 0.0, False
    for i, e in enumerate(log.events):
        st.apply(e)
        wall.observe(st)
        if i % 40 == 0 and st.t > FAULT_T:
            v = wall.view(st)
            best = max(best, v.car("mechanic_telemetry", "33")["gearbox"] or 0.0)
            best_other = max(best_other, v.car("mechanic_telemetry", "11")["gearbox"] or 0.0)
            flagged |= any(a.car == "33" and a.code == "mech_gearbox" for a in wall.alerts(st))
    assert best > 0.6 and flagged and best_other < 0.3


def test_chief_combines_telemetry_into_one_alert_with_a_reason():
    st, wall, _ = run(feed_events("22", "power"))
    a = [x for x in wall.alerts(st) if x.engineer == "mechanic_chief"]
    assert len(a) == 1 and a[0].car == "22" and a[0].code == "mech_power_loss"
    assert "power loss suspected" in a[0].message and "telemetry" in a[0].message
    assert a[0].since is not None and a[0].since >= FAULT_T


def test_values_are_scalars_and_json_clean():
    st, wall, _ = run(feed_events("22", "power"), until=T0 + 9 * LAP_S)
    snap = wall.snapshot(st)
    import json

    json.dumps(snap.to_dict(), allow_nan=False)
    assert any(k.startswith("mechanic_telemetry__") for k in snap.cars["22"])


def test_no_feeds_no_values_no_errors():
    """The ordinary timing-only log: engineers stay silent."""
    st, wall, _ = run(synthetic_events())
    assert wall.alerts(st) == []
    v = wall.view(st).car("mechanic_chief", "11")
    assert v["risk"] == 0.0 and v["issue"] == ""


def test_level_hysteresis_is_level_triggered():
    lv = Level(0.6, 0.3, hold=10, clear=20)
    for t in range(0, 8):
        lv.update(float(t), 0.9)
    assert not lv.active  # not held long enough
    for t in range(8, 14):
        lv.update(float(t), 0.9)
    assert lv.active and lv.since == 0.0
    lv.update(14.0, 0.5)  # between off and on: stays
    lv.update(40.0, 0.1)
    assert lv.active
    lv.update(61.0, 0.1)
    assert not lv.active and lv.since is None


# --------------------------------------------------------------------------- leak test
def _values(wall, st):
    v = wall.view(st)
    out = {n: {**v.car("mechanic_telemetry", n), **{f"c_{k}": x for k, x in v.car("mechanic_chief", n).items()},
               **{f"r_{k}": x for k, x in v.car("mechanic_radio", n).items()}} for n in CARS}
    return out, [(a.engineer, a.code, a.car, a.since) for a in wall.alerts(st)]


@pytest.mark.parametrize("fault", ["power", "gearbox"])
def test_values_at_t_are_identical_from_full_log_and_until_t(fault):
    ev = feed_events("22", fault)
    log = make_log(ev)
    for t in (T0 + 5 * LAP_S, FAULT_T + 90.0, T0 + 11 * LAP_S + 7.3):
        cut = RaceState(log.meta)
        wall_c = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=ENGINES)
        sub = log.until(t)
        # the full run is asked at t: replay it again, stopping at t (what "values at t from the full log" means)
        full = RaceState(log.meta)
        wall_f = PitWall(Context(prior=PitLossPrior(), meta=log.meta), engineers=ENGINES)
        for e in log.events:
            if e.t > t:
                break
            full.apply(e)
            wall_f.observe(full)
        for e in sub.events:
            cut.apply(e)
            wall_c.observe(cut)
        assert _values(wall_f, full) == _values(wall_c, cut)


def test_feeds_refuse_the_future():
    st, wall, _ = run(feed_events("22", "power"), until=T0 + 3 * LAP_S)
    with pytest.raises(LeakageError):
        st.feeds.telemetry.telemetry("22", t=st.t + 50)
    with pytest.raises(LeakageError):
        st.feeds.radio.messages("22", t=st.t + 50)


def test_asking_more_often_changes_nothing():
    ev = feed_events("22", "power")
    a_st, a_wall, _ = run(ev)
    b_log = make_log(ev)
    b_st = RaceState(b_log.meta)
    b_wall = PitWall(Context(prior=PitLossPrior(), meta=b_log.meta), engineers=ENGINES)
    for i, e in enumerate(b_log.events):
        b_st.apply(e)
        b_wall.observe(b_st)
        if i % 50 == 0:
            b_wall.alerts(b_st)  # ask a lot
            b_wall.snapshot(b_st)
    assert _values(a_wall, a_st) == _values(b_wall, b_st)


# --------------------------------------------------------------------------- radio
@pytest.mark.parametrize("text,issue", [
    ("I'm losing power, something is wrong with the engine.", "power"),
    ("No power! I have no power.", "power"),
    ("I can't brake, no brakes, no brakes!", "brakes"),
    ("My brake pedal's going long.", "brakes"),
    ("Stuck in first gear.", "gearbox"),
    ("Something's broken in my gearbox.", "gearbox"),
    ("I think I have a puncture, rear left.", "puncture"),
    ("There is a big vibration from the front.", "vibration"),
    ("Box box, I have a problem with the car.", "generic"),
    ("Something is wrong, something is wrong.", "generic"),
    ("We have front wing damage, box this lap.", "damage"),
    ("Smoke coming from the cockpit.", "leak_fire"),
])
def test_radio_problem_reports(text, issue):
    got, score, quote = classify(text)
    assert got == issue and score >= 0.5 and quote


@pytest.mark.parametrize("text", [
    "No problem, the car feels great.",
    "I have no issues with the brakes.",
    "Nothing wrong, everything is fine.",
    "I'm not losing power, it was just the mode.",
    "Brake balance plus two, please.",
    "Switch to engine mode 5 and use the power mode on the straight.",
    "Do you have any problem with the car?",
    "If you have damage, let us know.",
    "Box this lap for softs, undercut is on.",
    "Great job, the car is on fire today, well done.",
    "Gap to Leclerc is 2.3, tyres are degrading.",
    "Understood.",
    "",
])
def test_radio_negation_and_routine_talk_stay_silent(text):
    assert classify(text)[1] < 0.3


def _radio_state(texts: dict[float, tuple[str, str]], now: float):
    """A state whose radio store holds transcripts (car, text) at the given publish times."""
    st = RaceState({})
    st.feeds.radio = RadioStore(None, latency_s=15.0)
    for t, (car, _) in texts.items():
        st.feeds.radio.add(t, {"Captures": [{"Utc": "2025-01-01T00:00:00Z", "RacingNumber": car, "Path": f"TeamRadio/{car}_{int(t)}.mp3"}]})
    by_path = {f"TeamRadio/{c}_{int(t)}.mp3": tx for t, (c, tx) in texts.items()}
    st.feeds.radio.transcripts.get = by_path.get
    st.feeds.radio.now = now
    st.t = now
    return st


def test_radio_text_appears_only_after_the_latency():
    texts = {500.0: ("16", "I'm losing power.")}
    eng = MechanicRadio(Context(prior=PitLossPrior()), PitWall(Context(prior=PitLossPrior()), engineers=[]).memory)
    early = _radio_state(texts, 510.0)  # 10 s after the message: not yet transcribed
    assert eng.car(early, "16", None)["radio_risk"] == 0.0
    late = _radio_state(texts, 520.0)
    v = eng.car(late, "16", None)
    assert v["radio_issue"] == "power" and v["radio_risk"] > 0.6 and v["radio_t"] == 500.0
    assert v["radio_since"] == 515.0 and "losing power" in v["radio_quote"]
    # the risk decays with age
    old = _radio_state(texts, 500.0 + 2400)
    assert eng.car(old, "16", None)["radio_risk"] < v["radio_risk"] / 3


# --------------------------------------------------------------------------- race control
def test_race_control_stopped_car_messages():
    assert rc_car_problem("CAR 55 (SAI) STOPPED ON TRACK") == "55"
    assert rc_car_problem("SLOW CAR 14 (ALO) AT TURN 4") == "14"
    assert rc_car_problem("INCIDENT INVOLVING CAR 31 (OCO) NOTED - DRIVING UNNECESSARILY SLOWLY") is None
    assert rc_car_problem("FIA STEWARDS: 10 SECOND STOP/GO PENALTY FOR CAR 4") is None
    assert rc_car_problem("DRS ENABLED") is None
