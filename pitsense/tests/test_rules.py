"""Rules engineer: parser, penalty bookkeeping, regulation table, as-of values (synthetic messages)."""

from __future__ import annotations

from pitsense.pitwall.engineer import Context
from pitsense.pitwall.engineers.rules import RulesEngineer
from pitsense.pitwall.engineers.rules.book import RCBook
from pitsense.pitwall.engineers.rules.parser import parse
from pitsense.pitwall.engineers.rules.regs import rule_for
from pitsense.pitwall.memory import RaceMemory
from pitsense.pitwall.wall import PitWall
from pitsense.state import RaceState, replay

from .conftest import make_log, synthetic_events


def test_parse_penalties_and_served():
    e = parse("FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 7 (XXX) - SPEEDING IN THE PIT LANE (12:01:02)")
    assert (e.kind, e.car, e.pen, e.seconds, e.reason) == ("penalty", "7", "time", 10, "SPEEDING IN THE PIT LANE")
    e = parse("FIA STEWARDS: PENALTY SERVED - 5 SECOND TIME PENALTY FOR CAR 7 (XXX) - TRACK LIMITS (2ND OFFENCE)")
    assert (e.kind, e.seconds, e.reason) == ("penalty_served", 5, "TRACK LIMITS")
    e = parse("FIA STEWARDS: DRIVE THROUGH PENALTY FOR CAR 9 (YYY) - SOMETHING")
    assert (e.kind, e.pen, e.seconds) == ("penalty", "drive_through", None)
    e = parse("FIA STEWARDS: 10 SECOND STOP/GO PENALTY FOR CAR 9 (YYY) - SOMETHING")
    assert (e.pen, e.seconds) == ("stop_go", 10)


def test_parse_misc_kinds_and_unknown():
    cases = {
        "SAFETY CAR DEPLOYED": "sc_deployed", "SAFETY CAR IN THIS LAP": "sc_in_this_lap",
        "VIRTUAL SAFETY CAR ENDING": "vsc_ending", "RED FLAG - RACE SUSPENDED": "red_flag",
        "PIT LANE ENTRY CLOSED": "pit_entry_closed", "PIT EXIT OPEN": "pit_exit_open",
        "CHEQUERED FLAG": "chequered", "DRS ENABLED": "drs_enabled",
        "BLACK AND WHITE FLAG FOR CAR 3 (AAA) - TRACK LIMITS": "bw_flag",
        "CAR 3 (AAA) TIME 1:23.456 DELETED - TRACK LIMITS AT TURN 4 LAP 7 10:00:00": "track_limits_deleted",
        "FIA STEWARDS: INCIDENT INVOLVING CAR 3 (AAA) UNDER INVESTIGATION - X": "investigation",
        "FIA STEWARDS: INCIDENT INVOLVING CAR 3 (AAA) REVIEWED NO FURTHER INVESTIGATION": "investigation_closed",
        "SOMETHING NEVER SEEN BEFORE": "unknown", "": "unknown",
    }
    for msg, kind in cases.items():
        assert parse(msg).kind == kind, msg
    assert parse("RISK OF RAIN FOR THE RACE IS 40 %").value == 40.0
    assert parse(None).kind == "unknown"  # type: ignore[arg-type]


def test_penalty_bookkeeping():
    b = RCBook()
    b.add(1, parse("FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 7 (XXX) - A"))
    b.add(2, parse("FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 7 (XXX) - B"))
    b.add(3, parse("FIA STEWARDS: DRIVE THROUGH PENALTY FOR CAR 8 (YYY) - C"))
    assert b.time_pending_s("7") == 15 and b.drive_pending("8") and not b.drive_pending("7")
    b.add(4, parse("FIA STEWARDS: PENALTY SERVED - 10 SECOND TIME PENALTY FOR CAR 7 (XXX) - B"))
    assert b.time_pending_s("7") == 5
    b.add(5, parse("FIA STEWARDS: PENALTY SERVED - DRIVE THROUGH PENALTY FOR CAR 8 (YYY) - C"))
    assert not b.drive_pending("8")
    b.add(6, parse("garbage"))
    assert b.unknown == 1 and b.total == 6


def test_regulation_table():
    assert rule_for({"year": 2025, "meeting_name": "Monaco Grand Prix"}).min_stops == 2
    assert rule_for({"year": 2025, "meeting_name": "Qatar Grand Prix"}).max_stint_laps == 25
    assert rule_for({"year": 2026, "meeting_name": "Monaco Grand Prix"}).min_stops is None
    assert rule_for({"year": 2019}) is None and rule_for({}) is None


def test_engineer_values_follow_messages(race_log):
    ev = synthetic_events()
    ev.append((60.0, "RaceControlMessages", {"Messages": [
        {"Category": "Other", "Message": "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 22 (BBB) - TEST", "Lap": 1}]}))
    ev.append((61.0, "RaceControlMessages", {"Messages": [
        {"Category": "Other", "Message": "PIT LANE ENTRY CLOSED", "Lap": 1}]}))
    log = make_log(ev)
    st = RaceState(log.meta)
    wall = PitWall(Context(meta={"year": 2025, "meeting_name": "Test"}), RaceMemory(), [RulesEngineer])
    for e in log.events:
        st.apply(e)
        wall.observe(st)
    last = wall.car_values(st, "22") | wall.race_values(st)
    assert last["rules__penalty_s_pending"] == 5.0
    assert last["rules__pit_lane_open"] is False
    assert last["rules__sc_phase"] == "none"
    assert wall.car_values(st, "11")["rules__penalty_s_pending"] == 0.0
    assert any(a.code == "penalty_pending" for a in wall.alerts(st))


def test_stint_limit_makes_must_stop():
    log = make_log()
    st = replay(log)
    wall = PitWall(Context(meta={"year": 2025, "meeting_name": "Qatar Grand Prix"}), RaceMemory(), [RulesEngineer])
    wall.observe(st)
    v = wall.car_values(st, "11")
    assert v["rules__stint_laps_left"] is not None
