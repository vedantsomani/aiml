"""Weekend tools on synthetic data: tyre sets, qualifying engineer, quali benchmark rows, fetch --weekend."""

from __future__ import annotations

import json

import pytest

from pitsense import archive, quali as Q, weekend
from pitsense.archive import SessionRef
from pitsense.bench import quali as BQ
from pitsense.events import Event, EventLog
from pitsense.pitwall import Context, PitWall
from pitsense.pitwall.engineers.quali import QualiEngineer
from pitsense.pitwall.engineers.tyresets import TyreSetsEngineer
from pitsense.state import RaceState


# ------------------------------------------------------------------ tyre sets
def stint(compound, new, start, total, same="0"):
    return {"Compound": compound, "New": new, "TyresNotChanged": same, "StartLaps": start, "TotalLaps": total}


def test_sets_new_used_and_matching():
    sets: list = []
    weekend.add_stints(sets, [stint("SOFT", "true", 0, 5), stint("MEDIUM", "true", 0, 3),
                              stint("SOFT", "false", 5, 9)], "FP1")  # the soft is run again: still one set
    assert [(t.compound, t.laps, t.first) for t in sets] == [("SOFT", 9, "FP1"), ("MEDIUM", 3, "FP1")]
    weekend.add_stints(sets, [stint("HARD", "false", 4, 8), stint("INTERMEDIATE", "true", 0, 2)], "FP2")
    assert [t.compound for t in sets] == ["SOFT", "MEDIUM", "HARD"]  # unknown-origin used hard; wets skipped
    left = weekend.remaining(sets, weekend.allocation())
    assert left == {"SOFT": 7, "MEDIUM": 2, "HARD": 2}  # the used hard of unknown origin takes no new set
    assert weekend.used_str(sets) == "S9 M3 H8"


def test_allocation_sprint_and_returned():
    assert sum(weekend.allocation().values()) == 13
    assert sum(weekend.allocation(sprint=True).values()) == 12
    assert weekend.allocation(returned={"SOFT": 1})["SOFT"] == 7


def test_tyresets_engineer_takes_race_stints_off(race_log):
    pre = {"11": [["SOFT", 12, "FP1"], ["SOFT", 3, "Qualifying"], ["MEDIUM", 20, "FP2"]]}
    ctx = Context(meta={**race_log.meta, "weekend_sets": pre})
    wall = PitWall(ctx, engineers=[TyreSetsEngineer])
    state = RaceState(race_log.meta)
    for e in race_log.events:
        state.apply(e)
        wall.observe(state)
    end = wall.car_values(state, "11")
    assert end["tyresets__new_soft_left"] == 6
    assert end["tyresets__new_medium_left"] == 1  # the new FP2 medium + the new race medium are gone
    assert end["tyresets__new_hard_left"] == 2
    # car 11 starts the synthetic race on a NEW medium: not the used FP2 one, which stays free
    assert end["tyresets__used_sets"] == "S3 S12 M20"
    # a car absent from the earlier sessions: only its own race sets are counted
    c22 = wall.car_values(state, "22")
    assert c22["tyresets__sets_run"] == 2 and c22["tyresets__new_soft_left"] == 8


def test_tyresets_unknown_without_earlier_sessions(race_log):
    wall = PitWall(Context(meta=dict(race_log.meta)), engineers=[TyreSetsEngineer])
    state = RaceState(race_log.meta)
    for e in race_log.events:
        state.apply(e)
        wall.observe(state)
    assert wall.car_values(state, "11")["tyresets__new_soft_left"] is None


def _index(monkeypatch):
    def ref(name, day, key, hour=10):
        return SessionRef(2026, 7, "Test Grand Prix", "Test", "Land", 1, "Test", 1, key, name, "Race" if name in ("Race", "Sprint") else name,
                          f"2026-05-{day:02d}T{hour:02d}:00:00", "00:00:00", f"2026/m/{day}_{name.replace(' ', '_')}/")
    refs = [ref("Practice 1", 1, 1), ref("Sprint Qualifying", 1, 2), ref("Sprint", 2, 3), ref("Qualifying", 2, 4, 14), ref("Race", 3, 5)]
    monkeypatch.setattr(archive, "fetch_index", lambda year, refresh=False: refs)
    return refs


def test_weekend_sessions_are_the_earlier_ones(monkeypatch):
    refs = _index(monkeypatch)
    race = refs[-1]
    assert [r.session_name for r in weekend.before(race)] == ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying"]
    assert [r.session_name for r in weekend.before(refs[3])] == ["Practice 1", "Sprint Qualifying", "Sprint"]  # not the race
    assert weekend.is_sprint_weekend(race)


def test_download_weekend_skips_the_race(monkeypatch):
    refs = _index(monkeypatch)
    got = []
    monkeypatch.setattr(archive, "download_session", lambda ref, topics, force=False: got.append((ref.session_name, topics)))
    weekend.download_weekend(refs[-1], ("TimingData",), feed_topics=("Position.z",))
    assert [n for n, _ in got] == ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying"]
    assert dict(got)["Qualifying"] == ("TimingData", "Position.z") and dict(got)["Sprint"] == ("TimingData",)


# ------------------------------------------------------------------ qualifying
def fmt(sec):
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:06.3f}"


CARS = ["1", "2", "3", "4", "5", "6"]


def quali_log(cut_at=None, end=True):
    """Six cars, NoEntries [6, 4, 2]: Q1 600 s long. Times improve as the track evolves."""
    ev = []
    add = lambda t, topic, data: ev.append(Event(t, topic, data, len(ev)))  # noqa: E731
    add(0.0, "SessionInfo", {"Name": "Qualifying", "Type": "Qualifying"})
    add(0.5, "DriverList", {n: {"RacingNumber": n, "Tla": f"C{n}", "TeamName": f"T{n}"} for n in CARS})
    add(1.0, "TimingData", {"NoEntries": [6, 4, 2], "SessionPart": 1, "Lines": {
        n: {"Position": str(i + 1), "KnockedOut": False, "InPit": True, "PitOut": False, "NumberOfLaps": 0,
            "BestLapTimes": [{"Value": ""}, {}, {}]} for i, n in enumerate(CARS)}})
    add(1.0, "ExtrapolatedClock", {"Utc": "2026-01-01T10:00:00Z", "Remaining": "00:10:00"})
    add(10.0, "SessionStatus", {"Status": "Started"})
    add(10.0, "ExtrapolatedClock", {"Utc": "2026-01-01T10:00:10Z", "Remaining": "00:10:00", "Extrapolating": True})
    # (time, car, lap time): car 6 never sets one until late
    laps = [(100, "1", 90.0), (130, "2", 90.5), (180, "3", 91.0), (230, "4", 91.5), (300, "5", 92.0),
            (400, "2", 89.9), (480, "5", 90.6), (540, "4", 90.7), (560, "6", 91.8)]
    for t, n, lt in laps:
        add(float(t), "TimingData", {"Lines": {n: {"InPit": False, "NumberOfLaps": 1, "BestLapTimes": {"0": {"Value": fmt(lt)}}}}})
    if end:
        add(610.0, "SessionStatus", {"Status": "Finished"})
        add(620.0, "TimingData", {"SessionPart": 2, "Lines": {n: {"KnockedOut": n in ("3", "6")} for n in CARS}})
    log = EventLog(ev, {"session_name": "Qualifying"})
    return log.until(cut_at) if cut_at is not None else log


def run_wall(log, until=None):
    wall = PitWall(Context(meta=dict(log.meta)), engineers=[QualiEngineer])
    state = RaceState(log.meta)
    for e in log.events:
        if until is not None and e.t > until:
            break
        state.apply(e)
        wall.observe(state)
    return state, wall


def test_cut_line_and_gaps():
    state, wall = run_wall(quali_log(), until=500)  # the last message so far is at 480 s
    race = wall.race_values(state)
    # best times: 2 89.9, 1 90.0, 5 90.6, 3 91.0, 4 91.5; car 6 untimed; cut after P4 -> car 3 (91.0)
    assert race["quali__part"] == 1 and race["quali__cut_pos"] == 4
    assert race["quali__cut_time_s"] == 91.0
    assert race["quali__time_left_s"] == pytest.approx(600 - (480 - 10), abs=0.5)
    c5, c2, c6 = (wall.car_values(state, n) for n in ("4", "2", "6"))
    assert c5["quali__in_zone"] and c5["quali__gap_to_cut_s"] == pytest.approx(0.5)  # 91.5 - 91.0
    assert c2["quali__rank"] == 1 and not c2["quali__in_zone"] and c2["quali__gap_to_cut_s"] == pytest.approx(-1.1)
    assert c6["quali__best_s"] is None and c6["quali__in_zone"]


def test_alerts_and_send_now():
    state, wall = run_wall(quali_log(), until=500)  # 130 s left at 480 s
    msgs = {a.car: a for a in wall.alerts(state)}
    assert "6" in msgs and "drop zone" in msgs["6"].message and "2:10 left" in msgs["6"].message
    assert "needs one run" in msgs["6"].message and "SEND NOW" in msgs["6"].message
    car6 = wall.car_values(state, "6")
    # 130 s left, out-lap about 1.3 x 89.9 = 117 s: 13 s of slack, so go now
    assert car6["quali__send"] == "SEND_NOW" and car6["quali__laps_needed"] == 2
    # 50 s left: no time for an out-lap any more
    state3, wall3 = run_wall(quali_log(), until=565)
    assert wall3.car_values(state3, "6")["quali__send"] == "TOO_LATE" or wall3.car_values(state3, "6")["quali__send"] == "ON_TRACK"
    # earlier, with 4 minutes left, the same car still has time: wait or send, never too late
    state2, wall2 = run_wall(quali_log(), until=380)
    assert wall2.car_values(state2, "6")["quali__send"] in ("WAIT", "SEND_NOW")
    assert wall2.alerts(state2) == []  # more than 3 minutes left: no alert yet


def test_alert_since_does_not_depend_on_how_often_we_ask():
    log = quali_log()
    state, wall = run_wall(log, until=470)
    once = [(a.car, a.since) for a in wall.alerts(state)]
    state, wall = run_wall(log, until=None)
    wall2 = PitWall(Context(meta=dict(log.meta)), engineers=[QualiEngineer])
    state2 = RaceState(log.meta)
    seen = []
    for e in log.events:
        if e.t > 470:
            break
        state2.apply(e)
        wall2.observe(state2)
        seen = [(a.car, a.since) for a in wall2.alerts(state2)]  # asked after every event
    assert once == seen


def test_as_of_future_does_not_change_the_answer():
    full = quali_log()
    state, wall = run_wall(full, until=450)
    a = (wall.race_values(state), {n: wall.car_values(state, n) for n in CARS}, wall.alerts(state))
    cut = EventLog([e for e in full.events if e.t <= 450], full.meta)
    scrambled = EventLog(cut.events + [Event(451.0 + i, "TimingData", {"Lines": {"6": {"BestLapTimes": {"0": {"Value": "1:00.000"}}}}}, 900 + i)
                                      for i in range(3)], full.meta)
    s2, w2 = run_wall(cut)
    b = (w2.race_values(s2), {n: w2.car_values(s2, n) for n in CARS}, w2.alerts(s2))
    assert a == b
    assert json.dumps(a[0], allow_nan=False)
    assert scrambled  # events after 450 exist but are never applied by run_wall(until=450)


def test_part_change_moves_the_cut_and_drops_knocked_out_cars():
    state, wall = run_wall(quali_log())
    race = wall.race_values(state)
    assert race["quali__part"] == 2 and race["quali__cut_pos"] == 2
    assert wall.car_values(state, "3") == {"quali__eligible": False}
    assert wall.car_values(state, "1")["quali__eligible"]


def test_silent_outside_qualifying(race_log):
    wall = PitWall(Context(meta=dict(race_log.meta)), engineers=[QualiEngineer])
    state = RaceState(race_log.meta)
    for e in race_log.events:
        state.apply(e)
        wall.observe(state)
    assert wall.race_values(state) == {} and wall.car_values(state, "11") == {}


def test_benchmark_rows_and_labels():
    cut, ko = BQ.collect_events(quali_log(), "synthetic", 2026, False)
    assert cut and ko
    assert {r["y_cut"] for r in cut} == {90.7}  # the final cut time of part 1
    assert all(r["tl"] <= 600 for r in cut) and all(r["t"] >= 10 for r in cut)
    knocked = {r["car"] for r in ko if r["y_ko"]}
    assert knocked == {"3", "6"}  # final order 2, 1, 5, 4 | 3, 6
    last = [r for r in ko if r["t"] == max(x["t"] for x in ko)]
    assert {r["car"]: r["y_ko"] for r in last}["6"] == 1


def test_predict_cut_uses_the_model_table(monkeypatch):
    state, wall = run_wall(quali_log(), until=500)
    eng = wall.engineer("quali")
    pic = Q.read(state, eng.tracker)
    monkeypatch.setitem(Q._MODEL_CACHE, "m", {"evolution": {"1": {str(Q.time_bin(pic.time_left)): -0.4}}, "config": {}})
    assert Q.predict_cut(pic) == pytest.approx(90.6)
    monkeypatch.setitem(Q._MODEL_CACHE, "m", {})
    assert Q.predict_cut(pic) == 91.0  # no model: the cut now


def test_runtime_uses_the_qualifying_engineer():
    from pitsense.pitwall.runtime import PitWallRuntime, ReplaySource

    rt = PitWallRuntime(ReplaySource(quali_log(), 0), models=False, history=False)
    rt.start()
    assert rt.wait(30)
    assert rt.status == "finished", rt.error
    assert rt.latest["race"]["quali__part"] == 2
    assert any(k.startswith("quali__") for k in rt.latest["cars"]["1"])
    assert not any(k.startswith("head__") for k in rt.latest["race"])


# ------------------------------------------------------------------ plans respect the sets
def _field(total=57, A=20):
    import numpy as np
    from types import SimpleNamespace

    return SimpleNamespace(A=A, total=total, R=total - A, life=np.array([18.0, 28.0, 38.0]), used=np.zeros((1, 3), dtype=bool),
                           must=np.array([False]), reg_min_stops=0, max_stint=0.0)


def _info(sets):
    return {"stops_done": {"1": 0}, "stint_left": {"1": None}, "sets": {"1": sets}}


def test_candidates_only_use_compounds_the_car_has():
    from pitsense.pitwall.engineers.strategy import analysis as AN

    F = _field()
    free = AN.candidates(F, 0, _info(None), "1")
    sets = {"avail": [2, 1, 0], "new": [2, 1, 0], "used": [0, 0, 0], "used_laps": [None] * 3}
    info = _info(sets)
    got = AN.candidates(F, 0, info, "1")
    assert 0 < len(got) < len(free)
    assert all(cj != 2 for p in got for _, cj in p)  # no hard set left
    assert all(sum(1 for _, cj in p if cj == 1) <= 1 for p in got)  # one medium only
    assert "HARD" in info["sets_note"]["1"]
    assert "sets_note" not in _info(None)


def test_candidates_unchanged_when_nothing_known_and_never_empty():
    from pitsense.pitwall.engineers.strategy import analysis as AN

    F = _field()
    assert AN._car_sets({"sets_basis": "allocation", "avail_soft": 8}) is None
    assert AN._car_sets({}) is None
    none_left = {"avail": [0, 0, 0], "new": [0, 0, 0], "used": [0, 0, 0], "used_laps": [None] * 3}
    info = _info(none_left)
    F.must[0] = True  # a stop is compulsory, yet no set is left: the constraint cannot hold
    assert AN.candidates(F, 0, info, "1") == AN.candidates(F, 0, _info(None), "1")  # a constraint that blocks everything is dropped
    assert "not applied" in info["sets_note"]["1"]


def test_tyresets_fall_back_to_allocation(race_log):
    wall = PitWall(Context(meta=dict(race_log.meta)), engineers=[TyreSetsEngineer])
    state = RaceState(race_log.meta)
    for e in race_log.events:
        state.apply(e)
        wall.observe(state)
    v = wall.car_values(state, "11")
    assert v["tyresets__sets_basis"] == "allocation" and v["tyresets__new_soft_left"] is None
    assert v["tyresets__avail_soft"] >= 0 and v["tyresets__avail_hard"] >= 0


def test_used_offset_model():
    m = {"used": -0.03, "per_lap": 0.02}
    assert weekend.used_offset(None, 8) == 0.0 and weekend.used_offset(m, 0) == 0.0
    assert weekend.used_offset(m, 5) == pytest.approx(0.07)
    assert weekend.used_offset(m, 50) == pytest.approx(-0.03 + 0.02 * weekend.OFFSET_CAP)
    assert weekend.used_offset({"used": -1.0, "per_lap": 0.0}, 3) == 0.0  # never faster than new


def test_availability_counts_free_used_sets():
    sets = [weekend.TyreSet("MEDIUM", 12, "Practice 2"), weekend.TyreSet("MEDIUM", 3, "Qualifying"), weekend.TyreSet("HARD", 9, "")]
    sets[1].mounted = True
    av = weekend.availability(sets, {"SOFT": 8, "MEDIUM": 3, "HARD": 2})
    assert av["MEDIUM"] == {"new": 1, "used": 1, "used_laps": 12}
    assert av["HARD"] == {"new": 2, "used": 1, "used_laps": 9} and av["SOFT"]["used_laps"] is None
