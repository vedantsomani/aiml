from pitsense.state import RaceState, parse_gap, parse_lap_time, replay


def _lap(state: RaceState, driver: str, lap: int):
    return next(x for x in state.laps if x.driver == driver and x.lap == lap)


def test_parsers():
    assert parse_lap_time("1:26.103") == 86.103
    assert parse_lap_time("") is None
    assert parse_gap("+1.234") == (1.234, 0)
    assert parse_gap("LAP 23") == (0.0, 0)
    assert parse_gap("1L") == (None, 1)
    assert parse_gap("+2 LAPS") == (None, 2)


def test_laps_and_times(race_log):
    s = replay(race_log)
    assert len(s.laps) == 24
    assert all(x.lap_time is None for x in s.laps if x.lap == 1)  # the feed never sends lap 1
    assert _lap(s, "11", 2).lap_time == 90.1


def test_omitted_repeated_lap_time_is_inferred(race_log):
    s = replay(race_log)
    lap8 = _lap(s, "33", 8)
    assert lap8.lap_time == 91.5 and lap8.lap_time_inferred


def test_pit_stop_in_and_out_laps(race_log):
    s = replay(race_log)
    assert len(s.pit_events) == 1
    pe = s.pit_events[0]
    assert (pe.driver, pe.in_lap, pe.out_lap) == ("22", 3, 4)
    assert pe.lane_time is not None and 15 < pe.lane_time < 17
    assert _lap(s, "22", 3).is_in_lap and _lap(s, "22", 4).is_out_lap
    assert not any(x.is_in_lap for x in s.laps if x.driver != "22")


def test_late_tyre_confirmation_is_as_of(race_log):
    s = replay(race_log)
    # lap 4: the stint was still flagged 'tyres not changed' -> what we knew then
    assert _lap(s, "22", 4).compound == "MEDIUM" and _lap(s, "22", 4).tyre_age == 4
    # lap 5: correction arrived; the new set was fitted at the lap-3 stop
    assert _lap(s, "22", 5).compound == "HARD" and _lap(s, "22", 5).tyre_age == 2
    assert s.drivers["22"].compounds_used == ["MEDIUM", "HARD"]
    assert s.drivers["11"].tyre_age == 8


def test_track_status_per_lap(race_log):
    s = replay(race_log)
    assert "4" in _lap(s, "11", 5).track_status.split(",")
    assert _lap(s, "11", 3).track_status == "1"


def test_final_order(race_log):
    s = replay(race_log)
    assert [d.number for d in s.running_order()] == ["11", "33", "22"]


def test_replay_is_deterministic_and_prefix_consistent(race_log):
    a, b = replay(race_log), replay(race_log)
    assert a.fingerprint() == b.fingerprint()
    cut = race_log.events[len(race_log) // 2].t
    assert replay(race_log, until=cut).fingerprint() == replay(race_log.until(cut)).fingerprint()
