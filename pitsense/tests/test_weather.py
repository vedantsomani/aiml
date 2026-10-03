"""Weather engineer: weather series features, nowcast fallback, crossover from laps (synthetic)."""

from __future__ import annotations

from pitsense.asof import RaceSummary
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall.engineer import Context
from pitsense.pitwall.engineers.weather import WeatherEngineer, Series, series_features
from pitsense.pitwall.memory import RaceMemory
from pitsense.pitwall.wall import PitWall
from pitsense.state import LapRecord, RaceState

from .conftest import make_log, synthetic_events


def _series(rows):
    s = Series()
    for t, rain, trk, air, hum in rows:
        s.add(t, (rain, trk, air, hum))
    return s


def test_series_features_are_as_of_and_never_none():
    s = _series([(0, 0.0, 40.0, 25.0, 50.0), (600, 0.0, 38.0, 24.0, 55.0), (900, 1.0, 35.0, 23.0, 60.0), (1200, 0.0, 34.0, 23.0, 60.0)])
    f = series_features(s, 950, 30.0)
    assert f["rain_now"] == 1.0 and f["run_min"] == 0.0 + (950 - 900) / 60 and f["rc_known"] == 1.0 and f["rc_risk"] == 0.3
    assert f["trk_tr"] == 35.0 - 40.0  # against the reading 10 min earlier (t=350 -> sample at 0)
    g = series_features(s, 1500, None)  # the flag dropped at 1200; later samples don't exist
    assert g["rain_now"] == 0.0 and g["since_rain_min"] == (1500 - 1200) / 60 and g["rain_seen"] == 1.0
    assert all(isinstance(v, float) for v in series_features(Series(), 10, None).values())
    h = series_features(s, 500, None)  # nothing after t=500 is visible
    assert h["rain_seen"] == 0.0 and h["hum"] == 0.5


def _lap(drv, lap, lt, comp, t0=0.0, **kw):
    return LapRecord(driver=drv, lap=lap, t_start=None, t_end=t0 + lap * 90.0, lap_time=lt, lap_time_t=t0 + lap * 90.0,
                     position=None, gap_to_leader=None, laps_down=0, interval=None, compound=comp, tyre_age=None,
                     stint=1, is_in_lap=False, is_out_lap=False, track_status="1", **kw)


def _crossover(wet_adv: float):
    """Two cars on inters, one on mediums; inters are ``wet_adv`` s slower per lap."""
    w = WeatherEngineer(Context(), RaceMemory())
    st = RaceState({})
    for lap in range(2, 8):
        st.laps += [_lap("1", lap, 100.0, "MEDIUM"), _lap("2", lap, 100.0 + wet_adv, "INTERMEDIATE"),
                    _lap("3", lap, 100.2 + wet_adv, "INTERMEDIATE")]
    st.current_lap = 8
    st.t = 700.0
    return w._crossover(st)["call"]


def test_crossover_direction():
    assert _crossover(-3.0) == "to_inters"  # inters faster
    assert _crossover(+3.0) == "to_slicks"  # slicks faster
    assert _crossover(0.1) == "none"


def test_crossover_ignores_pit_and_neutralised_laps():
    w = WeatherEngineer(Context(), RaceMemory())
    st = RaceState({})
    for lap in range(2, 8):
        a, b = _lap("1", lap, 100.0, "MEDIUM"), _lap("2", lap, 90.0, "INTERMEDIATE")
        b.is_out_lap = True  # every wet lap is an out-lap: no evidence
        st.laps += [a, b]
    st.current_lap = 8
    assert w._crossover(st)["call"] == "none"


def test_values_without_history_and_with_a_summary():
    log = make_log(synthetic_events() + [(150.0, "WeatherData", {"Rainfall": "0", "TrackTemp": "40", "AirTemp": "25", "Humidity": "50"})])
    wall = PitWall(Context(prior=PitLossPrior()), RaceMemory())
    st = RaceState(log.meta)
    for e in log.events:
        st.apply(e)
        wall.observe(st)
    v = wall.view(st).race("weather")
    assert v["rainfall"] == 0.0 and v["crossover"] == "none" and 0 < v["rain_prob_10min"] < 0.2
    assert v["rain_minutes"] == 0.0 and v["wet_running"] is False
    assert not [a for a in wall.alerts(st) if a.engineer == "weather"]


def test_nowcast_learns_from_past_races_only():
    from datetime import datetime, timedelta

    # a past race where the flag is up half the time and the next 10 minutes mirror the flag
    samples = []
    for k in range(60):
        raining = 1.0 if (k // 10) % 2 else 0.0
        nxt = 1 if (k // 10) % 2 or ((k + 5) // 10) % 2 else 0
        samples.append([raining, -1.0 if not raining else 3.0, 5.0 * raining, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, raining, nxt])
    t0 = datetime(2025, 1, 1)
    past = tuple(RaceSummary(f"r{i}", 2025, 7, t0 + timedelta(days=i), t0 + timedelta(days=i, hours=2), [], [], [], 50, 0, 0,
                             {"weather": {"rained": 1, "rain_frac": 0.5, "wet_laps": 10, "samples": samples}}) for i in range(3))
    w = WeatherEngineer(Context(past_races=past, meta={"circuit_key": 7}), RaceMemory())
    f = series_features(_series([(0, 1.0, 40, 25, 60)]), 300, None)
    assert w._prob(f) > 0.7
    dry = series_features(_series([(0, 0.0, 40, 25, 60)]), 300, None)
    assert w._prob(dry) < w._prob(f)
    assert w._model is not None and w._prior_rate > 0.5


def test_alert_when_rain_is_likely():
    class Fake(WeatherEngineer):
        def _prob(self, f):
            return 0.6

    w = Fake(Context(), RaceMemory())
    st = RaceState({})
    st.weather = {"Rainfall": 0.0, "TrackTemp": 30.0, "AirTemp": 20.0, "Humidity": 70.0}
    st.t = 100.0
    w.observe(st)

    class V:
        def race(self, name):
            return {"rc_rain_risk": 40.0} if name == "rules" else w.race(st, self)

    codes = [a.code for a in w.alerts(st, V())]
    assert codes == ["rain_onset_likely"]
