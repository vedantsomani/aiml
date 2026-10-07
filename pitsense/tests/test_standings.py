"""Championship points by era, classification, the as-of season filter and the projection."""

from datetime import datetime, timezone
from types import SimpleNamespace as NS

from pitsense import standings as ST


def test_points_follow_the_era():
    assert ST.points_for(1, 2026) == 25 and ST.points_for(10, 2026) == 1 and ST.points_for(11, 2026) == 0
    assert ST.points_for(5, 2024, fastest=True) == 11 and ST.points_for(11, 2024, fastest=True) == 0
    assert ST.points_for(5, 2025, fastest=True) == 10  # no fastest-lap point from 2025
    assert ST.points_for(1, 2026, sprint=True) == 8 and ST.points_for(1, 2021, sprint=True) == 3
    assert ST.points_for(1, 2018, fastest=True) == 25


def test_classification_needs_ninety_percent_of_the_laps():
    st = NS(drivers={"1": NS(number="1", tla="AAA", team="T", position=1, laps=50, best_lap_time=90.0),
                     "2": NS(number="2", tla="BBB", team="U", position=2, laps=44, best_lap_time=89.5),
                     "3": NS(number="3", tla="CCC", team="U", position=3, laps=45, best_lap_time=None)})
    c = {r["tla"]: r for r in ST.classification(st)}
    assert c["AAA"]["classified"] and not c["BBB"]["classified"] and c["CCC"]["classified"] and c["BBB"]["fastest"]


def test_season_uses_only_sessions_that_ended_before_and_projects_the_race():
    res = [{"session": "Race", "end_utc": "2026-03-01T15:00:00+00:00",
            "results": [{"tla": "AAA", "team": "T", "pos": 1, "classified": True, "fastest": False},
                        {"tla": "BBB", "team": "U", "pos": 2, "classified": True, "fastest": False}]},
           {"session": "Race", "end_utc": "2026-03-20T15:00:00+00:00",
            "results": [{"tla": "BBB", "team": "U", "pos": 1, "classified": True, "fastest": False}]}]
    pts = ST.season_points(ST.before(res, datetime(2026, 3, 10, tzinfo=timezone.utc)), 2026)
    assert pts == {"AAA": {"points": 25, "team": "T"}, "BBB": {"points": 18, "team": "U"}}  # the later race is unknown
    proj = {r["tla"]: r for r in ST.project(pts, [("BBB", "U"), ("CCC", "V"), ("AAA", "T")], 2026)}
    assert proj["BBB"]["projected"] == 43 and proj["BBB"]["rank"] == 1 and proj["BBB"]["delta"] == 1
    assert proj["AAA"]["projected"] == 40 and proj["AAA"]["delta"] == -1
    assert proj["CCC"]["projected"] == 18 and proj["CCC"]["rank_now"] is None and proj["CCC"]["delta"] is None
