"""Briefing and report on the synthetic race: valid output, and the briefing reads nothing after the start."""

import json
from html.parser import HTMLParser

from pitsense.events import Event
from pitsense.reports import _json, add_commands, briefing, postrace, prerace
from pitsense.pitwall.runtime import shadow_score
from pitsense.state import replay

from .conftest import START, make_log

TEAM = "Team 22"


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.attrs = [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs += attrs


def check_html(text: str) -> Tags:
    p = Tags()
    p.feed(text)
    assert text.startswith("<!doctype html>") and "</html>" in text
    assert p.tags.count("svg") >= 1
    assert "script" not in p.tags and "link" not in p.tags and "img" not in p.tags
    urls = [v for k, v in p.attrs if k in ("src", "href") and v]
    assert not urls, urls  # self-contained: nothing is fetched
    return p


def test_briefing_synthetic():
    data = prerace.prerace(make_log(), TEAM, history=None, sims=48)
    assert data["cars"] == ["22"] and data["as_of"]["cut"] and data["total_laps"] == 8
    assert data["plans"]["ok"] and data["plans"]["cars"]["22"]["plans"]
    assert json.loads(_json(data))["team"] == TEAM
    check_html(briefing.render(data))


def test_briefing_is_as_of():
    log = make_log()
    start = prerace.session_start(log)
    assert start == START
    full = _json(prerace.prerace(log, TEAM, history=None, sims=48))
    cut = _json(prerace.prerace(log.until(start), TEAM, history=None, sims=48))
    assert full == cut
    # and the future cannot change it: scramble everything after the start
    bad = [e if e.t <= start else Event(e.t, e.topic, {"Status": "Finished"} if e.topic == "SessionStatus" else e.data, e.seq, e.kind) for e in log.events]
    bad = [Event(e.t, "LapCount", {"CurrentLap": 99, "TotalLaps": 99}, e.seq, e.kind) if e.t > start and e.topic == "LapCount" else e for e in bad]
    log2 = type(log)(bad, dict(log.meta))
    assert _json(prerace.prerace(log2, TEAM, history=None, sims=48)) == full


def test_report_synthetic_replay_and_calls_log():
    log = make_log()
    rep = postrace.build_report(log, TEAM, sims=48)
    check_html(postrace.render(rep))
    assert json.loads(_json(rep))["cars"] == ["22"]
    story = rep["cars_story"]["22"]
    assert [s["in_lap"] for s in story["stops"]] == [3]
    # a hand-written calls log: graded per call exactly as shadow-score does
    calls = [
        {"kind": "call", "t": 1.0, "lap": 2, "car_lap": 2, "car": "22", "action": "BOX", "compound": "HARD"},
        {"kind": "call", "t": 2.0, "lap": 3, "car_lap": 3, "car": "22", "action": "STAY_OUT"},
        {"kind": "call", "t": 3.0, "lap": 6, "car_lap": 6, "car": "22", "action": "PREPARE_BOX", "compound": "SOFT"},
        {"kind": "call", "t": 3.0, "lap": 2, "car_lap": 2, "car": "11", "action": "STAY_OUT"},
        {"kind": "alert", "t": 4.0, "lap": 4, "engineer": "mechanic_chief", "code": "x", "severity": "warn", "message": "m", "car": "22", "since": 3.0},
    ]
    rep2 = postrace.build_report(log, TEAM, calls=calls, sims=48)
    g = rep2["grading"]
    assert [c["right"] for c in g["calls"]] == [True, False, False]  # BOX l2 (stop l3), STAY_OUT l3 (stop l3), PREPARE l6
    ref = shadow_score(calls, replay(log), 2, {"22"})
    assert ref["box_precision"] == g["summary"]["box_precision"] == 0.5
    assert ref["stay_out_accuracy"] == g["summary"]["stay_out_accuracy"] == 0.0
    assert rep2["mechanic_alerts"][0]["team"] is True
    check_html(postrace.render(rep2))


def test_cli_registered():
    import argparse

    p = argparse.ArgumentParser()
    add_commands(p.add_subparsers())
    assert p.parse_args(["briefing", "--race", "x", "--team", "t"]).sims == 288
    assert p.parse_args(["report", "--race", "x", "--team", "t", "--log", "c.jsonl"]).k == 2
