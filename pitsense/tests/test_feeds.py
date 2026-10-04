"""Telemetry, positions and radio: synthetic payloads only (no timing data or audio)."""

from __future__ import annotations

import base64
import json
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pitsense import trackmap
from pitsense.asof import LeakageError
from pitsense.events import Event, decode_z, load_archive_session
from pitsense.feeds import RADIO_LATENCY_S, RadioStore, Ring, TelemetryStore, parse_utc
from pitsense.state import RaceState

UTC0 = datetime(2025, 8, 3, 12, 0, 0, tzinfo=timezone.utc)


def utc(s: float) -> str:
    return (UTC0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "0Z"  # 7 fractional digits, as the feed


def pack(obj) -> str:
    c = zlib.compressobj(wbits=-15)
    return base64.b64encode(c.compress(json.dumps(obj).encode()) + c.flush()).decode()


def car_data(entries: list[tuple[float, dict]]) -> dict:
    return {"Entries": [{"Utc": utc(u), "Cars": {c: {"Channels": {"0": 9000, "2": spd, "3": 5, "4": 80, "5": 0, "45": 12}}
                                                  for c, spd in cars.items()}} for u, cars in entries]}


def positions(entries: list[tuple[float, dict]]) -> dict:
    return {"Position": [{"Timestamp": utc(u), "Entries": {c: {"Status": "OnTrack", "X": x, "Y": y, "Z": 0}
                                                           for c, (x, y) in cars.items()}} for u, cars in entries]}


def test_decode_z_round_trip():
    payload = car_data([(0.0, {"1": 300})])
    assert decode_z(pack(payload)) == payload
    assert decode_z(payload) is payload  # already decoded
    assert parse_utc(utc(1.5)) == UTC0.timestamp() + 1.5


def test_entries_not_visible_before_their_message():
    s = TelemetryStore()
    # message published at t=100 carries samples measured at 97 and 99 (on the UTC0 scale)
    s.add_car_data(100.0, car_data([(97.0, {"1": 200}), (99.0, {"1": 210})]))
    s.add_position(100.2, positions([(99.0, {"1": (5, 6)})]))
    assert len(s.telemetry("1", None)["speed"]) == 2
    with pytest.raises(LeakageError):
        s.telemetry("1", None, t=101.0)  # beyond what has been published
    s.add_car_data(110.0, car_data([(108.0, {"1": 250})]))
    # as of t=105 only the first message's samples exist, though the store holds more
    first = s.telemetry("1", None, t=105.0)
    assert list(first["speed"]) == [200, 210] and first["t"].max() == 100.0
    assert s.latest_position(t=100.2)["1"]["x"] == 5
    with pytest.raises(LeakageError):
        s.latest_position(t=200.0)
    # a sample measured earlier than its message is still stamped with the message time
    assert (s.telemetry("1", None)["utc"] - UTC0.timestamp() <= s.telemetry("1", None)["t"]).all()


def test_window_and_latest():
    s = TelemetryStore()
    for k in range(20):
        s.add_car_data(10.0 + k, car_data([(k, {"1": 100 + k, "2": 50})]))
        s.add_position(10.0 + k, positions([(k, {"1": (k, 2 * k)})]))
    assert list(s.telemetry("1", last_s=5.0)["speed"]) == [114, 115, 116, 117, 118, 119]
    assert list(s.telemetry("1", last_s=5.0, t=15.0)["speed"]) == [100, 101, 102, 103, 104, 105]
    assert s.latest_telemetry()["1"]["speed"] == 119
    assert s.latest_position()["1"]["y"] == 38
    assert s.cars() == ["1", "2"]
    assert len(s.telemetry("99")["t"]) == 0  # unknown car


def test_ring_is_bounded():
    r = Ring(2, 5)
    for i in range(12):
        r.append((i, i))
    assert len(r) == 5 and list(r.rows()[:, 0]) == [7, 8, 9, 10, 11]
    s = TelemetryStore(telemetry_cap=50, position_cap=50)
    for k in range(1000):
        s.add_car_data(k, car_data([(k, {"1": k})]))
        s.add_position(k, positions([(k, {"1": (k, k)})]))
    assert s.nbytes() == 50 * 8 * (8 + 6)  # fixed, however long the race
    tel = s.telemetry("1", None)
    assert len(tel["t"]) == 50 and tel["speed"][-1] == 999


def test_radio_transcript_known_after_latency(tmp_path):
    mp3 = tmp_path / "TeamRadio" / "MAXVER01_1_20250803_150810.mp3"
    mp3.parent.mkdir()
    mp3.write_bytes(b"x")
    mp3.with_suffix(".json").write_text(json.dumps({"text": "Box box"}), encoding="utf-8")
    r = RadioStore(tmp_path)
    cap = {"Utc": utc(5), "RacingNumber": "1", "Path": "TeamRadio/MAXVER01_1_20250803_150810.mp3"}
    r.add(50.0, {"Captures": [cap]})
    r.add(60.0, {"Captures": {"0": cap, "1": {"Utc": utc(9), "RacingNumber": "44", "Path": "TeamRadio/b.mp3"}}})  # repeat ignored
    r.add(70.0, {})
    m = r.messages("1", t=50.0 + RADIO_LATENCY_S - 0.1)
    assert len(m) == 1 and m[0].text is None and m[0].t == 50.0 and m[0].audio.endswith(".mp3")
    assert r.messages("1", t=50.0 + RADIO_LATENCY_S)[0].text == "Box box"
    assert [x.car for x in r.messages(t=55.0)] == ["1"]  # car 44 not published yet
    assert [x.car for x in r.messages(t=70.0)] == ["1", "44"] and r.messages("44")[0].text is None
    with pytest.raises(LeakageError):
        r.messages(t=71.0)


def test_state_routes_feeds_and_keeps_view_unchanged():
    st, st2 = RaceState(), RaceState()
    base = Event(1.0, "SessionStatus", {"Status": "Started"}, 0)
    st.apply(base)
    st2.apply(base)
    st.apply(Event(1.0, "CarData.z", car_data([(1.0, {"1": 300})]), 1))
    st.apply(Event(1.0, "Position.z", pack(positions([(1.0, {"1": (1, 2)})])), 2))  # still packed: decoded on the way in
    st.apply(Event(1.0, "TeamRadio", {"Captures": [{"Utc": utc(1), "RacingNumber": "1", "Path": "TeamRadio/a.mp3"}]}, 3))
    assert "CarData.z" not in st.topics and "TeamRadio" not in st.topics
    assert st.fingerprint() == st2.fingerprint()
    assert st.feeds.telemetry.latest_telemetry()["1"]["speed"] == 300
    assert st.feeds.telemetry.latest_position()["1"]["x"] == 1
    assert len(st.feeds.radio.messages("1")) == 1


def test_archive_loader_decodes_and_skips_feeds_by_default(tmp_path):
    (tmp_path / "TimingData.jsonStream").write_text('00:00:01.000{"Lines":{}}\n', encoding="utf-8")
    (tmp_path / "CarData.z.jsonStream").write_text(
        f'﻿00:00:02.000"{pack(car_data([(1.0, {"1": 300})]))}"\n', encoding="utf-8")
    assert load_archive_session(tmp_path).topics() == {"TimingData"}
    log = load_archive_session(tmp_path, feeds=True)
    e = next(e for e in log if e.topic == "CarData.z")
    assert e.t == 2.0 and e.data["Entries"][0]["Cars"]["1"]["Channels"]["2"] == 300


def test_track_outline_helpers():
    th = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    xy = np.column_stack([1000 * np.cos(th), 1000 * np.sin(th)])
    line = trackmap._resample(np.vstack([xy, xy[:1]]), 100, closed=True)
    assert len(line) == 100
    assert abs(trackmap._length(np.vstack([line, line[:1]])) - 2 * np.pi * 1000) < 20


def test_track_asof_uses_only_finished_races(tmp_path):
    def put(slug, end):
        d = tmp_path / "7"
        d.mkdir(exist_ok=True)
        (d / f"{slug}.json").write_text(json.dumps({"race_id": slug, "end_utc": end.isoformat()}), encoding="utf-8")

    put("a", UTC0 - timedelta(days=400))
    put("b", UTC0 - timedelta(days=30))
    put("c", UTC0 + timedelta(hours=2))  # the race itself or later
    assert trackmap.track_asof(7, UTC0, tmp_path)["race_id"] == "b"
    assert trackmap.track_asof(7, UTC0 - timedelta(days=500), tmp_path) is None
    assert trackmap.track_asof(8, UTC0, tmp_path) is None
