"""Live feeds on synthetic recordings: outline from SessionInfo, async radio download + transcription, offline."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request

import pytest

from pitsense.live import RecordingTail, feed_radio
from pitsense.pitwall.runtime import FollowSource, PitWallRuntime
from pitsense.radio import LiveRadio
from pitsense.web.server import serve

from .conftest import synthetic_events
from .test_feeds import positions, utc

RECV0 = 1_754_000_000.0
SESSION_PATH = "2025/2025-08-03_Test_Grand_Prix/2025-08-03_Race/"
INFO = {"Meeting": {"Key": 1, "Name": "Test GP", "Circuit": {"Key": 4, "ShortName": "Test"}},
        "StartDate": "2025-08-03T15:00:00", "GmtOffset": "02:00:00", "Path": SESSION_PATH, "Name": "Race"}
OLD_TRACK = {"circuit_key": 4, "end_utc": "2025-01-05T15:00:00+00:00", "start_utc": "2025-01-05T13:00:00+00:00",
             "x": [0.0, 100.0, 100.0, 0.0], "y": [0.0, 0.0, 50.0, 50.0], "start": {"x": 0, "y": 0, "heading_deg": 0}, "pit": None}
CLIP = "TeamRadio/BBB22_22_20250803_150000.mp3"


def write_recording(path, *, extra=(), radio=True, circle=False):
    rows = [(t, topic, data) for t, topic, data in synthetic_events()]
    rows.append((0.1, "SessionInfo", INFO))
    if radio:
        rows.append((150.0, "TeamRadio", {"Captures": {"1": {"Utc": utc(120), "RacingNumber": "22", "Path": CLIP}}}))
    for k in range(1600 if circle else 20):  # car 11 round an ellipse, one lap every 90 s
        s = 100.0 + (0.5 if circle else 4) * k
        a = 2 * math.pi * (s - 100.0) / 90.0
        rows.append((s + 2.0, "Position.z", positions([(s, {"11": (5000 * math.cos(a), 3000 * math.sin(a))})])))
    rows += list(extra)
    rows.sort(key=lambda r: r[0])
    with open(path, "w", encoding="utf-8") as f:
        for t, topic, data in rows:
            f.write(json.dumps({"recv": RECV0 + t, "topic": topic, "kind": "delta", "data": data}) + "\n")


def run(path, radio, **kw):
    src = FollowSource(path, follow=False, radio=radio)
    rt = PitWallRuntime(src, models=False, **kw)
    rt.start()
    assert rt.wait(30)
    return rt


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSENSE_DATA", str(tmp_path))
    return tmp_path


def test_outline_is_looked_up_from_session_info(data, tmp_path):
    f = data / "feeds" / "tracks" / "4"
    f.mkdir(parents=True)
    (f / "old.json").write_text(json.dumps(OLD_TRACK), encoding="utf-8")
    (data / "feeds" / "tracks" / "5").mkdir()
    (data / "feeds" / "tracks" / "5" / "other.json").write_text(json.dumps(OLD_TRACK | {"x": [9.0]}), encoding="utf-8")
    rec = tmp_path / "rec.jsonl"
    write_recording(rec, radio=False)
    rt = run(rec, False)
    assert rt.track["x"] == OLD_TRACK["x"] and rt.track["key"].startswith("4:")  # circuit 4, not 5
    assert json.loads(rt.latest_bytes)["extra"]["track"]["x"] == OLD_TRACK["x"]


def test_outline_not_taken_from_a_race_that_ends_after_the_start(data, tmp_path):
    f = data / "feeds" / "tracks" / "4"
    f.mkdir(parents=True)
    (f / "later.json").write_text(json.dumps(OLD_TRACK | {"end_utc": "2025-08-03T17:00:00+00:00"}), encoding="utf-8")
    rec = tmp_path / "rec.jsonl"
    write_recording(rec, radio=False)
    rt = run(rec, False)
    assert rt.track is None  # the stored one ends after the start: not used


def test_outline_built_from_positions_after_two_laps(data, tmp_path):
    rec = tmp_path / "rec.jsonl"
    write_recording(rec, radio=False, circle=True)
    rt = run(rec, False)
    rt._live_track_wall = 0.0
    rt._live_track()
    t = rt.track
    assert t and t["provisional"] and len(t["x"]) == 400
    assert max(t["x"]) == pytest.approx(5000, rel=0.05) and max(t["y"]) == pytest.approx(3000, rel=0.05)


def test_session_info_deltas_merge(data, tmp_path):
    rec = tmp_path / "rec.jsonl"
    write_recording(rec, radio=False, extra=[(0.2, "SessionInfo", {"Meeting": {"Name": "Renamed"}})])
    rt = run(rec, False)
    assert rt.session_info["Meeting"]["Name"] == "Renamed" and rt.session_info["Meeting"]["Circuit"]["Key"] == 4


def test_radio_arrives_asynchronously(data, tmp_path):
    rec = tmp_path / "rec.jsonl"
    write_recording(rec)
    gate, urls = threading.Event(), []

    def fake_download(url, dest):
        urls.append(url)
        gate.wait(10)  # a slow network
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ID3fake-mp3")

    radio = LiveRadio(tmp_path / "rec.session", downloader=fake_download,
                      transcriber=lambda mp3: {"file": mp3.name, "text": "Box box box"})
    rt = run(rec, radio)  # the loop finished while the clip is still downloading: it never blocked
    m = rt._team_radio(True)
    assert len(m) == 1 and m[0]["text"] is None and m[0]["audio"] is None
    assert rt.team_radio_file("22", 0) is None
    gate.set()
    assert radio.join(10)
    m = rt._team_radio(True)
    assert m[0]["text"] == "Box box box" and m[0]["audio"] == "/api/teamradio?car=22&i=0"
    assert urls == ["https://livetiming.formula1.com/static/" + SESSION_PATH + CLIP]
    server = serve(rt, port=0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/teamradio?car=22&i=0", timeout=10) as r:
            assert r.status == 200 and r.read() == b"ID3fake-mp3"
    finally:
        server.shutdown()
    assert rt.health()["radio"]["transcribed"] == 1


def test_offline_degrades_gracefully(data, tmp_path):
    rec = tmp_path / "rec.jsonl"
    write_recording(rec)

    def offline(url, dest):
        raise urllib.error.URLError("no network")

    radio = LiveRadio(tmp_path / "rec.session", downloader=offline, transcriber=lambda p: {"text": "x"}, retries=2, backoff_s=0)
    rt = run(rec, radio)
    assert radio.join(10)
    assert rt.status == "finished" and rt.n_snapshots > 0
    m = rt._team_radio(True)
    assert len(m) == 1 and m[0]["audio"] is None and m[0]["text"] is None
    h = rt.health()["radio"]
    assert h["download_failed"] == 1 and "URLError" in h["error"]
    server = serve(rt, port=0)
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/teamradio?car=22&i=0", timeout=10)
        assert err.value.code == 404
    finally:
        server.shutdown()


def test_transcriber_failure_keeps_audio(data, tmp_path):
    rec = tmp_path / "rec.jsonl"
    write_recording(rec)

    def boom(mp3):
        raise RuntimeError("no gpu")

    def ok(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ID3x")

    radio = LiveRadio(tmp_path / "rec.session", downloader=ok, transcriber=boom)
    rt = run(rec, radio)
    assert radio.join(10)
    m = rt._team_radio(True)
    assert m[0]["audio"] and m[0]["text"] is None and rt.health()["radio"]["transcribe_failed"] == 1


def test_following_a_finished_recording_uses_saved_clips(data, tmp_path):
    rec = tmp_path / "rec.jsonl"
    write_recording(rec)
    d = tmp_path / "rec.session" / "TeamRadio"
    d.mkdir(parents=True)
    (d / "BBB22_22_20250803_150000.mp3").write_bytes(b"ID3saved")
    (d / "BBB22_22_20250803_150000.json").write_text(json.dumps({"text": "Saved text"}), encoding="utf-8")
    rt = run(rec, False)  # no worker at all
    m = rt._team_radio(True)
    assert m[0]["text"] == "Saved text" and m[0]["audio"]
    assert rt.team_radio_file("22", 0).read_bytes() == b"ID3saved"


def test_radio_ignores_unsafe_paths_and_waits_for_session_path(tmp_path):
    got = []
    radio = LiveRadio(tmp_path, downloader=lambda u, d: got.append(u), transcribe=False)
    feed_radio(radio, "TeamRadio", {"Captures": [{"Path": "TeamRadio/../../x.mp3"}, {"Path": "TeamRadio/a.mp3"},
                                                 {"Path": "http://evil/a.mp3"}, {"Path": "TeamRadio/a.json"}]})
    assert got == [] and radio.stats["queued"] == 1  # held until SessionInfo names the session
    feed_radio(radio, "SessionInfo", {"Path": "2025/x/"})
    assert radio.join(5) and got == ["https://livetiming.formula1.com/static/2025/x/TeamRadio/a.mp3"]


def test_tail_keeps_feed_topics_for_the_runtime(tmp_path):
    rec = tmp_path / "rec.jsonl"
    write_recording(rec)
    topics = {e.topic for e in RecordingTail(rec, feeds=True).poll()}
    assert {"TeamRadio", "Position.z", "SessionInfo"} <= topics
