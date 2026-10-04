"""Dashboard feeds on the synthetic race: positions, track and radio in the snapshot, the pos event, the mp3 endpoint."""

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pitsense.events import EventLog
from pitsense.pitwall.runtime import PitWallRuntime, ReplaySource
from pitsense.pitwall.types import TeamConfig
from pitsense.web.server import serve

from .conftest import make_log, synthetic_events
from .test_feeds import positions, utc

TRACK = {"x": [i * 100 for i in range(20)] + [1900 - i * 100 for i in range(20)],
         "y": [0] * 20 + [500] * 20, "start": {"x": 0, "y": 0, "heading_deg": 0}, "pit": None, "key": "t"}


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSENSE_DATA", str(tmp_path))
    d = tmp_path / "raw" / "sess" / "TeamRadio"
    d.mkdir(parents=True)
    (d / "a.mp3").write_bytes(b"ID3real-audio")
    (d / "a.json").write_text(json.dumps({"text": "Box this lap"}), encoding="utf-8")
    (d / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.mp3").write_bytes(b"secret")
    return tmp_path


def feed_log():
    rows = list(synthetic_events())
    for k in range(60):  # a position message every 5 s of session time, cars moving along the outline
        rows.append((110.0 + 5 * k, "Position.z", positions([(100.0 + 5 * k, {"11": (100 * k, 0), "22": (100 * k, 10), "33": (50, 20)})])))
    cap = lambda path: {"Captures": [{"Utc": utc(120), "RacingNumber": "22", "Path": path}]}  # noqa: E731
    rows += [(150.0, "TeamRadio", cap("TeamRadio/a.mp3")),
             (160.0, "TeamRadio", cap("TeamRadio/../../secret.mp3")),  # escapes the folder
             (170.0, "TeamRadio", cap("TeamRadio/b.json")),  # not an mp3
             (790.0, "TeamRadio", cap("TeamRadio/late.mp3"))]
    return EventLog(make_log(rows).events, {"slug": "test", "path": "sess/"})


def run(**kw):
    rt = PitWallRuntime(ReplaySource(feed_log(), 0), models=False, team=TeamConfig(cars=("22",)), **kw)
    q = []
    rt._notify = q.append  # every published message, none dropped
    rt.start()
    assert rt.wait(30)
    return rt, q


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.status, r.headers, r.read()


def test_snapshot_extra_has_positions_track_and_radio(session):
    rt, q = run(track=TRACK)
    x = json.loads(rt.latest_bytes)["extra"]
    assert set(x["positions"]["cars"]) == {"11", "22", "33"} and x["positions"]["cars"]["22"][2] == 1
    assert x["track"]["x"] == TRACK["x"]
    radio = x["team_radio"]
    assert [m["id"] for m in radio][:2] == ["22:0", "22:1"] and radio[0]["car"] == "22" and radio[0]["tla"] == "BBB"
    assert radio[0]["text"] == "Box this lap" and radio[0]["audio"] == "/api/teamradio?car=22&i=0"
    assert radio[0]["lap"] >= 1 and radio[-1]["text"] is None  # published 790 s, no transcript file / not yet known


def test_transcript_not_shown_before_it_is_known(session):
    rt = PitWallRuntime(ReplaySource(feed_log(), 0), models=False)
    for e in feed_log().events:
        if e.t > 155.0:  # 5 s after message "a": inside the 15 s transcript latency
            break
        if rt.wall is None:
            rt._build_wall()
        rt.handle(e)
    assert rt._team_radio(True)[0]["text"] is None


def test_without_feeds_or_track_snapshot_still_works():
    rt = PitWallRuntime(ReplaySource(make_log(), 0), models=False)
    rt.start()
    assert rt.wait(30)
    x = json.loads(rt.latest_bytes)["extra"]
    assert x["positions"]["cars"] == {} and x["track"] is None and x["team_radio"] == []


def test_pos_event_is_light_and_throttled(session):
    rt, q = run(track=TRACK)
    msgs = [json.loads(m["data"]) for m in q if m["event"] == "pos"]
    assert msgs and set(msgs[0]) >= {"t", "speed", "cars"} and len(msgs) < 20
    assert len(rt.snap_ms) == rt.n_snapshots  # feed events publish no snapshots of their own


def test_server_serves_teamradio_and_rejects_bad_requests(session):
    rt, _ = run(track=TRACK)
    server = serve(rt, port=0)
    port = server.server_address[1]
    try:
        code, hdr, body = get(port, "/api/teamradio?car=22&i=0")
        assert code == 200 and hdr["Content-Type"] == "audio/mpeg" and body == b"ID3real-audio"
        for bad in ("car=22&i=1",  # path traversal
                    "car=22&i=2",  # not an mp3
                    "car=22&i=3",  # mp3 that does not exist
                    "car=22&i=9", "car=22&i=-1", "car=22&i=x", "car=22", "car=..%2F..&i=0", "car=1&i=0",
                    "car=%2e%2e&i=0", "i=0"):
            with pytest.raises(urllib.error.HTTPError) as err:
                get(port, "/api/teamradio?" + bad)
            assert err.value.code == 404, bad
        assert rt.team_radio_file("22", 1) is None
    finally:
        server.shutdown()


def test_radio_wav_by_id_and_unknown_id(session):
    rt, _ = run(track=TRACK)
    rt.wall_msgs.append({"id": 77, "kind": "voice", "car": "22", "t": 1.0, "lap": 1, "text": "Box now", "action": "BOX"})
    assert rt.wall_text("22", 77) == "Box now" and rt.wall_text("22", 78) is None and rt.wall_text("11") is None


def test_js_has_no_syntax_errors():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    f = Path(__file__).parents[1] / "src" / "pitsense" / "web" / "app.js"
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
