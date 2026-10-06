"""Replay lab: pause/resume, jumps that equal a straight play, bookmarks, the controls API, the alarm banner."""

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pitsense.pitwall.runtime import PitWallRuntime, ReplaySource
from pitsense.web.server import serve

from .conftest import make_log

WALL_KEYS = {"wall_time", "wall"}
WEB = Path(__file__).parents[1] / "src" / "pitsense" / "web"


def scrub(o):
    """Drop wall-clock fields (wall_time, wall, *_ms timings) so two runs of the same events can be compared."""
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if k not in WALL_KEYS and not k.endswith("_ms")}
    if isinstance(o, (list, tuple)):
        return [scrub(v) for v in o]
    return o


def runtime(**kw):
    kw.setdefault("models", False)
    return PitWallRuntime(ReplaySource(make_log(), 0), **kw)


def straight_to(lap):
    """A fresh runtime fed the log event by event until the leader is on ``lap`` (no controls involved)."""
    rt = runtime()
    rt._ensure_wall()
    for e in rt.source.log.events:
        rt.handle(e)
        if rt.state.current_lap >= lap:
            break
    return rt


def capture_snaps(rt):
    got = []
    rt._notify = lambda m: got.append(json.loads(m["data"])) if m["event"] == "snapshot" else None
    return got


def view(rt):
    return scrub({"snap": rt.latest, "calls": list(rt.calls_log), "alerts": list(rt.alerts_log),
                  "wall_msgs": list(rt.wall_msgs), "n": rt.n_events})


def play_all(rt):
    rt.start()
    assert rt.wait(60)
    rt.replay_pause()  # a jump after the end plays on from there: hold it still so the state can be compared
    return rt


@pytest.mark.parametrize("lap", [1, 2, 3, 4, 5, 6, 9])
def test_seek_back_equals_straight_play(lap):
    ref = straight_to(lap)
    rt = play_all(runtime())
    assert rt.state.current_lap == 9
    res = rt.replay_seek(lap)
    assert res["ok"] and res["via"] == "checkpoint" and res["events_replayed"] == 0
    assert view(rt) == view(ref)
    assert (json.dumps(scrub(json.loads(rt.latest_bytes)), sort_keys=True)
            == json.dumps(scrub(json.loads(ref.latest_bytes)), sort_keys=True))


def test_seek_without_checkpoints_replays_from_start_and_matches():
    ref = straight_to(5)
    rt = play_all(runtime(checkpoint_every=100))  # only the start is saved
    res = rt.replay_seek(5)
    assert res["via"] == "replay" and res["events_replayed"] > 0
    assert view(rt) == view(ref)


def test_seek_forward_from_live_position_matches():
    ref = straight_to(6)
    rt = runtime(checkpoint_every=100)
    rt._ensure_wall()
    for e in rt.source.log.events:
        rt.handle(e)
        if rt.state.current_lap >= 2:
            break
    rt.source.jump(rt.n_events)
    res = rt._seek_exec(6)
    assert res["via"] == "replay" and res["lap"] >= 6
    assert view(rt) == view(ref)


def test_engineer_memory_is_rebuilt_not_kept():
    """After a jump back and playing on, every later snapshot equals the straight run's."""
    ref = runtime()
    ref_snaps = capture_snaps(ref)
    play_all(ref)
    rt = play_all(runtime())
    rt.replay_seek(2)
    after = capture_snaps(rt)
    rt.replay_resume()
    assert rt.wait(60)
    assert after, "playing went on from the jump"
    assert [scrub(s) for s in after] == [scrub(s) for s in ref_snaps[-len(after):]]
    assert scrub(list(rt.calls_log)) == scrub(list(ref.calls_log))


def test_seek_beyond_the_end_is_clamped_and_future_is_gone():
    rt = play_all(runtime())
    res = rt.replay_seek(99)
    assert res["ok"] and res["clamped"] and res["lap"] == 9
    rt.replay_seek(2)
    assert rt.state.current_lap == 2 and max(lap.lap for lap in rt.state.laps) <= 2


def test_pause_resume():
    rt = PitWallRuntime(ReplaySource(make_log(), 200), models=False)  # slow enough to catch mid-race
    rt.start()
    time.sleep(0.4)
    assert rt.replay_pause()["paused"]
    time.sleep(0.1)
    n = rt.n_events
    assert 0 < n < len(rt.source.log.events)
    time.sleep(0.4)
    assert rt.n_events == n and rt.thread.is_alive()
    assert "stale" not in " ".join(rt.health()["alarms"])
    assert not rt.replay_resume()["paused"]
    assert rt.replay_speed("max")["speed"] == 0
    assert rt.wait(60) and rt.status == "finished" and rt.n_events > n


def test_speed_validation():
    rt = runtime()
    assert rt.replay_speed(5)["speed"] == 5
    assert not rt.replay_speed("fast")["ok"] and not rt.replay_speed(-1)["ok"]


def test_marks_cover_stops_safety_car_and_calls():
    rt = play_all(runtime())
    for _ in range(200):
        if any(m["kind"] == "SC" for m in rt.replay_index.mark_list()):
            break
        time.sleep(0.05)
    out = rt.replay_marks()
    kinds = {m["kind"] for m in out["marks"]}
    assert {"pit", "SC"} <= kinds
    pit = next(m for m in out["marks"] if m["kind"] == "pit")
    assert pit["car"] == "22" and pit["lap"] == 3
    sc = next(m for m in out["marks"] if m["kind"] == "SC")
    assert 5 <= sc["lap"] <= 7
    r = rt.replay_seek(mark=sc["id"])
    assert r["ok"] and r["lap"] == sc["lap"]
    assert not rt.replay_seek(mark="nope")["ok"]
    after = rt.replay_marks()["marks"]  # marks later than now are flagged
    assert any(m["ahead"] for m in after) and not any(m["ahead"] for m in after if m["lap"] < sc["lap"])


class Live(ReplaySource):
    mode, seekable = "live", False


def test_live_mode_has_no_controls():
    rt = PitWallRuntime(Live(make_log(), 0), models=False)
    for out in (rt.replay_pause(), rt.replay_resume(), rt.replay_speed(5), rt.replay_seek(3)):
        assert out["ok"] is False and "disabled" in out["error"]
    got = rt.replay_marks()
    assert got["enabled"] is False and got["marks"] == []


def call(port, path, body=None, token=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET" if body is None else "POST",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"X-PitSense-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_replay_api_and_token():
    rt = play_all(runtime())
    srv = serve(rt, port=0, token="tok")
    port = srv.server_address[1]
    try:
        assert call(port, "/api/replay/seek", {"lap": 3})[0] == 401
        assert call(port, "/api/replay/marks")[0] == 401
        code, out = call(port, "/api/replay/seek", {"lap": 3}, token="tok")
        assert code == 200 and out["ok"] and out["lap"] == 3
        assert call(port, "/api/replay/seek", {"lap": "x"}, token="tok")[0] == 422
        assert call(port, "/api/replay/pause", {}, token="tok")[1]["paused"] is True
        assert call(port, "/api/replay/resume", {}, token="tok")[1]["paused"] is False
        assert call(port, "/api/replay/speed", {"speed": 20}, token="tok")[1]["speed"] == 20
        assert call(port, "/api/replay/speed", {"speed": "max"}, token="tok")[1]["speed"] == 0
        code, m = call(port, "/api/replay/marks", token="tok")
        assert code == 200 and m["enabled"] and "marks" in m and m["lap"] == rt.state.current_lap
        assert call(port, "/api/health", token="tok")[1]["replay"]["enabled"]
    finally:
        srv.shutdown()
        rt.stop()


def test_live_api_disables_seek():
    rt = play_all(PitWallRuntime(Live(make_log(), 0), models=False))
    srv = serve(rt, port=0)
    port = srv.server_address[1]
    try:
        code, out = call(port, "/api/replay/seek", {"lap": 3}, {})
        assert code == 409 and not out["ok"]
        assert call(port, "/api/replay/marks")[1]["enabled"] is False
    finally:
        srv.shutdown()


def test_indexing_pass_fills_checkpoints():
    rt = play_all(runtime(index=True))
    for _ in range(300):
        if rt.replay_index.done:
            break
        time.sleep(0.05)
    assert rt.replay_index.done and rt.replay_index.laps()[-1] == 9


# --- the page
def js_source():
    return (WEB / "app.js").read_text(encoding="utf-8")


def test_banner_markup_and_styles():
    html, css, js = (WEB / "index.html").read_text(encoding="utf-8"), (WEB / "style.css").read_text(encoding="utf-8"), js_source()
    assert 'id="alarmbar"' in html and 'id="replaybar"' in html and 'id="timeline"' in html
    assert ".alarmbar.red" in css and ".alarmbar.green" in css
    assert '"/api/replay/" + path' in js and "/api/replay/marks" in js and "/api/health" in js and '"seek"' in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_banner_logic_and_syntax():
    assert subprocess.run(["node", "--check", str(WEB / "app.js")], capture_output=True).returncode == 0
    js = js_source()
    a, b = js.index("// alarms:begin"), js.index("// alarms:end")
    prog = js[a:b] + """
const red = alarmView(["RED: Feed stale for 40s (>30s)", "RED: 2 engineer errors"]);
const green = alarmView([]);
const odd = alarmView(["<b>x</b>"]);
console.log(JSON.stringify({red, green, odd}));
"""
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["red"]["cls"] == "red" and got["red"]["html"].count("<li>") == 2 and "Feed stale" in got["red"]["html"]
    assert got["green"]["cls"] == "green" and "No alarms" in got["green"]["html"]
    assert "&lt;b&gt;x" in got["odd"]["html"] and "<li><b>" not in got["odd"]["html"]  # escaped


def test_seek_to_lap_zero_is_the_start_of_the_race():
    rt = play_all(runtime())
    res = rt.replay_seek(0)
    assert res["ok"] and res["events"] == 0 and rt.state.current_lap == 0 and rt.latest is None
