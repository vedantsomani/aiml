"""The live pit-wall runtime on the synthetic race: following a recording, the server, shadow scoring."""

import copy
import json
import time
import urllib.error
import urllib.request

from pitsense.events import Event, EventLog
from pitsense.live import RecordingTail, load_recording
from pitsense.pitwall.runtime import (
    FollowSource, PitWallRuntime, ReplaySource, infer_positions, load_call_log, shadow_score,
)
from pitsense.pitwall.types import Call, Reason, TeamConfig
from pitsense.state import RaceState, replay
from pitsense.web.server import serve

from .conftest import make_log

T0 = 1_700_000_000.0


def rec_lines():
    return [
        json.dumps({"recv": T0 + e.t, "topic": e.topic, "kind": "delta", "data": e.data}) + "\n"
        for e in make_log().events  # a recorder writes in arrival order
    ]


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.status, r.read()


def run_replay(log=None, **kw):
    rt = PitWallRuntime(ReplaySource(log or make_log(), 0), models=False, **kw)
    rt.start()
    assert rt.wait(30)
    return rt


def test_tail_keeps_partial_last_line(tmp_path):
    path = tmp_path / "r.jsonl"
    lines = rec_lines()
    tail = RecordingTail(path)
    assert tail.poll() == []  # file does not exist yet
    blob = "".join(lines[:5]).encode()
    cut = len(blob) - 20  # the 5th line is only half written
    path.write_bytes(blob[:cut])
    assert len(tail.poll()) == 4
    with open(path, "ab") as f:
        f.write(blob[cut:] + b"not json\n")
    assert len(tail.poll()) == 1 and tail.bad_lines == 1
    with open(path, "ab") as f:
        f.write("".join(lines[5:]).encode())
    assert len(tail.poll()) == len(lines) - 5


def test_follow_progressively_written_recording(tmp_path):
    path = tmp_path / "rec.jsonl"
    lines = rec_lines()
    rt = PitWallRuntime(FollowSource(path, follow=True, poll_s=0.01), models=False, log_dir=tmp_path / "logs")
    rt.start()
    blob = "".join(lines).encode()
    with open(path, "wb") as f:  # uneven chunks that split lines
        for i in range(0, len(blob), 997):
            f.write(blob[i:i + 997])
            f.flush()
            time.sleep(0.005)
    deadline = time.time() + 20
    while rt.n_events < len(lines) and time.time() < deadline:
        time.sleep(0.05)
    assert rt.n_events == len(lines)
    snap = rt.latest
    rt.stop()
    assert snap["lap"] == 9
    # the same file replayed in one go ends in the same state
    assert replay(load_recording(path)).fingerprint() == rt.state.fingerprint()


def test_replay_publishes_each_lap_and_is_deterministic(tmp_path):
    a, b = run_replay(log_dir=tmp_path / "a"), run_replay(log_dir=tmp_path / "b")
    assert a.status == "finished" and a.n_snapshots >= 8
    assert a.state.fingerprint() == b.state.fingerprint() and a.n_snapshots == b.n_snapshots
    assert [s["tla"] for s in a.latest["tower"]] == ["AAA", "CCC", "BBB"]
    h = a.health()
    assert h["snapshot_ms_p50"] is not None and h["observe_errors"] == 0 and h["snapshot_errors"] == 0


def test_inferred_order_when_feed_has_no_positions():
    events = make_log().events
    stripped = []
    for e in events:
        data = e.data
        if e.topic == "TimingData":
            data = copy.deepcopy(data)
            for upd in data["Lines"].values():
                upd.pop("Position", None)
        stripped.append(Event(e.t, e.topic, data, e.seq, e.kind))
    rt = run_replay(EventLog(stripped, {}))
    assert rt.inferred_order and rt.latest["extra"]["inferred_order"] is True
    tower = rt.latest["tower"]
    assert tower[0]["car"] == "11" and all(r["position"] for r in tower)
    assert not run_replay().inferred_order  # positions in the feed: nothing inferred


def test_infer_positions_orders_by_laps_then_gap():
    s = RaceState()
    for n, laps, gap in [("1", 5, 0.0), ("2", 5, 12.0), ("3", 4, 3.0), ("4", 5, 4.0)]:
        d = s.driver(n)
        d.laps, d.gap_to_leader = laps, gap
    infer_positions(s)
    assert [d.number for d in s.running_order()] == ["1", "4", "2", "3"]


class Boxer:
    """Stand-in head of strategy: box car 22 from lap 3 on."""

    name = "fake"
    requires = ()
    features = ()
    in_bench = False

    def __init__(self, ctx, memory):
        pass

    def observe(self, state):
        pass

    def car(self, state, n, view):
        return {}

    def race(self, state, view):
        return {}

    def alerts(self, state, view):
        return []

    def calls(self, state, view):
        if state.current_lap < 3:
            return []
        return [Call(t=state.t, car="22", action="BOX", compound="HARD", confidence=0.8,
                     reasons=(Reason("x", "because"),))]


def test_calls_are_logged_once_with_both_clocks(tmp_path):
    rt = run_replay(log_dir=tmp_path, engineers=[Boxer])
    boxes = [r for r in load_call_log(rt.log_path) if r["kind"] == "call"]
    assert len(boxes) == 1 and boxes[0]["action"] == "BOX"
    assert boxes[0]["wall"] > 1e9 and "t" in boxes[0] and boxes[0]["car_lap"] is not None
    assert boxes[0]["reasons"][0]["text"] == "because"


def test_server_snapshot_health_and_stream():
    rt = run_replay(team=TeamConfig(cars=("22",)))
    server = serve(rt, port=0)
    port = server.server_address[1]
    try:
        code, body = get(port, "/api/snapshot")
        snap = json.loads(body)
        assert code == 200 and snap["lap"] == 9 and snap["focus"] == ["22"] and len(snap["tower"]) == 3
        assert snap["extra"]["mode"] == "replay"
        health = json.loads(get(port, "/api/health")[1])
        assert health["status"] == "finished" and health["events"] == rt.n_events
        assert set(json.loads(get(port, "/api/calls")[1])) == {"current", "log"}
        assert set(json.loads(get(port, "/api/alerts")[1])) == {"active", "log"}
        assert "Timing tower" in get(port, "/")[1].decode()
        assert b"EventSource" in get(port, "/app.js")[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stream", timeout=10) as r:
            assert r.headers["Content-Type"].startswith("text/event-stream")
            head = b"".join(r.readline() for _ in range(3))
            assert b"event: snapshot" in head and b'"tower"' in head
        try:
            get(port, "/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as err:
            assert err.code == 404
    finally:
        server.shutdown()
        rt.stop()


def test_shadow_score_against_real_stops():
    final = replay(make_log())  # car 22 stops at the end of lap 3
    assert [(p.driver, p.in_lap) for p in final.pit_events] == [("22", 3)]
    calls = [
        {"kind": "call", "car": "22", "action": "BOX", "car_lap": 2},  # right: the stop is 1 lap later
        {"kind": "call", "car": "22", "action": "BOX", "car_lap": 7},  # wrong: 4 laps off
        {"kind": "call", "car": "33", "action": "PREPARE_BOX", "car_lap": 3},  # wrong: 33 never stops
        {"kind": "call", "car": "11", "action": "STAY_OUT", "car_lap": 3},  # right
        {"kind": "call", "car": "22", "action": "STAY_OUT", "car_lap": 3},  # wrong: it stops now
        {"kind": "call", "car": "11", "action": "NO_CALL", "car_lap": 3},  # not scored
        {"kind": "alert", "car": "22"},
    ]
    r = shadow_score(calls, final, k=2)
    assert r["box_calls"] == 3 and r["box_precision"] == round(1 / 3, 4)
    assert r["stay_out_calls"] == 2 and r["stay_out_accuracy"] == 0.5
    assert r["real_stops"] == 1 and r["stop_recall"] == 1.0
    assert shadow_score([], final)["stop_recall"] == 0.0


def test_cli_command_registered(capsys):
    from pitsense import cli

    try:
        cli.main(["pitwall", "-h"])
    except SystemExit as e:
        assert e.code == 0
    assert "--speed" in capsys.readouterr().out
