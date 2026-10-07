"""The live pit-wall runtime on the synthetic race: following a recording, the server, shadow scoring, and the slow
work kept off the loop (the voice in its own thread, questions on a copy)."""

import argparse
import copy
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

from pitsense.events import Event, EventLog
from pitsense.live import RecordingTail, load_recording
from pitsense.pitwall import runtime as runtime_mod
from pitsense.pitwall.runtime import (
    SNAP_MS_KEEP, FollowSource, PitWallRuntime, ReplaySource, _VoiceJob, infer_positions,
    load_call_log, shadow_score,
)
from pitsense.pitwall.commands import add_commands
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
    assert a.snap_ms.maxlen == SNAP_MS_KEEP  # timings of the latest snapshots only, not of the whole session


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


class Flipper(Boxer):
    """Stand-in head of strategy whose call changes every 7 s of session time, between laps too."""

    def calls(self, state, view):
        action = "BOX" if int(state.t // 7) % 2 else "STAY_OUT"
        return [Call(t=state.t, car="22", action=action, compound="HARD" if action == "BOX" else None)]


def test_calls_do_not_depend_on_replay_speed():
    def calls(**kw):
        rt = run_replay(engineers=[Flipper], **kw)
        return [(r["t"], r["action"]) for r in rt.calls()["log"] if r["kind"] == "call"]

    live = calls(publish_every_s=3.0)  # what the live pit wall decides at 1x
    assert calls(publish_every_s=60.0) == live  # a fast replay sends fewer snapshots, decides the same
    assert calls(publish_every_s=60.0, coarse=True) != live  # --coarse: deciding only when sending is not live


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
        assert set(json.loads(get(port, "/api/calls")[1])) == {"current", "log", "acks"}
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


def test_token_defaults_to_the_environment(monkeypatch):
    def parse(*argv):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_commands(sub)
        return p.parse_args(["pitwall", *argv]), sub.choices["pitwall"].format_help()

    monkeypatch.setenv("PITSENSE_TOKEN", "from-env")
    a, help_text = parse()
    assert a.token == "from-env" and "PITSENSE_TOKEN" in help_text
    assert parse("--token", "given")[0].token == "given"
    monkeypatch.delenv("PITSENSE_TOKEN")
    assert parse()[0].token is None


# --------------------------------------------------------------- slow work off the loop
class SlowVoice:
    """Stand-in voice model, slow like the fine-tuned LLM on a CPU."""

    def __init__(self, delay):
        self.delay, self.n = delay, 0

    def write(self, facts, task, target=None):
        time.sleep(self.delay)
        self.n += 1
        return SimpleNamespace(text="Box, box.", source="stub")


def test_voice_runs_off_the_loop_and_its_message_follows(tmp_path):
    rt = PitWallRuntime(ReplaySource(make_log(), 0), models=False, engineers=[Boxer], log_dir=tmp_path)
    rt.voice = SlowVoice(1.0)
    q = rt.subscribe()
    rt._ensure_wall()
    for e in rt.source.log.events:
        t0 = time.perf_counter()
        rt.handle(e)
        took = time.perf_counter() - t0
        if rt.calls_log:
            break
    assert took < 0.5, "the snapshot with the new call waited for the voice"
    assert rt.latest["calls"][0]["action"] == "BOX" and "22" not in rt.radio
    call = load_call_log(rt.log_path)[-1]
    assert call["kind"] == "call" and "radio" not in call  # logged when made, the same at any speed

    deadline = time.time() + 10
    while "22" not in rt.radio and time.time() < deadline:
        time.sleep(0.02)
    r = rt.radio["22"]
    assert r["text"] == "Box, box." and r["action"] == "BOX" and rt.voice.n == 1
    assert rt.wall_text("22", r["id"]) == "Box, box."  # what /api/radio.wav speaks
    radio = [x for x in load_call_log(rt.log_path) if x["kind"] == "radio"]
    assert len(radio) == 1 and (radio[0]["car"], radio[0]["t"], radio[0]["action"]) == ("22", call["t"], "BOX")
    while True:  # viewers get it at once, not with the next snapshot
        m = q.get(timeout=5)
        s = json.loads(m["data"]) if m["event"] == "snapshot" else {}
        if "22" in s.get("extra", {}).get("radio", {}):
            break
    assert s["extra"]["radio"]["22"]["text"] == "Box, box." and s["t"] == rt.latest["t"]
    assert any(m["kind"] == "voice" and m["id"] == r["id"] for m in s["extra"]["wall_msgs"])
    assert rt.health()["voice"]["done"] == 1
    rt.stop()


def test_a_jump_back_never_logs_a_call_or_its_radio_again(tmp_path):
    rt = PitWallRuntime(ReplaySource(make_log(), 0), models=False, engineers=[Boxer], log_dir=tmp_path)
    rt.voice = SlowVoice(0.0)
    rt.start()
    assert rt.wait(30)

    def wait_for(cond):
        end = time.time() + 10
        while not cond() and time.time() < end:
            time.sleep(0.02)
        return cond()

    assert wait_for(lambda: any(r["kind"] == "radio" for r in load_call_log(rt.log_path)))
    done = rt.voice_stats["done"]
    assert rt.replay_seek(2)["ok"]  # plays on from lap 2: the BOX call of lap 3 is made and voiced again
    assert rt.wait(30)
    assert wait_for(lambda: rt.voice_stats["done"] == done + 1 and "22" in rt.radio)
    assert [r["kind"] for r in load_call_log(rt.log_path) if r["kind"] != "alert"] == ["call", "radio"]
    rt.stop()


def test_voice_keeps_the_newest_call_per_car_and_never_speaks_while_indexing():
    rt = PitWallRuntime(ReplaySource(make_log(), 0), models=False)
    go, said = threading.Event(), []

    def say(job):
        go.wait(5)
        said.append((job.call.car, job.call.action))
        return f"{job.call.action} {job.call.car}"

    rt._say = say
    job = lambda car, action, i: _VoiceJob(SimpleNamespace(car=car, action=action), None, i, 0.0, 0.0, 1, 0, False)  # noqa: E731
    rt._voice_put(job("11", "BOX", 1))
    deadline = time.time() + 5
    while rt._voice_jobs and time.time() < deadline:  # the voice took it and is busy with it
        time.sleep(0.01)
    rt._voice_put(job("22", "BOX", 2))
    rt._voice_put(job("22", "STAY_OUT", 3))  # replaces 22's call still waiting
    go.set()
    while "22" not in rt.radio and time.time() < deadline:
        time.sleep(0.01)
    assert said == [("11", "BOX"), ("22", "STAY_OUT")] and rt.voice_stats["superseded"] == 1
    assert rt.radio["22"] == {"text": "STAY_OUT 22", "lap": 1, "action": "STAY_OUT", "id": 3}
    rt.stop()

    ix = PitWallRuntime(ReplaySource(make_log(), 0), models=False, engineers=[Boxer], indexer=True)
    ix.voice = SlowVoice(0.0)
    ix.start()
    assert ix.wait(30) and ix.calls_log
    assert ix.voice.n == 0 and ix._voice_thread is None


def test_questions_run_on_a_copy_outside_the_loop_lock(monkeypatch):
    from pitsense import whatif as wi

    rt = run_replay()
    inside, release, seen = threading.Event(), threading.Event(), {}

    def slow_what_if(wall, state, car, q, **kw):
        seen.update(wall=wall, state=state)
        inside.set()
        release.wait(5)
        return {"ok": False, "why": "stub"}

    monkeypatch.setattr(wi, "what_if", slow_what_if)
    out = {}
    th = threading.Thread(target=lambda: out.update(r=rt.whatif("11", 7, "HARD")))
    th.start()
    try:
        assert inside.wait(5)
        assert rt.sim_lock.acquire(timeout=1), "the loop waits for the simulation"
        rt.sim_lock.release()
        assert seen["state"] is not rt.state and seen["wall"] is not rt.wall
        assert seen["state"].fingerprint() == rt.state.fingerprint()
        monkeypatch.setattr(runtime_mod, "ASK_WAIT_S", 0.1)
        busy = rt.ask("11", "what if we box now")  # one simulation at a time, whoever asks
        assert busy["ok"] is False and busy["busy"]
    finally:
        release.set()
        th.join(5)
    assert out["r"] == {"ok": False, "error": "stub", "car": "11"}


def test_questions_never_change_the_calls():
    """A question runs on a copy: the live engineers (memory, call history, analysis cache) never see it."""
    def play(ask):
        rt = PitWallRuntime(ReplaySource(make_log(), 0), team=TeamConfig(cars=("11", "22")), models=False)
        rt._ensure_wall()
        for e in rt.source.log.events:
            rt.handle(e)
            if ask and rt.state.current_lap >= 3 and rt.n_events % 25 == 0:
                assert rt.ask("11", "what if we box now")["ok"] and rt.ask("22", "how old are the tyres?")["ok"]
                rt.whatif("11", rt.state.current_lap + 2, "HARD")
        return rt

    a, b = play(False), play(True)
    assert b._frozen is not None
    keep = lambda rt: [{k: v for k, v in r.items() if k != "wall"} for r in [*rt.calls_log, *rt.alerts_log]]  # noqa: E731
    assert keep(a) == keep(b)
    assert {k: v for k, v in a.latest.items() if k != "extra"} == {k: v for k, v in b.latest.items() if k != "extra"}


# ------------------------------------------------------------------ the operator
def test_a_changed_call_says_what_it_was_and_why():
    rt = run_replay(engineers=[Flipper], publish_every_s=3.0)
    calls = [r for r in rt.calls()["log"] if r["kind"] == "call"]
    assert "change" not in calls[0]  # the first call has nothing to change from
    ch = calls[1]["change"]
    assert ch["from"] == calls[0]["action"].replace("_", " ") + (f" {calls[0]['compound']}" if calls[0].get("compound") else "")
    assert ch["to"].startswith(calls[1]["action"].replace("_", " ")) and ch["why"]
    assert rt.latest["extra"]["changes"]["22"] == calls[-1]["change"]


def test_operator_accepts_and_rejects_calls_and_they_are_logged(tmp_path):
    rt = run_replay(log_dir=tmp_path, engineers=[Boxer])
    assert rt.ack("22", "maybe")["ok"] is False and rt.ack("99", "accept")["ok"] is False
    r = rt.ack("22", "reject", "  tyres   still good ")
    assert r["ok"] and r["ack"]["action"] == "BOX" and r["ack"]["reason"] == "tyres still good"
    assert rt.acks["22"]["decision"] == "reject" and rt.calls()["acks"][-1] == r["ack"]
    assert [x["decision"] for x in load_call_log(rt.log_path) if x["kind"] == "ack"] == ["reject"]
    assert rt.ack("22", "accept", call_t=-5)["ok"] is False  # no call logged at that time


def test_ack_over_http():
    rt = run_replay(engineers=[Boxer])
    server = serve(rt, port=0)
    try:
        port = server.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/ack", data=b'{"car":"22","decision":"accept"}',
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert json.loads(resp.read())["ack"]["decision"] == "accept"
    finally:
        server.shutdown()


def test_shadow_score_reports_the_operator():
    final = replay(make_log())  # car 22 stops at the end of lap 3
    calls = [{"kind": "ack", "car": "22", "action": "BOX", "call_lap": 2, "decision": "accept"},  # right call, accepted
             {"kind": "ack", "car": "22", "action": "BOX", "call_lap": 3, "decision": "reject"},  # right call, overruled
             {"kind": "ack", "car": "11", "action": "BOX", "call_lap": 3, "decision": "reject"}]  # wrong call, overruled
    op = shadow_score(calls, final, k=2)["operator"]
    assert op["accept"] == {"answers": 1, "call_right": 1, "call_right_rate": 1.0}
    assert op["reject"] == {"answers": 2, "call_right": 1, "call_right_rate": 0.5}


def test_risk_can_be_changed_while_running_and_replans():
    rt = run_replay()
    assert rt.set_risk("reckless")["ok"] is False
    assert rt.set_risk("protect") == {"ok": True, "risk": "protect"}
    assert rt.team.risk == "protect" and rt.wall.ctx.team.risk == "protect"  # the strategy cache key changes with it
    assert rt.publish()["extra"]["risk"] == "protect"


def test_live_recorder_is_restarted_when_it_dies(tmp_path):
    from pitsense.pitwall.sources import LiveSource

    class Flaky(LiveSource):
        BACKOFF_S = (0.05, 0.1)
        runs = 0

        def _record(self, minutes):
            Flaky.runs += 1
            if Flaky.runs == 1:
                raise ConnectionError("feed gone")
            time.sleep(5)  # the second recorder keeps running

    src = Flaky(tmp_path / "live.jsonl", minutes=1, radio=False)
    src._deadline = time.time() + 60  # skip the fastf1 check
    stop = threading.Event()
    threading.Thread(target=lambda: [None for _ in src.events(stop)], daemon=True).start()
    end = time.time() + 5
    while src.recorder_restarts < 1 and time.time() < end:
        time.sleep(0.05)
    stop.set()
    assert src.recorder_restarts == 1 and Flaky.runs == 2 and "feed gone" in src.recorder_error
