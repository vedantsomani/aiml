"""Ask the pit wall: the question parser, the what-if engine (as-of, deterministic), /api/ask and /api/ask_audio."""

import io
import json
import urllib.error
import urllib.request
import wave

import pytest

from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, PitWall, TeamConfig
from pitsense.pitwall.runtime import PitWallRuntime, ReplaySource
from pitsense.state import RaceState
from pitsense.web.server import serve
from pitsense.whatif import Question, Spec, parse_question, what_if

from .conftest import make_log

TLA = {"16": "LEC", "44": "HAM", "1": "VER", "4": "NOR"}


# ------------------------------------------------------------------ the parser
def _p(text, car="16"):
    return parse_question(text, TLA, car)


@pytest.mark.parametrize("text,labels", [
    ("What if we box now?", ["Box now"]),
    ("box now", ["Box now"]),
    ("should we box?", ["Box now"]),
    ("what if we pit this lap", ["Box now"]),
    ("Box now for hards", ["Box now for HARD"]),
    ("what if we box now for mediums?", ["Box now for MEDIUM"]),
    ("what happens if we pit for softs", ["Box now for SOFT"]),
    ("box in 3 laps", ["Box in 3 laps"]),
    ("what if we box in three laps", ["Box in 3 laps"]),
    ("pit in five laps for the hard tyre", ["Box in 5 laps for HARD"]),
    ("what about boxing next lap", ["Box in 1 lap"]),
    ("what if we pit on lap 30 for softs", ["Box on lap 30 for SOFT"]),
    ("should we box now or in 3 laps", ["Box now", "Box in 3 laps"]),
    ("box now or stay out?", ["Box now", "Stay out to the end"]),
    ("box now for hards or box now for mediums", ["Box now for HARD", "Box now for MEDIUM"]),
    ("what if we stay out to the end", ["Stay out to the end"]),
    ("can we stay out until the end?", ["Stay out to the end"]),
    ("what if we never box again", ["Stay out to the end"]),
    ("what if we go to the end without stopping", ["Stay out to the end"]),
])
def test_parser_whatif_options(text, labels):
    q = _p(text)
    assert q.kind == "whatif" and [o.label() for o in q.options] == labels


@pytest.mark.parametrize("text,k,vsc", [
    ("what if the safety car comes next lap?", 1, False),
    ("what if SC next lap", 1, False),
    ("safety car this lap, should we box", 0, False),
    ("what if there is a safety car in 3 laps", 3, False),
    ("what if a VSC comes out in two laps", 2, True),
    ("if the virtual safety car is deployed now", 0, True),
])
def test_parser_safety_car(text, k, vsc):
    q = _p(text)
    assert q.kind == "sc" and q.sc_in == k and q.vsc == vsc


@pytest.mark.parametrize("text,rival,ref", [
    ("if we box now do we stay ahead of VER?", "1", None),
    ("what does boxing now do against Norris", "4", None),
    ("effect of pitting next lap on Hamilton", "44", None),
    ("can we undercut the car ahead", None, "ahead"),
    ("if we box now, will the car behind get us", None, "behind"),
    ("box now versus car 44", "44", None),
])
def test_parser_rival(text, rival, ref):
    q = _p(text)
    assert q.kind == "whatif" and q.rival == rival and q.rival_ref == ref


def test_parser_never_picks_the_asked_car_as_the_rival():
    q = parse_question("what if LEC boxes now against HAM", TLA, "16")
    assert q.rival == "44"


@pytest.mark.parametrize("text,fact", [
    ("what is the gap to the car ahead?", "gap"),
    ("how far behind is the car behind", "gap"),
    ("gap to Verstappen", "gap"),
    ("who is behind us", "gap"),
    ("how old are the tyres", "tyre_age"),
    ("what is our tyre age?", "tyre_age"),
    ("what is plan B", "plan_b"),
    ("what is the backup plan", "plan_b"),
    ("when should we box", "pit_window"),
    ("what is the pit window", "pit_window"),
    ("which lap is the next stop", "pit_window"),
    ("why are we staying out", "why"),
    ("why box now", "why"),
    ("what is the chance of a safety car", "sc_prob"),
    ("where would we rejoin if we box now", "free"),
    ("what is the pit loss", "free"),
    ("how many laps are left", "free"),
    ("how is the weather", "free"),
])
def test_parser_facts(text, fact):
    q = _p(text)
    assert q.kind == "fact" and q.fact == fact


def test_parser_gap_target_and_neighbours():
    assert _p("gap to VER").rival == "1"
    assert _p("how far is the car in front").rival_ref == "ahead"
    assert _p("what is the interval to the car behind").rival_ref == "behind"


def test_parser_handles_noise():
    for text in ("", "   ", "???", "uh", "blah blah"):
        q = _p(text)
        assert isinstance(q, Question) and q.kind == "fact"
    assert _p("  WHAT   IF we BOX   NOW!!!  ").options == (Spec("now", laps=0),)


# ------------------------------------------------------------------ the engine
def _run(log, until=None, team=("11", "22")):
    wall = PitWall(Context(prior=PitLossPrior(), team=TeamConfig(cars=team)))
    state = RaceState(log.meta)
    for e in log.events:
        if until is not None and e.t > until:
            break
        state.apply(e)
        wall.observe(state)
    return state, wall


def _strip(r):
    return {k: v for k, v in r.items() if k not in ("ms", "sim_ms")}


QUESTIONS = ["what if we box now", "box now for hards or box in 2 laps", "what if we stay out to the end",
             "what if safety car next lap", "if we box now do we stay ahead of the car behind"]


def _cut(race_log, frac=0.6):
    return race_log.events[int(len(race_log) * frac)].t


def test_whatif_answers_are_deterministic_and_as_of(race_log):
    cut = _cut(race_log)
    s_full, w_full = _run(race_log.until(cut))  # the future is not in the log at all
    s_cut, w_cut = _run(race_log, until=cut)  # the future exists but is never applied
    s_again, w_again = _run(race_log, until=cut)
    tla = {n: d.tla for n, d in s_cut.drivers.items()}
    oks = 0
    for text in QUESTIONS:
        a = what_if(w_full, s_full, "11", parse_question(text, tla, "11"))
        b = what_if(w_cut, s_cut, "11", parse_question(text, tla, "11"))
        c = what_if(w_again, s_again, "11", parse_question(text, tla, "11"))
        assert _strip(a) == _strip(b) == _strip(c), text
        oks += a["ok"]
        assert json.dumps(a, default=str)  # serialisable
    assert oks >= 3


def test_whatif_structure_and_probabilities(race_log):
    state, wall = _run(race_log, until=_cut(race_log))
    r = what_if(wall, state, "11", parse_question("what if we box now", {}, "11"))
    assert r["ok"] and r["kind"] == "whatif" and r["car"] == "11"
    s = r["scenario"]
    assert 1 <= s["exp_pos"] <= 3 and s["stops"]
    assert r["p_gain"] is None or 0 <= r["p_gain"] + r["p_loss"] + r["p_same"] <= 1.0001
    assert r["reasons"] and isinstance(r["answer"], str) and "expected position" in r["answer"]
    assert r["ms"] < 2000


def test_whatif_refuses_when_there_is_nothing_to_simulate(race_log):
    state, wall = _run(race_log, until=race_log.events[3].t)  # before any lap
    r = what_if(wall, state, "11", parse_question("what if we box now", {}, "11"))
    assert not r["ok"] and r["why"]
    r = what_if(wall, state, "99", parse_question("box now", {}, "99"))
    assert not r["ok"]


# ------------------------------------------------------------------ /api/ask, /api/ask_audio
def _runtime(until_lap=5, team=("11", "22")):
    rt = PitWallRuntime(ReplaySource(make_log(), 0), team=TeamConfig(cars=team), models=False)
    for e in rt.source.events(rt.stop_event):
        if rt.wall is None:
            rt._build_wall()
        rt.handle(e)
        if rt.state.current_lap >= until_lap:
            break
    return rt


def _post(port, path, body, ctype="application/json", headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method="POST",
                                 headers={"Content-Type": ctype, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def _ask(port, car, text):
    return _post(port, "/api/ask", json.dumps({"car": car, "text": text}).encode())


def test_ask_whatif_and_fact_with_the_voice_off():
    rt = _runtime()
    server = serve(rt, port=0)
    port = server.server_address[1]
    try:
        code, r = _ask(port, "11", "What if we box now?")
        assert code == 200 and r["ok"] and r["source"] == "whatif" and r["whatif"]["scenario"]["stops"]
        assert r["you"]["kind"] == "you" and r["reply"]["kind"] == "voice" and r["reply"]["id"] > r["you"]["id"]
        assert r["audio"] == f"/api/radio.wav?car=11&i={r['reply']['id']}"
        assert rt.wall_text("11", r["reply"]["id"]) == r["answer"]  # the speech endpoint will say exactly this
        code, r2 = _ask(port, "11", "how old are the tyres?")
        assert code == 200 and "lap" in r2["answer"].lower()
        for text in ("gap to the car ahead", "what is the weather like", "what is plan B", "what is the chance of a safety car"):
            code, r3 = _ask(port, "11", text)
            assert code == 200 and r3["answer"]
        kinds = [m["kind"] for m in rt.wall_msgs if m.get("ask")]
        assert kinds == ["you", "voice"] * 6
        assert _post(port, "/api/ask", b"not json")[0] == 400
        assert _ask(port, "11", "")[1]["ok"] is False
        # another site must not be able to drive the wall
        code, bad = _post(port, "/api/ask", json.dumps({"car": "11", "text": "box now"}).encode(), headers={"Origin": "http://evil.example"})
        assert code == 403 and not bad["ok"]
        # an unknown car falls back to our first car
        assert _ask(port, "999", "box now")[1]["car"] == "11"
    finally:
        server.shutdown()
        rt.stop()


def _wav(seconds=0.6, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes([0, 16]) * int(seconds * rate))
    return buf.getvalue()


def test_ask_audio_with_a_fake_transcriber():
    rt = _runtime()
    heard = []
    rt.transcriber = lambda data: (heard.append(len(data)), "what if we box now for hards")[1]
    server = serve(rt, port=0)
    port = server.server_address[1]
    try:
        code, r = _post(port, "/api/ask_audio?car=22", _wav(), "audio/webm;codecs=opus")
        assert code == 200 and r["ok"] and r["heard"] == "what if we box now for hards" and heard
        assert r["car"] == "22" and r["you"]["source"] == "voice" and r["whatif"]["scenario"]["label"] == "Box now for HARD"
        rt.transcriber = lambda data: "  "
        code, r = _post(port, "/api/ask_audio?car=22", _wav(), "audio/webm")
        assert code == 422 and r["error"] == "no speech heard"

        def boom(data):
            raise RuntimeError("no model")

        rt.transcriber = boom
        code, r = _post(port, "/api/ask_audio?car=22", _wav(), "audio/webm")
        assert code == 503 and "could not transcribe" in r["error"]
    finally:
        server.shutdown()
        rt.stop()


def test_audio_decoding_with_pyav():
    av = pytest.importorskip("av")
    import numpy as np

    from pitsense import asr

    a = asr.decode(_wav(0.5, 16000))
    assert a.dtype.name == "float32" and 7000 < len(a) < 9000
    # what a browser sends: opus in webm
    buf = io.BytesIO()
    try:
        out = av.open(buf, "w", format="webm")
        st = out.add_stream("libopus", rate=48000)
        st.layout = "mono"
    except Exception:
        pytest.skip("this PyAV build cannot encode opus")
    sig = (np.sin(np.arange(48000) * 0.05) * 8000).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(sig.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = 48000
    for p in st.encode(frame):
        out.mux(p)
    for p in st.encode(None):
        out.mux(p)
    out.close()
    b = asr.decode(buf.getvalue())
    assert 14000 < len(b) < 18000  # one second at 16 kHz
    with pytest.raises(Exception):
        asr.decode(b"this is not audio")


def test_page_has_the_ask_box_and_microphone():
    from pathlib import Path

    web = Path(__file__).parent.parent / "src" / "pitsense" / "web"
    html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
    for needle in ('id="asktext"', 'id="micbtn"', 'id="askform"'):
        assert needle in html
    for needle in ("/api/ask_audio", "/api/ask", "MediaRecorder", "getUserMedia"):
        assert needle in js


# ------------------------------------------------------------------ /api/whatif and the chart history payload
def test_api_whatif_runs_the_plan_as_of_now_and_is_protected():
    rt = _runtime(until_lap=6)
    server = serve(rt, port=0, token="s3cret")
    port = server.server_address[1]
    lap = rt.state.current_lap + 1
    body = json.dumps({"car": "11", "stop_lap": lap, "compound": "HARD"}).encode()
    try:
        assert _post(port, "/api/whatif", body)[0] in (401, 403)  # no token: refused like /api/ask
        h = {"Authorization": "Bearer s3cret"}
        code, r = _post(port, "/api/whatif?token=s3cret", body, headers=h)
        assert code == 200 and r["ok"], r
        s = r["scenario"]
        assert s["stops"] == [[lap, "HARD"]] and 1 <= s["p10"] <= s["p90"]
        assert "delta_pos" in r and "versus" in r
        assert json.dumps(r)
        assert _post(port, "/api/whatif?token=s3cret", json.dumps({"car": "11", "stop_lap": "x"}).encode(), headers=h)[0] == 422
        assert _post(port, "/api/whatif?token=s3cret", json.dumps({"car": "11", "stop_lap": lap, "compound": "MUSH"}).encode(), headers=h)[0] == 422
        assert _post(port, "/api/whatif?token=s3cret", json.dumps({"car": "999", "stop_lap": lap}).encode(), headers=h)[0] == 422
        assert _post(port, "/api/whatif?token=s3cret", b"nope", headers=h)[0] == 400
    finally:
        server.shutdown()
        rt.stop()


def test_chart_history_payload_is_small_and_as_of():
    rt = _runtime(until_lap=8)
    d = rt.publish(force=True)
    ch = d["extra"]["charts"]
    assert set(ch) == {"stints", "gaps", "order"}
    assert set(ch["gaps"]) <= {"11", "22"} and ch["gaps"]
    g = next(iter(ch["gaps"].values()))
    assert 1 <= len(g["laps"]) <= 15 and len(g["ahead"]) == len(g["laps"]) == len(g["behind"])
    assert all(len(r) == 3 for r in g["ahead"] + g["behind"])
    assert max(g["laps"]) <= rt.state.current_lap
    for rows in ch["stints"].values():
        assert rows and all(len(r) == 3 and r[1] <= r[2] for r in rows)
    assert len(json.dumps(ch)) < 30_000
