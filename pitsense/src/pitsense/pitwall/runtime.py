"""Run the pit wall on a race and publish what it sees.

One loop, three kinds of source::

    archive replay at N x speed     Source -> events (paced by the session clock)
    a recording followed as it grows
    live: start the recorder, follow its file

    loop:  state.apply(event) -> wall.observe(state) -> maybe decide -> maybe publish

Deciding (the engineers' snapshot; new calls and alerts go to the log) happens every ``PUBLISH_EVERY_S``
of session time and on every leader lap: the live pit wall's cadence, whatever the replay speed, so a
replay at any speed logs exactly the calls the live pit wall would have made. Viewers get the latest
decision on every leader lap and every ``publish_every_s`` (about 3 s of wall time at the starting speed).
Both use the session clock, never the wall clock, so the same input and starting speed also give the
same published snapshots. Each published snapshot is serialised once and handed to the web server
(``pitsense.web``); calls and alerts are appended to a JSONL log with the session time and the
wall-clock time.

Nothing slow runs in the loop: the voice writes radio messages in its own thread (the call is logged
and published at once; its text follows as a ``radio`` record and a re-sent snapshot, so only when the
text arrives depends on the voice), and questions (``ask``, ``whatif``) run one at a time on a private
copy of the state and engineers taken under the lock, never on the live ones.

Around it: ``sources.py`` (replay, follow, live), ``shadow.py`` (scoring a call log against the real stops) and
``commands.py`` (``pitsense pitwall``, ``doctor``, ``shadow-score``).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import events as _events  # the feeder: reads the log; exempt by name in tests/test_pitwall.py
from ..config import FEED_TOPICS
from ..state import RaceState
from . import charts
from . import replay as _replay
from .types import Snapshot, TeamConfig
from .shadow import load_call_log, shadow_score  # noqa: F401  (moved; imported from here by older code)
from .sources import (  # noqa: F401
    LEAD_IN_S, FollowSource, LiveSource, ReplaySource, Source, feed_has_positions, infer_positions,
)

log = logging.getLogger("pitsense.pitwall")

PUBLISH_EVERY_S = 3.0  # session seconds between decisions (the live cadence), and between snapshots at 1x
MAX_SPEED_TICK_S = 30.0  # snapshot spacing for viewers when replaying as fast as possible
LOG_KEEP = 500  # calls and alerts kept in memory for the API
POS_EVERY_S = 0.5  # session seconds between car positions
POS_MIN_WALL_S = 0.3  # ...and never faster than ~3 Hz on the wall clock (high replay speeds)
RADIO_EVERY_WALL_S = 1.0  # how often the team-radio list is rebuilt
RADIO_KEEP = 80  # driver radio messages in the snapshot
WALL_KEEP = 120  # pit-wall conversation entries (voice calls and alerts) in the snapshot
SNAPSHOT_SCHEMA = 2  # snapshot layout version: 2 adds "schema", "predictions" and extra.telemetry
SNAP_MS_KEEP = 1000  # snapshot build times kept for health()
VOICE_QUEUE = 4  # calls waiting for the voice (one per car; when full the oldest is dropped)
ASK_WAIT_S = 10.0  # a question waits this long for the one running before it, then gets "busy"


# ----------------------------------------------------------------------------- runtime
def _merge(into: dict, update: dict) -> None:
    """Deep-merge a SessionInfo delta into what we know (deltas carry only the changed fields)."""
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            _merge(into[k], v)
        else:
            into[k] = json.loads(json.dumps(v))


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=str).encode()


def _call_key(c: dict) -> tuple:
    return (c["car"], c["action"], c.get("compound"))


def _call_label(action: str | None, compound: str | None) -> str:
    return (action or "NO_CALL").replace("_", " ") + (f" {compound}" if compound else "")


def call_change(before: dict, c: dict, track: str | None) -> dict:
    """What changed since a car's previous call: from / to, and why (the reasons that are new, most important first)."""
    from ..config import TRACK_STATUS

    why = [r.get("text") for r in c.get("reasons") or () if r.get("code") not in before["codes"] and r.get("text")][:3]
    if track != before.get("track"):
        why.insert(0, f"track status now {TRACK_STATUS.get(str(track), track)}")
    if not why:
        why = ["same reasons, weighed differently as the race moved on"]
    return {"from": _call_label(before["action"], before.get("compound")), "to": _call_label(c["action"], c.get("compound")),
            "why": why}


@dataclass
class _VoiceJob:
    """A call waiting for its radio message (written by the voice thread, never by the loop)."""

    call: object  # the Call
    snap: Snapshot  # the snapshot it was made in (what the voice reads)
    id: int  # its id in the pit-wall conversation, reserved when the call was made
    t_call: float  # the call record's ``t``
    t: float  # session time and leader lap of the snapshot
    lap: int
    gen: int  # jump generation: after a jump the message no longer belongs to the race shown
    logged: bool  # the call record went to the log file, so its radio record goes too


@dataclass
class _Frozen:
    """A private copy of the state and the engineers that questions run on, outside the loop's lock."""

    key: tuple  # (jump generation, events, snapshots): reused while nothing has happened
    state: RaceState
    wall: object
    snap: Snapshot | None = None  # the snapshot at this moment, for fact questions (made once)


class PitWallRuntime:
    """Follows one source, keeps the latest snapshot, logs calls and alerts, fans out to listeners."""

    def __init__(self, source: Source, *, team: TeamConfig | None = None, log_dir: Path | None = None,
                 models: bool | object = True, history=None, publish_every_s: float | None = None,
                 engineers=None, track: dict | None = None, checkpoint_every: int = 1,
                 index: bool = False, replay_index: "_replay.ReplayIndex | None" = None,
                 indexer: bool = False, coarse: bool = False) -> None:
        self.source = source
        self.team = team or TeamConfig()
        self.log_dir = Path(log_dir) if log_dir else None
        self.use_models = models
        self.history = history
        self.engineers = engineers
        speed = source.speed
        if publish_every_s is not None:
            self.publish_every_s = publish_every_s
        else:
            self.publish_every_s = MAX_SPEED_TICK_S if speed <= 0 else PUBLISH_EVERY_S * max(speed, 1.0)
        self.coarse = coarse  # decide only when publishing: faster at high speed, NOT the live pit wall's calls
        self.decide_every_s = self.publish_every_s if coarse else PUBLISH_EVERY_S
        self.state = RaceState(source.meta)
        self.wall = None
        self._wall_quali = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.sim_lock = threading.RLock()  # the loop applies an event under it; questions copy the state under it
        self._ask_gate = threading.Lock()  # one question (simulation) at a time; held outside sim_lock
        self._frozen: _Frozen | None = None
        self._gen = 0  # bumped by every jump
        self._log_lock = threading.Lock()  # the call-log file is written by the loop and the voice thread
        self._voice_lock = threading.Lock()
        self._voice_cv = threading.Condition()
        self._voice_jobs: dict[str, _VoiceJob] = {}  # car -> its newest call waiting for the voice
        self._voice_thread: threading.Thread | None = None
        self._voice_stop = False
        self.voice_stats = {"done": 0, "superseded": 0, "dropped": 0, "last_ms": None}
        self.transcriber = None  # bytes -> text for /api/ask_audio (default: pitsense.asr.transcribe)
        self._ask_tpl = None
        self.listeners: list[queue.Queue] = []
        self.latest: dict | None = None
        self.latest_bytes: bytes = _json_bytes({"waiting": True})
        self.calls_log: deque = deque(maxlen=LOG_KEEP)
        self.alerts_log: deque = deque(maxlen=LOG_KEEP)
        self.model_info: dict = {"loaded": False}
        self.inferred_order = False
        self._feed_positions = False
        self.status = "starting"  # starting | running | finished | stopped | error
        self.error: str | None = None
        self.started_wall = time.time()
        self.last_event_wall: float | None = None
        self.n_events = 0
        self.n_snapshots = 0
        self.observe_errors = 0
        self.snapshot_errors = 0
        self.snap_ms: deque = deque(maxlen=SNAP_MS_KEEP)
        self._decided_t: float | None = None  # session time and leader lap of the last decision
        self._decided_lap = -1
        self._sent_t: float | None = None  # session time of the last snapshot published
        self._last_calls: dict[str, tuple] = {}
        self._last_call_info: dict[str, dict] = {}  # car -> its previous logged call (to say what changed and why)
        self.changes: dict[str, dict] = {}  # car -> what changed at its latest call change
        self.acks_log: deque = deque(maxlen=LOG_KEEP)  # the operator's accept / reject answers
        self.acks: dict[str, dict] = {}  # car -> the latest answer
        self.radio: dict[str, dict] = {}  # car -> latest radio message from the voice
        self.wall_msgs: deque = deque(maxlen=WALL_KEEP)  # pit-wall conversation: voice calls and alerts
        self._wall_id = 0
        self.track: dict | None = track
        self._track_done = track is not None
        self._pos_t: float | None = None
        self._pos_wall = 0.0
        self.positions: dict = {}
        self.session_info: dict = {}  # SessionInfo merged over its messages: circuit, start, static path
        self.radio_worker = getattr(source, "radio", None)
        self._live_track_wall = 0.0
        self._live_track_tries = 0
        self._radio_lap: dict[str, int] = {}  # mp3 path -> leader lap when it was published
        self._radio_list: list[dict] = []
        self._radio_wall = 0.0
        self._radio_sig: tuple = ()
        self.voice = None if os.environ.get("PITSENSE_VOICE", "") != "off" else False  # loaded on first call
        self._seen_alerts: set[tuple] = set()
        self._last_alarms: list[str] = []  # track previous alarms to detect changes
        self._wall_t0 = time.time()
        self._prev_lap = -1  # leader lap at the previous event (lap changes get a checkpoint)
        self._silent = indexer  # catching up after a jump (or indexing): no listeners, no call-log file
        self._hwm = 0  # events whose calls/alerts are already in the call-log file (a replayed jump adds none)
        self._is_indexer = indexer
        self._want_index = index
        self._helpers_started = False
        self._indexer_rt: "PitWallRuntime | None" = None
        self.replay_index = replay_index
        if self.replay_index is None and getattr(source, "seekable", False):
            self.replay_index = _replay.ReplayIndex(checkpoint_every)
        self._log_fh = None
        self.log_path: Path | None = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in (source.title or "session"))
            self.log_path = self.log_dir / f"calls-{slug}-{stamp}.jsonl"
            self._log_fh = open(self.log_path, "w", encoding="utf-8")

    # --------------------------------------------------------------- setup
    def _build_wall(self) -> None:
        from ..bench import history as H
        from ..config import bench_dir
        from ..pitloss import PitLossPrior
        from .engineer import Context
        from .wall import PitWall

        src = self.source
        start = src.start_utc
        history = self.history
        if history is None:
            path = bench_dir() / "history.json"
            history = H.load(path) if path.exists() else None
        prior = PitLossPrior()
        if history is not None and src.ref is not None and start is not None:
            prior = H.prior_for(history, src.ref.circuit_key, start)
        models = None
        if self.use_models is True and start is not None:
            try:
                from ..modelstore import latest_bundle_for

                models = latest_bundle_for(start)
            except Exception as exc:  # missing sklearn pickle, bad file ...
                log.warning("no model bundle: %s", exc)
        elif self.use_models not in (True, False, None):
            models = self.use_models  # an explicit bundle
        if models is not None:
            self.model_info = {"loaded": True, "cutoff_utc": str(models.cutoff_utc),
                               "train_end_utc": str(models.train_end_utc), "races": len(models.trained_on)}
        meta = dict(src.meta)
        engineers = self.engineers
        self._wall_quali = self._is_quali()
        if self._wall_quali:  # qualifying: the race engineers have nothing to say
            from .engineers.quali import QualiEngineer

            meta.setdefault("session_name", self.session_info.get("Name") or "Qualifying")
            engineers = engineers if engineers is not None else [QualiEngineer]
        ctx = Context.for_race(prior, meta, history=history if start is not None else None,
                               race_start_utc=start, team=self.team, models=models)
        self.wall = PitWall(ctx, engineers=engineers)

    def _is_quali(self) -> bool:
        """Qualifying or Sprint Qualifying: from the archive ref, else the SessionInfo message."""
        from ..quali import QUALI_NAMES

        name = self.source.meta.get("session_name") or self.session_info.get("Name")
        return name in QUALI_NAMES

    # --------------------------------------------------------------- loop
    def start(self) -> "PitWallRuntime":
        self.thread = threading.Thread(target=self.run, name="pitsense-pitwall", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        with self._voice_cv:  # messages still waiting are dropped; one being written is not logged after close
            self._voice_stop = True
            self._voice_jobs.clear()
            self._voice_cv.notify_all()
        if self._voice_thread is not None:
            self._voice_thread.join(timeout=2)
        if self.radio_worker is not None:
            self.radio_worker.stop()
        with self._log_lock:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the source to run out. True if it did."""
        if self.thread is not None:
            self.thread.join(timeout)
            return not self.thread.is_alive()
        return True

    def run(self) -> None:
        try:
            self.status = "running"
            self._start_helpers()
            for e in self.source.events(self.stop_event):
                if isinstance(e, _replay.Control):
                    self._control(e)
                    continue
                if self.wall is None:
                    self._ensure_wall()
                self.handle(e)
            if self.wall is not None:
                with self.sim_lock:
                    self.publish(force=True)
            self.status = "stopped" if self.stop_event.is_set() else "finished"
        except Exception as exc:
            log.exception("pit wall loop failed")
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            with self._log_lock:
                if self._log_fh:
                    self._log_fh.flush()
            if self._is_indexer and self.replay_index is not None:
                self.replay_index.done = self.status == "finished"
                self.replay_index.error = self.error
            self._notify({"event": "status", "status": self.status})

    def handle(self, e: _events.Event) -> None:
        with self.sim_lock:
            self._handle(e)
            self._hwm = max(self._hwm, self.n_events)
            lap = self.state.current_lap
            if lap != self._prev_lap:
                prev, self._prev_lap = self._prev_lap, lap
                ix = self.replay_index
                if ix is not None and self.wall is not None and ix.wanted(lap, prev) \
                        and (self._is_indexer or self.n_events not in ix.checkpoints):  # the indexing pass saved it already
                    self._checkpoint(prev)

    def _handle(self, e: _events.Event) -> None:
        state = self.state
        state.apply(e)
        self.n_events += 1
        self.last_event_wall = time.time()
        if e.topic == "SessionInfo" and isinstance(e.data, dict):
            _merge(self.session_info, e.data)
            if self.wall is not None and not self._wall_quali and self._is_quali():
                self._build_wall()  # a recording whose first message names the session
            self._feed_radio(e)
        if e.topic in FEED_TOPICS:  # feeds never change the timing state: no engineers, no snapshot
            if e.topic == "TeamRadio":
                self._note_radio_laps()
                self._feed_radio(e)
            if e.topic == "Position.z":
                self._maybe_push_positions()
            return
        if e.topic == "TimingData":
            if not self._feed_positions and feed_has_positions(e.data):
                self._feed_positions, self.inferred_order = True, False
            if not self._feed_positions and state.drivers and state.started:
                self.inferred_order = True
                infer_positions(state)
        try:
            self.wall.observe(state)
        except Exception:
            self.observe_errors += 1
            if self.observe_errors <= 3:
                log.exception("engineer observe failed at t=%.1f", state.t)
        lap_changed = state.current_lap != self._decided_lap
        if lap_changed or self._decided_t is None or state.t - self._decided_t >= self.decide_every_s:
            self._tick(lap_changed)

    # --------------------------------------------------------------- replay lab
    def _ensure_wall(self) -> None:
        self._build_wall()
        ix = self.replay_index
        if ix is not None and 0 not in ix.checkpoints and not ix.disabled:
            self._prev_lap = self.state.current_lap
            self._checkpoint(-1)  # the start of the race: where a jump to lap 0 or a replay from scratch begins

    def _checkpoint(self, prev_lap: int) -> None:
        ix = self.replay_index
        try:
            payload = _replay.capture(self)
        except Exception:
            log.exception("checkpoint failed; jumps will replay from the start")
            ix.disabled = True
            return
        ix.add(_replay.Checkpoint(self.n_events, self.state.current_lap, prev_lap, self.state.t, payload))
        if self._is_indexer:
            ix.indexed_idx, ix.indexed_lap = self.n_events, self.state.current_lap

    def _start_helpers(self) -> None:
        """Once: the bookmark scan (timing only, seconds) and, if asked, the background indexing pass."""
        if self._helpers_started or self._is_indexer or self.replay_index is None:
            return
        self._helpers_started = True
        src = self.source
        ix = self.replay_index
        threading.Thread(target=lambda: _replay.scan_marks(src.log, ix), name="pitsense-marks", daemon=True).start()
        if self._want_index:
            sub = ReplaySource(src.log, 0, ref=src.ref, title=src.title, start_utc=src.start_utc)
            self._indexer_rt = PitWallRuntime(sub, team=self.team, models=self.use_models, history=self.history,
                                              publish_every_s=self.publish_every_s, engineers=self.engineers,
                                              replay_index=ix, indexer=True, coarse=self.coarse)
            self._indexer_rt.start()

    def _control(self, ctl: "_replay.Control") -> None:
        ctl.taken = True
        try:
            if ctl.kind == "seek":
                ctl.result = self._seek_exec(int(ctl.args["lap"]))
            else:
                ctl.result = {"ok": False, "error": f"unknown command {ctl.kind}"}
        except Exception as exc:
            log.exception("replay command failed")
            ctl.result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._silent = self._is_indexer
            ctl.done.set()

    def _seek_exec(self, lap: int) -> dict:
        """Put the race at the first event where the leader is on lap >= ``lap`` (runs in the loop thread).

        A saved copy of state and engineers taken at exactly that event is restored if there is one; otherwise
        the nearest earlier copy (or the start, or the present if it is earlier) is restored and the log played
        forward silently through the normal handler until that event. Nothing after it is ever applied.
        """
        t0 = time.perf_counter()
        src, ix = self.source, self.replay_index
        lap = max(0, lap)
        with self.sim_lock:
            self._gen += 1  # radio messages on their way and the last question's copy belong to the old place
            self._frozen = None
            with self._voice_cv:
                self._voice_jobs.clear()
            if self.wall is None:
                self._ensure_wall()
            evs = src.log.events
            ck = ix.first_at_or_after(lap)
            via, replayed = "checkpoint", 0
            if ck is not None:
                _replay.restore(self, ck.payload)
            else:
                start = ix.before(lap)
                if start is None:
                    start = ix.checkpoints.get(0)
                if start is None:  # copies are off (a deep copy failed): rebuild from nothing
                    self._reset_fresh()
                elif not (self.state.current_lap < lap and self.n_events >= start.idx):
                    _replay.restore(self, start.payload)
                via = "replay"
                self._silent = True
                while self.state.current_lap < lap and self.n_events < len(evs):
                    self.handle(evs[self.n_events])
                    replayed += 1
                self._silent = self._is_indexer
            src.jump(self.n_events)
            self._prev_lap = self.state.current_lap
            self.last_event_wall = time.time()
            d = self.latest
            if d is not None:
                d["extra"]["wall_time"] = time.time()
                d["extra"]["speed"] = src.speed
                self.latest_bytes = _json_bytes(d)
                self._notify({"event": "snapshot", "data": self.latest_bytes})
            res = {"ok": True, "requested": lap, "lap": self.state.current_lap, "t": round(self.state.t, 1),
                   "events": self.n_events, "via": via, "events_replayed": replayed,
                   "clamped": self.state.current_lap < lap, "ms": round((time.perf_counter() - t0) * 1000)}
        self._notify({"event": "status", "status": self.status, "seek": res})
        return res

    def _reset_fresh(self) -> None:
        """Back to the state before the first event (used only when checkpoints are unavailable)."""
        keep = self.calls_log.maxlen
        self.state = RaceState(self.source.meta)
        self.session_info, self.calls_log, self.alerts_log = {}, deque(maxlen=keep), deque(maxlen=keep)
        self.inferred_order = self._feed_positions = False
        self.observe_errors = self.snapshot_errors = self.n_events = self.n_snapshots = 0
        self._decided_t, self._decided_lap, self._sent_t, self._last_calls = None, -1, None, {}
        self._last_call_info, self.changes = {}, {}
        self.radio, self.wall_msgs, self._wall_id = {}, deque(maxlen=WALL_KEEP), 0
        self._radio_lap, self._radio_list, self._radio_sig, self._seen_alerts = {}, [], (), set()
        self.positions, self._pos_t, self._prev_lap = {}, None, 0
        self.latest, self.latest_bytes = None, _json_bytes({"waiting": True})
        self._wall_quali = False
        self._build_wall()

    # --- the public controls (any thread)
    def _replay_ok(self) -> str | None:
        if not getattr(self.source, "seekable", False):
            return f"{self.source.mode} mode: replay controls are disabled"
        return None

    def replay_pause(self) -> dict:
        err = self._replay_ok()
        if err:
            return {"ok": False, "error": err}
        self.source.pause()
        return {"ok": True, **self.replay_status()}

    def replay_resume(self) -> dict:
        err = self._replay_ok()
        if err:
            return {"ok": False, "error": err}
        self.source.resume()
        return {"ok": True, **self.replay_status()}

    def replay_speed(self, speed) -> dict:
        err = self._replay_ok()
        if err:
            return {"ok": False, "error": err}
        try:
            v = 0.0 if str(speed).lower() == "max" else float(speed)
        except (TypeError, ValueError):
            return {"ok": False, "error": "speed must be a number or 'max'"}
        if not (0 <= v <= 1000):
            return {"ok": False, "error": "speed must be between 0 (max) and 1000"}
        self.source.set_speed(v)
        return {"ok": True, **self.replay_status()}

    def replay_seek(self, lap=None, mark: str | None = None, timeout: float = 600.0) -> dict:
        """Jump to a lap (or to the lap of a bookmark). Blocks until the jump is done; returns timing."""
        err = self._replay_ok()
        if err:
            return {"ok": False, "error": err}
        if mark is not None:
            m = self.replay_index.marks.get(str(mark))
            if m is None:
                return {"ok": False, "error": "no such bookmark"}
            lap = m["lap"]
        try:
            lap = int(lap)
        except (TypeError, ValueError):
            return {"ok": False, "error": "give {lap: N} or {mark: id}"}
        if lap < 0 or lap > 1000:
            return {"ok": False, "error": "lap out of range"}
        ctl = _replay.Control("seek", lap=lap)
        if self.thread is not None and self.thread.is_alive():
            self.source.request(ctl)
            end = time.monotonic() + timeout
            while not ctl.done.wait(0.05):
                if not self.thread.is_alive() and not ctl.taken:  # the loop ended just now: do it here
                    try:
                        self.source._ctl.remove(ctl)
                    except ValueError:
                        pass
                    break
                if time.monotonic() > end:
                    return {"ok": False, "error": "jump timed out"}
        if not ctl.done.is_set():
            self._control(ctl)
            if self.thread is not None and not self.thread.is_alive() and not self.stop_event.is_set():
                self.start()  # a finished replay plays on from the new place
        return ctl.result or {"ok": False, "error": "jump failed"}

    def replay_status(self) -> dict:
        src, ix = self.source, self.replay_index
        if not getattr(src, "seekable", False) or ix is None:
            return {"enabled": False, "mode": src.mode, "reason": self._replay_ok()}
        return {"enabled": True, "mode": src.mode, "paused": src.paused, "speed": src.speed,
                "lap": self.state.current_lap, "total_laps": self.state.total_laps, "t": round(self.state.t, 1),
                "pos": src.pos, "n_events": len(src.log.events), "checkpoints": ix.laps(),
                "indexed_lap": ix.indexed_lap, "index_done": ix.done, "index_error": ix.error,
                "copies": not ix.disabled}

    def replay_marks(self) -> dict:
        out = self.replay_status()
        if not out["enabled"]:
            return {**out, "marks": []}
        t = self.state.t
        out["marks"] = [{**m, "ahead": m["t"] > t} for m in self.replay_index.mark_list()]
        return out

    # --------------------------------------------------------------- publishing
    def _tick(self, lap_changed: bool) -> None:
        """Decide now; send to viewers on a leader lap or when ``publish_every_s`` has passed since the last send."""
        t0 = time.perf_counter()
        got = self._decide()
        if got is None:
            return
        t = self.state.t
        if lap_changed or self._sent_t is None or t - self._sent_t >= self.publish_every_s:
            self._send(*got, t0)

    def publish(self, force: bool = False) -> dict | None:
        """Decide and send now (end of a run, tests)."""
        t0 = time.perf_counter()
        got = self._decide()
        return None if got is None else self._send(*got, t0)

    def _decide(self) -> tuple[Snapshot, dict] | None:
        """The engineers' snapshot as of now; new calls and alerts go to the log."""
        state = self.state
        self._decided_t, self._decided_lap = state.t, state.current_lap
        try:
            snap: Snapshot = self.wall.snapshot(state)
        except Exception:
            self.snapshot_errors += 1
            if self.snapshot_errors <= 3:
                log.exception("snapshot failed at t=%.1f", state.t)
            return None
        d = snap.to_dict()
        self._log_changes(state, d, snap)
        return snap, d

    def _send(self, snap: Snapshot, d: dict, t0: float) -> dict:
        """Complete a decision with what only viewers need, serialise it once and hand it to them."""
        state = self.state
        d["extra"] = self._extra(state)
        try:
            d["extra"]["details"] = self.wall.details(state)
        except Exception:
            log.exception("engineer details failed at t=%.1f", state.t)
        try:
            d["extra"]["charts"] = charts.build(state, self.wall.memory, snap.focus)
        except Exception:
            log.exception("chart history failed at t=%.1f", state.t)
        d["extra"]["radio"] = dict(self.radio)
        d["extra"].update(self._feed_extra(state))
        d["schema"] = SNAPSHOT_SCHEMA
        d["predictions"] = self._predictions(state, d)
        payload = _json_bytes(d)
        ms = (time.perf_counter() - t0) * 1000
        self.snap_ms.append(ms)
        self._sent_t = state.t
        self.n_snapshots += 1
        self._log_alarms_if_changed()  # print to console if alarms change
        with self.lock:
            self.latest, self.latest_bytes = d, payload
        self._notify({"event": "snapshot", "data": payload})
        return d

    # --------------------------------------------------------------- positions, track, team radio
    def _feed_radio(self, e: _events.Event) -> None:
        """Queue clip downloads (the worker never blocks this loop). Safe if the recorder queued them too."""
        if self.radio_worker is not None:
            from ..live import feed_radio

            feed_radio(self.radio_worker, e.topic, e.data)

    def _find_track(self) -> None:
        """The circuit outline known before this race (``trackmap.track_asof``); once per session.

        The circuit and start come from the archive ref, or, live, from the SessionInfo message
        (``Meeting.Circuit.Key``, ``StartDate`` + ``GmtOffset``); until that arrives we ask again later.
        """
        from ..trackmap import session_circuit, track_asof

        ref, start = self.source.ref, self.source.start_utc
        if ref is not None and start is not None:
            key = ref.circuit_key
        else:
            got = session_circuit(self.session_info)
            if got is None:
                return
            key, start = got
        self._track_done = True
        try:
            t = track_asof(key, start)
        except Exception:
            log.exception("track outline failed")
            return
        if t:
            self.track = {k: t.get(k) for k in ("x", "y", "start", "pit")} | {"key": f"{key}:{t.get('end_utc')}"}
            if self._wall_quali and self.wall is not None:  # the qualifying engineer judges traffic on it
                self.wall.engineer("quali").outline = self.track

    def _live_track(self) -> None:
        """No stored outline: build one from the cars' published positions once 2 laps are done (as-of)."""
        now = time.monotonic()
        if self.track is not None or self.state.current_lap < 3 or self._live_track_tries >= 20                 or now - self._live_track_wall < 10.0:
            return
        self._live_track_wall, self._live_track_tries = now, self._live_track_tries + 1
        try:
            from ..trackmap import live_outline

            t = live_outline(self.state)
        except Exception:
            log.exception("live track outline failed")
            return
        if t:
            self.track = {k: t.get(k) for k in ("x", "y", "start", "pit")} | {"key": f"live:{round(self.state.t)}", "provisional": True}

    def _positions(self) -> dict:
        """Newest published position of every car: {car: [x, y, on_track]} (as-of the feed clock)."""
        try:
            raw = self.state.feeds.telemetry.latest_position()
        except Exception:
            return {}
        return {c: [round(p["x"]), round(p["y"]), 1 if p["on_track"] else 0] for c, p in raw.items()
                if p["x"] == p["x"] and p["y"] == p["y"]}

    def _note_radio_laps(self) -> None:
        for m in self.state.feeds.radio.messages(last_s=1.0):
            self._radio_lap.setdefault(m.path, self.state.current_lap)

    def _team_radio(self, force: bool = False) -> list[dict]:
        """Latest driver radio messages (real clips only). Text is None until the transcript is known."""
        now = time.monotonic()
        if not force and now - self._radio_wall < RADIO_EVERY_WALL_S:
            return self._radio_list
        self._radio_wall = now
        store = self.state.feeds.radio
        tla = {n: d.tla for n, d in self.state.drivers.items()}
        live = self.radio_worker is not None  # the mp3 may still be downloading: link it only once it is there
        idx: dict[str, int] = {}
        out = []
        for m in store.messages():
            i = idx[m.car] = idx.get(m.car, -1) + 1  # index among that car's messages (see team_radio_file)
            lap = self._radio_lap.setdefault(m.path, self.state.current_lap)
            out.append({"id": f"{m.car}:{i}", "car": m.car, "tla": tla.get(m.car), "t": round(m.t, 1), "lap": lap,
                        "text": m.text or None, "audio": f"/api/teamradio?car={m.car}&i={i}" if m.audio and (not live or Path(m.audio).is_file()) else None})
        self._radio_list = out[-RADIO_KEEP:]
        return self._radio_list

    def _feed_extra(self, state: RaceState) -> dict:
        if not self._track_done:
            self._find_track()
        self._live_track()
        self.positions = self._positions()
        self._pos_t = state.t
        return {"positions": {"t": round(state.t, 1), "cars": self.positions},
                "track": self.track, "team_radio": self._team_radio(True), "telemetry": self._telemetry(),
                "wall_msgs": list(self.wall_msgs)}

    def _championship(self, state: RaceState) -> list | None:
        """Drivers' standings before this race (sessions that ended before it started) and if it finished in the
        current running order. None without cached results (``pitsense standings --year Y``) or outside a race."""
        if self._wall_quali or self.source.start_utc is None:
            return None
        if not hasattr(self, "_standings"):
            from .. import standings as ST

            start = self.source.start_utc
            res = ST.load(start.year)
            self._standings = (ST.season_points(ST.before(res, start), start.year), start.year) if res else None
        if self._standings is None:
            return None
        from ..standings import project

        pts, year = self._standings
        order = [(d.tla, d.team) for d in state.running_order() if d.tla and not d.retired]
        return project(pts, order, year)

    def _telemetry(self) -> dict:
        """Newest published CarData sample of every car: {car: [speed, gear, throttle, brake, drs, rpm]} (as-of)."""
        try:
            raw = self.state.feeds.telemetry.latest_telemetry()
        except Exception:
            return {}

        def num(v):
            return None if v is None or v != v else round(float(v))

        return {c: [num(x.get(k)) for k in ("speed", "gear", "throttle", "brake", "drs", "rpm")] for c, x in raw.items()}

    def _predictions(self, state: RaceState, d: dict) -> dict:
        """What the models said, pre-registered with when: ``as_of`` (session time of the decision), ``input_t``
        (newest feed message used) and the model bundle; per car the pit probabilities and the rejoin position."""
        cars = {}
        for n, v in (d.get("cars") or {}).items():
            p1 = v.get("models__pit_prob_1", v.get("rivals__pit_prob_1"))
            p3 = v.get("models__pit_prob_3", v.get("rivals__pit_prob_3"))
            cars[n] = {"pit_prob_1": p1, "pit_prob_3": p3, "rejoin_if_box_now": v.get("pitstop__rejoin_if_box_now"),
                       "undercut_threat": v.get("rivals__undercut_threat")}
        return {"as_of": round(state.t, 3), "input_t": round(state.t, 3), "model": self.model_info, "cars": cars}

    def _maybe_push_positions(self) -> None:
        """A light "pos" event: car positions (and radio changes) between snapshots, about 3 Hz on the wall clock."""
        state, now = self.state, time.monotonic()
        if self._silent:
            return
        if self._pos_t is not None and (state.t - self._pos_t < POS_EVERY_S or now - self._pos_wall < POS_MIN_WALL_S):
            return
        self._pos_t, self._pos_wall = state.t, now
        self.positions = pos = self._positions()
        if not pos:
            return
        msg = {"t": round(state.t, 1), "speed": self.source.speed, "cars": pos, "telemetry": self._telemetry()}
        radio = self._team_radio()
        sig = (len(radio), sum(1 for r in radio if r["text"]))
        if sig != self._radio_sig:
            self._radio_sig, msg["team_radio"] = sig, radio
        self._notify({"event": "pos", "data": _json_bytes(msg)})

    @staticmethod
    def _teams(state: RaceState) -> dict:
        """team name -> car numbers: lets each browser pick its own focus team."""
        out: dict[str, list[str]] = {}
        for n, d in state.drivers.items():
            if d.team:
                out.setdefault(d.team, []).append(n)
        return {t: sorted(c, key=lambda x: int(x) if x.isdigit() else 999) for t, c in sorted(out.items())}

    def _extra(self, state: RaceState) -> dict:
        rc = [{"t": round(m.t, 1), "message": m.message, "category": m.category}
              for m in state.rc if "BLUE FLAG" not in m.message][-6:]
        return {
            "mode": self.source.mode, "speed": self.source.speed, "title": self.source.title,
            "inferred_order": self.inferred_order,
            "model": self.model_info,
            "weather": dict(state.weather),
            "colours": {n: d.team_colour for n, d in state.drivers.items() if d.team_colour},
            "teams": self._teams(state),
            "rc": rc,
            "track_status_since": round(state.track_status_since, 1),
            "team": self.team.team, "wall_time": time.time(),
            "changes": dict(self.changes), "acks": dict(self.acks), "risk": self.team.risk,
            "championship": self._championship(state),
        }

    def _log_changes(self, state: RaceState, d: dict, snap: Snapshot | None = None) -> None:
        wall = round(time.time(), 3)
        voice = snap is not None and self.voice is not False and not (self._silent or self._is_indexer)
        for c in d["calls"]:
            key = _call_key(c)
            prev = self._last_calls.get(c["car"])
            self._last_calls[c["car"]] = key
            if prev == key or (prev is None and c["action"] == "NO_CALL"):
                continue
            car = state.drivers.get(c["car"])
            rec = {"kind": "call", "t": c["t"], "wall": wall, "lap": state.current_lap,
                   "car_lap": (car.laps + 1) if car else None, **{k: v for k, v in c.items() if k not in ("t",)}}
            before = self._last_call_info.get(c["car"])
            if before is not None:
                rec["change"] = self.changes[c["car"]] = call_change(before, c, state.track_status)
            self._last_call_info[c["car"]] = {"action": c["action"], "compound": c.get("compound"), "track": state.track_status,
                                              "codes": [r.get("code") for r in c.get("reasons") or ()]}
            logged = self._record(rec, self.calls_log)
            call = next((x for x in snap.calls if x.car == c["car"]), None) if voice else None
            if call is not None and call.action != "NO_CALL":  # its radio message follows from the voice thread
                self._wall_id += 1
                self._voice_put(_VoiceJob(call, snap, self._wall_id, c["t"], round(state.t, 1), state.current_lap,
                                          self._gen, logged))
        for a in d["alerts"]:
            key = (a["engineer"], a["code"], a.get("car"), a.get("since"))
            if key in self._seen_alerts:
                continue
            self._seen_alerts.add(key)
            self._record({"kind": "alert", "wall": wall, "lap": state.current_lap, **a}, self.alerts_log)
            self._wall_id += 1
            self.wall_msgs.append({"id": self._wall_id, "kind": "alert", "car": a.get("car"), "t": round(state.t, 1),
                                   "lap": state.current_lap, "text": a.get("message"), "engineer": a.get("engineer"),
                                   "severity": a.get("severity")})

    def _record(self, rec: dict, mem: deque) -> bool:
        """Keep a call or alert record. True if it went to the log file (a replayed jump never logs again)."""
        mem.append(rec)
        if self.replay_index is not None:
            try:
                _replay.marks_from_record(self.replay_index, rec, self.team.focus(self.state), self.state)
            except Exception:
                log.exception("bookmark failed")
        return self.n_events > self._hwm and self._write_log(rec)

    def _write_log(self, rec: dict) -> bool:
        with self._log_lock:
            if not self._log_fh:
                return False
            self._log_fh.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
            self._log_fh.flush()
            return True

    # --------------------------------------------------------------- the voice (its own thread)
    def _voice_put(self, job: _VoiceJob) -> None:
        """Queue a call for the voice; never blocks. A newer call for a car replaces the one still waiting."""
        car = job.call.car
        with self._voice_cv:
            if self._voice_stop:
                return
            if car in self._voice_jobs:
                self.voice_stats["superseded"] += 1
            elif len(self._voice_jobs) >= VOICE_QUEUE:  # the longest-waiting message is stale by now
                del self._voice_jobs[next(iter(self._voice_jobs))]
                self.voice_stats["dropped"] += 1
            self._voice_jobs[car] = job  # a replaced call keeps its car's place in the queue
            if self._voice_thread is None:
                self._voice_thread = threading.Thread(target=self._voice_loop, name="pitsense-voice", daemon=True)
                self._voice_thread.start()
            self._voice_cv.notify()

    def _voice_loop(self) -> None:
        """One radio message at a time, the car that has waited longest first."""
        while True:
            with self._voice_cv:
                while not self._voice_jobs and not self._voice_stop:
                    self._voice_cv.wait()
                if self._voice_stop:
                    return
                job = self._voice_jobs.pop(next(iter(self._voice_jobs)))
            if job.gen != self._gen:
                continue
            t0 = time.perf_counter()
            text = self._say(job)
            if text:
                self.voice_stats["done"] += 1
                self.voice_stats["last_ms"] = round((time.perf_counter() - t0) * 1000)
                self._voice_done(job, text)

    def _say(self, job: _VoiceJob) -> str | None:
        """The voice's radio message for a waiting call (None if the voice is off or fails)."""
        try:
            from ..voice.api import Voice, say

            with self._voice_lock:
                if self.voice is None:
                    self.voice = Voice()  # loaded on the first call, in this thread
                voice = self.voice
                return say(job.call, job.snap, voice) if voice is not False else None
        except Exception:
            log.exception("voice failed; continuing without it")
            self.voice = False
            return None

    def _voice_done(self, job: _VoiceJob, text: str) -> None:
        """A radio message is ready: log it, add it to the conversation, send it to the viewers."""
        c = job.call
        if job.logged:  # a record of its own after the call's, which stays the same at any speed
            self._write_log({"kind": "radio", "car": c.car, "t": job.t_call, "action": c.action, "lap": job.lap,
                             "wall": round(time.time(), 3), "text": text})
        with self.sim_lock:
            if job.gen != self._gen:  # a jump since: the race shown is elsewhere
                return
            self.radio[c.car] = {"text": text, "lap": job.lap, "action": c.action, "id": job.id}
            self.wall_msgs.append({"id": job.id, "kind": "voice", "car": c.car, "t": job.t, "lap": job.lap,
                                   "text": text, "action": c.action})
            old = self.latest
            extra = {"radio": dict(self.radio), "wall_msgs": list(self.wall_msgs)}
        if old is None:
            return
        new = {**old, "extra": {**old["extra"], **extra}}
        payload = _json_bytes(new)
        with self.sim_lock:  # in order: a snapshot published since carries the message already
            if self.latest is not old:
                return
            with self.lock:
                self.latest, self.latest_bytes = new, payload
            self._notify({"event": "snapshot", "data": payload})

    # --------------------------------------------------------------- listeners
    def subscribe(self, limit: int | None = None) -> queue.Queue | None:
        """A queue of events for one viewer, or None if ``limit`` viewers are connected (checked atomically)."""
        q: queue.Queue = queue.Queue(maxsize=8)
        with self.lock:
            if limit is not None and len(self.listeners) >= limit:
                return None
            self.listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.listeners:
                self.listeners.remove(q)

    def _notify(self, msg: dict) -> None:
        if self._silent:
            return
        with self.lock:
            targets = list(self.listeners)
        for q in targets:
            try:
                q.put_nowait(msg)
            except queue.Full:  # slow browser: drop its oldest message, keep the newest
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (queue.Empty, queue.Full):
                    pass

    # --------------------------------------------------------------- API views
    def _check_alarms(self) -> list[str]:
        """Compute current health alarms: model not loaded, no calls after 5 laps, feed stale >30s, engineer errors."""
        now = time.time()
        last_event_age = None if self.last_event_wall is None else (now - self.last_event_wall)

        alarms = []
        if self.state.session_status == "Started" and self.state.current_lap >= 5:
            # Check if no calls have been produced for focus cars after 5 laps
            if not self.calls_log:
                alarms.append("RED: No strategy calls produced after 5 laps (may indicate strategy bug)")

        if self.use_models is True and not self.model_info.get("loaded"):
            alarms.append("RED: Model bundle requested but not loaded")

        quiet = getattr(self.source, "paused", False) or self.status in ("finished", "stopped")  # nothing is due
        if last_event_age is not None and last_event_age > 30 and not quiet:
            alarms.append(f"RED: Feed stale for {last_event_age:.0f}s (>30s)")

        if self.observe_errors > 0:
            alarms.append(f"RED: {self.observe_errors} engineer errors")

        return alarms

    def _log_alarms_if_changed(self) -> None:
        """Log to console when health alarms change (called during publish)."""
        alarms = self._check_alarms()
        if alarms != self._last_alarms:
            self._last_alarms = alarms
            if alarms:
                for alarm in alarms:
                    log.error(alarm)
            else:
                log.info("All health checks passed")

    def health(self) -> dict:
        ms = sorted(self.snap_ms)
        now = time.time()
        last_event_age = None if self.last_event_wall is None else (now - self.last_event_wall)
        alarms = self._check_alarms()

        return {
            "status": self.status, "error": self.error, "mode": self.source.mode,
            "title": self.source.title, "speed": self.source.speed,
            "t": round(self.state.t, 1), "lap": self.state.current_lap, "total_laps": self.state.total_laps,
            "events": self.n_events, "snapshots": self.n_snapshots,
            "last_event_age_s": None if self.last_event_wall is None else round(last_event_age, 1),
            "uptime_s": round(now - self.started_wall, 1),
            "inferred_order": self.inferred_order, "model": self.model_info,
            "viewers": len(self.listeners),
            "observe_errors": self.observe_errors, "snapshot_errors": self.snapshot_errors,
            "snapshot_ms_p50": round(ms[len(ms) // 2], 2) if ms else None,
            "snapshot_ms_max": round(ms[-1], 2) if ms else None,
            "call_log": str(self.log_path) if self.log_path else None,
            "recorder_error": getattr(self.source, "recorder_error", None),
            "recorder_restarts": getattr(self.source, "recorder_restarts", None),
            "radio": None if self.radio_worker is None else dict(self.radio_worker.stats, error=self.radio_worker.error),
            "voice": None if self.voice is False else dict(self.voice_stats, waiting=len(self._voice_jobs)),
            "track": None if self.track is None else ("provisional" if self.track.get("provisional") else "stored"),
            "alarms": alarms,
            "replay": self.replay_status(),
        }

    def team_radio_file(self, car: str, i: int) -> Path | None:
        """The mp3 of a driver's i-th published radio message, or None. Only files inside the session's
        TeamRadio folder are ever returned; ``car`` and ``i`` come from the browser and are validated."""
        if not (isinstance(car, str) and car.isdigit() and len(car) <= 3 and isinstance(i, int) and i >= 0):
            return None
        store = self.state.feeds.radio
        base = store.transcripts.dir
        if base is None:
            return None
        msgs = store.messages(car)  # published only: a clip is not served before it exists
        if i >= len(msgs) or not msgs[i].audio:
            return None
        try:
            root = (Path(base) / "TeamRadio").resolve()
            f = Path(msgs[i].audio).resolve()
            f.relative_to(root)
        except (ValueError, OSError):
            return None
        return f if f.suffix.lower() == ".mp3" and f.is_file() else None

    def wall_text(self, car: str, msg_id: int | None = None) -> str | None:
        """Text of one of our own pit-wall voice messages (by id), else the car's latest."""
        if msg_id is not None:
            for m in list(self.wall_msgs):  # a copy: the loop and the voice thread append to it
                if m["id"] == msg_id and m["kind"] == "voice" and m["car"] == car:
                    return m["text"]
            return None
        r = self.radio.get(car)
        return r["text"] if r else None

    # --------------------------------------------------------------- ask the pit wall
    def _ask_voice(self):
        """The voice for answers: the loaded model, or templates only when the voice is off (or failed)."""
        from ..voice.api import Voice

        if self.voice is False:
            if self._ask_tpl is None:
                self._ask_tpl = Voice(use_model=False)
            return self._ask_tpl
        if self.voice is None:
            self.voice = Voice()
        return self.voice

    def _push_wall(self, **m) -> dict:
        with self.sim_lock:
            self._wall_id += 1
            m = {"id": self._wall_id, "t": round(self.state.t, 1), "lap": self.state.current_lap, "ask": True, **m}
            self.wall_msgs.append(m)
            return m

    def _freeze(self) -> _Frozen:
        """A private copy of the state and the engineers as of now, for questions (call under sim_lock).

        Questions run on it after the lock is released, so the loop never waits for a simulation, and nothing a viewer
        asks reaches what the live engineers keep (memory, call history, the strategy cache): the call log is the same
        whoever asks what. The Context (models, history, priors; read-only) is shared, its in-place caches and the
        models engineer's memo of predictions are copied shallowly (their entries never change once made; a deep copy
        of that memo took ~1.7 s at lap 40, the rest of the copy ~30 ms). One copy serves every question until an
        event, a snapshot or a jump.
        """
        key = (self._gen, self.n_events, self.n_snapshots)
        f = self._frozen
        if f is not None and f.key == key:
            return f
        ctx = self.wall.ctx
        own = copy.copy(ctx)
        for k, v in ctx.__dict__.items():
            if k.startswith("_") and type(v) is dict:  # caches filled in place (the strategy analysis)
                own.__dict__[k] = dict(v)
        memo = {id(ctx): own}
        rows = getattr(next((e for e in self.wall.engineers if e.name == "models"), None), "_memo", None)
        if type(rows) is dict:
            memo[id(rows)] = dict(rows)
        state, wall = copy.deepcopy((self.state, self.wall), memo)
        self._frozen = f = _Frozen(key, state, wall)
        return f

    @staticmethod
    def _busy() -> dict:
        return {"ok": False, "busy": True, "error": "the pit wall is answering another question: try again in a moment"}

    def ask(self, car: str | None, text: str, *, source: str = "text") -> dict:
        """Answer a question about ``car`` as of now: a what-if on the simulator, or a fact through the voice.

        Adds the question ("you") and the answer (pit wall, spoken via ``/api/radio.wav?i=<id>``) to the conversation.
        """
        from .. import whatif
        from ..voice import api as vapi
        from ..voice.facts import from_snapshot
        from .types import Call

        text = " ".join((text or "").split())[:300]
        if not text:
            return {"ok": False, "error": "empty question"}
        t0 = time.perf_counter()
        if not self._ask_gate.acquire(timeout=ASK_WAIT_S):
            return self._busy()
        res = snap = None
        try:
            with self.sim_lock:  # only the copy is made under the loop's lock
                state = self.state
                if self.wall is None or self._wall_quali or not state.drivers:
                    return {"ok": False, "error": "the pit wall has no race data yet"}
                if not car or car not in state.drivers:
                    focus = self.team.focus(state)
                    order = [d.number for d in state.running_order() if d.running]
                    car = focus[0] if focus else (order[0] if order else next(iter(state.drivers)))
                tla_of = {n: d.tla for n, d in state.drivers.items()}
                q = whatif.parse_question(text, tla_of, car)
                f = self._freeze()
            state = f.state
            if q.kind in ("whatif", "sc", "rain", "dry"):
                res = whatif.what_if(f.wall, state, car, q)
                answer, src = res.get("answer") or whatif.compose(res), "whatif"
            else:
                if f.snap is None:
                    f.snap = f.wall.snapshot(state)
                snap = f.snap
                call = next((c for c in snap.calls if c.car == car), None) or Call(snap.t, car, "NO_CALL")
                nb = {"ahead": snap.cars.get(car, {}).get("rivals__ahead"), "behind": snap.cars.get(car, {}).get("rivals__behind")}
        finally:
            self._ask_gate.release()
        you = self._push_wall(kind="you", car=car, text=text, source=source)
        if snap is not None:
            try:
                with self._voice_lock:
                    answer, src = self._fact_answer(q, snap, call, nb, state, car, from_snapshot, vapi)
            except Exception:
                log.exception("voice failed on a question; using the template")
                self.voice = False
                with self._voice_lock:
                    answer, src = self._fact_answer(q, snap, call, nb, state, car, from_snapshot, vapi)
        reply = self._push_wall(kind="voice", car=car, text=answer, action=None, source=src)
        return {"ok": True, "car": car, "tla": tla_of.get(car, car), "you": you, "reply": reply, "answer": answer, "source": src,
                "intent": q.to_dict(), "whatif": res, "audio": f"/api/radio.wav?car={car}&i={reply['id']}",
                "ms": round((time.perf_counter() - t0) * 1000)}

    def _fact_answer(self, q, snap, call, nb, state, car, from_snapshot, vapi) -> tuple[str, str]:
        voice = self._ask_voice()
        facts = from_snapshot(call, snap)
        tla = {n: d.tla for n, d in state.drivers.items()}
        src = lambda r: r.source  # noqa: E731
        if q.fact == "sc_prob":
            p = snap.race.get("strategy__sc_prob_5")
            if isinstance(p, (int, float)):
                return f"A safety car in the next five laps is {int(round(100 * p))} percent likely.", "facts"
        elif q.fact == "gap":
            ref = q.rival_ref
            target = q.rival or q.target
            if target is None and ref:
                target = nb.get(ref)
            if target is not None:
                near = {x.car for x in (facts.ahead, facts.behind) if x}
                if str(target) in near:
                    r = voice.write(facts, "ask_gap", str(target))
                    return r.text, src(r)
                a, b = state.drivers.get(car), state.drivers.get(str(target))
                if a and b and a.gap_to_leader is not None and b.gap_to_leader is not None:
                    d = b.gap_to_leader - a.gap_to_leader
                    return f"{tla.get(str(target), target)} is {abs(d):.1f} seconds {'behind' if d > 0 else 'ahead of'} {a.tla}.", "facts"
        elif q.fact in ("tyre_age", "pit_window", "plan_b", "why"):
            r = voice.write(facts, f"ask_{q.fact}", None)
            return r.text, src(r)
        r = voice.write(facts, "free", q.text.strip().lower().rstrip("?"))
        return r.text, src(r)

    def ask_audio(self, car: str | None, data: bytes) -> dict:
        """Transcribe a spoken question (browser audio) and answer it like a typed one."""
        fn = self.transcriber
        if fn is None:
            from .. import asr

            fn = asr.transcribe
        t0 = time.perf_counter()
        try:
            heard = fn(data)
        except Exception as exc:
            log.warning("transcription failed: %s", exc)
            return {"ok": False, "error": f"could not transcribe the audio: {type(exc).__name__}: {exc}"}
        if not heard or not heard.strip():
            return {"ok": False, "error": "no speech heard", "heard": ""}
        out = self.ask(car, heard, source="voice")
        out["heard"] = heard
        out["asr_ms"] = round((time.perf_counter() - t0) * 1000) - out.get("ms", 0)
        return out

    def whatif(self, car: str | None, stop_lap, compound: str | None) -> dict:
        """Run the simulator for "car stops on lap N for COMPOUND" as of now (no future data); compact result for the dashboard."""
        from .. import whatif as wi
        from .engineers.strategy.priors import DRY

        comp = str(compound).upper() if compound else None
        if comp is not None and comp not in DRY + wi.WET:  # wet tyres go to the wet simulator
            return {"ok": False, "error": f"compound must be one of {', '.join(DRY + wi.WET)}"}
        try:
            lap = int(stop_lap)
        except (TypeError, ValueError):
            return {"ok": False, "error": "stop lap must be a whole number"}
        t0 = time.perf_counter()
        if not self._ask_gate.acquire(timeout=ASK_WAIT_S):
            return self._busy()
        try:
            with self.sim_lock:  # only the copy is made under the loop's lock
                state = self.state
                if self.wall is None or self._wall_quali or not state.drivers:
                    return {"ok": False, "error": "the pit wall has no race data yet"}
                car = str(car or "")
                if car not in state.drivers:
                    return {"ok": False, "error": f"unknown car {car!r}"}
                f = self._freeze()
            q = wi.Question(f"stop on lap {lap}", "whatif", (wi.Spec("lap", lap=lap, compound=comp),))
            r = wi.what_if(f.wall, f.state, car, q)
        finally:
            self._ask_gate.release()
        if not r.get("ok"):
            return {"ok": False, "error": r.get("why") or "no answer", "car": car}
        keep = ("car", "tla", "lap", "total", "position", "compound", "tyre_age", "scenario", "versus", "delta_pos", "delta_time_s",
                "p_gain", "p_loss", "legal", "notes", "reasons", "answer", "n_sims", "t", "wet", "horizon_laps")
        return {"ok": True, **{k: r.get(k) for k in keep}, "stop_lap": max(lap, r["lap"]), "ms": round((time.perf_counter() - t0) * 1000)}

    def calls(self) -> dict:
        with self.lock:
            cur = (self.latest or {}).get("calls", [])
            hist = list(self.calls_log)
        return {"current": cur, "log": hist, "acks": list(self.acks_log)}

    def analysis(self, kind: str, car: str | None = None, other: str | None = None, lap=None, ref_lap=None) -> dict:
        """The analysis panels as of now (``pitsense.insights``, ``pitsense.coach``): deg | evolution | compare | trace |
        coach. ``coach`` compares ``car``'s ``lap`` with ``other``'s ``ref_lap`` (default: ``car``'s own best clean lap)."""
        from .. import coach as C
        from .. import insights as I

        def as_lap(v):
            try:
                return int(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                return -1

        lap, ref_lap = as_lap(lap), as_lap(ref_lap)
        if -1 in (lap, ref_lap):
            return {"ok": False, "error": "lap must be a whole number"}
        with self.sim_lock:
            st = self.state
            if car is not None and car not in st.drivers or other not in (None, "") and other not in st.drivers:
                return {"ok": False, "error": "unknown car"}
            if kind == "deg":
                return {"ok": True, "kind": kind, "stints": I.tyre_deg(st, [car] if car else None)}
            if kind == "evolution":
                return {"ok": True, "kind": kind, **I.track_evolution(st)}
            if kind == "compare" and car and other:
                return {"ok": True, "kind": kind, **I.compare(st, car, other)}
            if kind == "trace" and car and lap:
                tr = I.speed_trace(st, car, lap)
                return {"ok": True, "kind": kind, **tr} if tr else {"ok": False, "error": "no telemetry for that lap (the feeds keep about ten minutes)"}
            if kind == "coach" and car and lap:
                src = C.F1Adapter(st)
                rc = other or car
                rl = ref_lap or src.best_lap(rc)
                a, b = (src.lap(rc, rl) if rl else None), src.lap(car, lap)
                if a is None or b is None:
                    return {"ok": False, "error": "no telemetry for one of the laps (the feeds keep about ten minutes)"}
                return {"ok": True, "kind": kind, **C.compare(a, b)}
        return {"ok": False, "error": "kind must be deg, evolution, compare (car, other), trace (car, lap) or coach (car, lap)"}

    def set_risk(self, risk) -> dict:
        """How the strategy ranks plans from now on (expected | protect | aggressive); the next decision replans."""
        import dataclasses

        risk = str(risk or "").lower()
        if risk not in ("expected", "protect", "aggressive"):
            return {"ok": False, "error": "risk must be expected, protect or aggressive"}
        with self.sim_lock:
            self.team = dataclasses.replace(self.team, risk=risk)
            if self.wall is not None:
                self.wall.ctx.team = self.team
        return {"ok": True, "risk": risk}

    def ack(self, car, decision, reason: str | None = None, call_t=None) -> dict:
        """The operator's answer to a car's call: accept or reject, with an optional reason. Kept for the post-race
        review (``shadow-score``) and written to the call log; it never changes what the engineers decide."""
        decision = str(decision or "").lower()
        if decision not in ("accept", "reject"):
            return {"ok": False, "error": "decision must be accept or reject"}
        car, reason = str(car or ""), " ".join(str(reason or "").split())[:200]
        with self.sim_lock:
            calls = [r for r in self.calls_log if r.get("kind") == "call" and r.get("car") == car and r.get("action") != "NO_CALL"]
            if call_t is not None:
                try:
                    calls = [r for r in calls if abs(float(r["t"]) - float(call_t)) < 1e-3]
                except (TypeError, ValueError):
                    return {"ok": False, "error": "call_t must be a number"}
            if not calls:
                return {"ok": False, "error": f"no call to answer for car {car!r}"}
            c = calls[-1]
            rec = {"kind": "ack", "car": car, "call_t": c["t"], "action": c["action"], "compound": c.get("compound"),
                   "call_lap": c.get("car_lap"), "decision": decision, "reason": reason or None,
                   "t": round(self.state.t, 1), "lap": self.state.current_lap, "wall": round(time.time(), 3)}
            self.acks_log.append(rec)
            self.acks[car] = rec
        self._write_log(rec)
        return {"ok": True, "ack": rec}

    def alerts(self) -> dict:
        with self.lock:
            cur = (self.latest or {}).get("alerts", [])
            hist = list(self.alerts_log)
        return {"active": cur, "log": hist}
