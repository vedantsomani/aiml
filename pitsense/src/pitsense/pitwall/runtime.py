"""Run the pit wall on a race and publish what it sees.

One loop, three kinds of source::

    archive replay at N x speed     Source -> events (paced by the session clock)
    a recording followed as it grows
    live: start the recorder, follow its file

    loop:  state.apply(event) -> wall.observe(state) -> maybe publish a snapshot

A snapshot is published when the session clock has advanced ``publish_every_s`` since the
last one, and on every leader lap. Both triggers use the session clock, never the wall clock,
so the same input gives the same snapshots and the same call log at any replay speed.
Each published snapshot is serialised once and handed to the web server (``pitsense.web``);
calls and alerts are appended to a JSONL log with the session time and the wall-clock time.

Also here: ``pitsense pitwall`` and ``pitsense shadow-score`` (compare a call log with the
stops that really happened).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .. import events as _events  # the feeder: reads the log; exempt by name in tests/test_pitwall.py
from ..config import FEED_TOPICS
from ..state import RaceState
from .types import Snapshot, TeamConfig

log = logging.getLogger("pitsense.pitwall")

PUBLISH_EVERY_S = 3.0  # session seconds between snapshots (wall seconds at 1x)
MAX_SPEED_TICK_S = 30.0  # snapshot spacing when replaying as fast as possible
LEAD_IN_S = 60.0  # replay starts pacing this long before the session start
LOG_KEEP = 500  # calls and alerts kept in memory for the API
POS_EVERY_S = 0.5  # session seconds between car positions
POS_MIN_WALL_S = 0.3  # ...and never faster than ~3 Hz on the wall clock (high replay speeds)
RADIO_EVERY_WALL_S = 1.0  # how often the team-radio list is rebuilt
RADIO_KEEP = 80  # driver radio messages in the snapshot
WALL_KEEP = 120  # pit-wall conversation entries (voice calls and alerts) in the snapshot


# ----------------------------------------------------------------------------- sources
class Source:
    """Yields events. ``start_utc`` (race start) and ``meta`` are known once the first event is out."""

    mode = "source"
    speed: float = 1.0
    ref = None  # archive SessionRef, if any
    title = ""
    start_utc: datetime | None = None
    meta: dict

    def events(self, stop: threading.Event):  # pragma: no cover - interface
        raise NotImplementedError


class ReplaySource(Source):
    """An EventLog replayed against the wall clock at ``speed`` x (0 = as fast as possible)."""

    mode = "replay"

    def __init__(self, log_: _events.EventLog, speed: float = 1.0, *, ref=None, title: str = "", start_utc=None) -> None:
        self.log, self.speed, self.ref = log_, float(speed), ref
        self.meta = dict(log_.meta)
        self.title = title or (ref.slug if ref else "replay")
        self.start_utc = start_utc if start_utc is not None else (ref.start_utc if ref else None)

    def events(self, stop: threading.Event):
        # the grid wait before the start is not paced: pacing begins LEAD_IN_S before the lights
        t0 = self.log.start
        for e in self.log.events:
            if e.topic == "SessionStatus" and isinstance(e.data, dict) and e.data.get("Status") == "Started":
                t0 = max(t0, e.t - LEAD_IN_S)
                break
        t0_wall = time.monotonic()
        for e in self.log.events:
            if stop.is_set():
                return
            if self.speed > 0 and e.t > t0:
                while not stop.is_set():
                    delay = t0_wall + (e.t - t0) / self.speed - time.monotonic()
                    if delay <= 0:
                        break
                    time.sleep(min(delay, 0.2))
            yield e


class FollowSource(Source):
    """A recording followed as it grows (``pitsense record`` output).

    The backlog is read at once; after that new lines are picked up every ``poll_s``.
    With ``follow=False`` it stops at the end of the file. ``idle_exit_s`` ends a follow after
    that long without a new line (used by tests).
    """

    mode = "follow"

    def __init__(self, path: Path, *, follow: bool = True, poll_s: float = 0.25,
                 idle_exit_s: float | None = None, mode: str | None = None, radio=None) -> None:
        from ..live import RecordingTail, session_dir_for

        self.path, self.follow, self.poll_s, self.idle_exit_s = Path(path), follow, poll_s, idle_exit_s
        self.tail = RecordingTail(self.path, feeds=True)
        self.title = self.path.name
        self.radio_dir = session_dir_for(self.path)  # mp3s and transcripts saved next to the recording
        self.meta = {"source": str(self.path), "radio_dir": str(self.radio_dir)}
        self.speed = 1.0
        if radio is None:  # downloads missing clips and transcribes them in background threads
            from ..radio import LiveRadio

            radio = LiveRadio(self.radio_dir)
        self.radio = radio or None  # radio=False: use only what is already on disk
        if mode:
            self.mode = mode

    def events(self, stop: threading.Event):
        idle_since = time.monotonic()
        while not stop.is_set():
            new = self.tail.poll()
            if new:
                idle_since = time.monotonic()
                if self.start_utc is None and self.tail.first_recv is not None:
                    self.start_utc = datetime.fromtimestamp(self.tail.first_recv, tz=timezone.utc)
                yield from new
                continue
            if not self.follow:
                return
            if self.idle_exit_s is not None and time.monotonic() - idle_since > self.idle_exit_s:
                return
            time.sleep(self.poll_s)


class LiveSource(FollowSource):
    """Start the recorder (needs ``pitsense[live]``) and follow the file it writes."""

    mode = "live"

    def __init__(self, out: Path, *, minutes: float = 180.0, no_auth: bool = False, **kw) -> None:
        super().__init__(out, follow=True, **kw)  # makes self.radio
        self.out, self.minutes, self.no_auth = Path(out), minutes, no_auth
        self.recorder_error: str | None = None
        self._thread: threading.Thread | None = None

    def start_recorder(self) -> None:
        try:
            import fastf1  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("live recording needs the optional dependency: pip install pitsense[live]") from exc
        from ..live import record

        def run() -> None:
            try:
                record(self.out, minutes=self.minutes, no_auth=self.no_auth, radio=self.radio)
            except Exception as exc:  # shown in /api/health
                self.recorder_error = f"{type(exc).__name__}: {exc}"
                log.exception("recorder stopped")

        self._thread = threading.Thread(target=run, name="pitsense-recorder", daemon=True)
        self._thread.start()

    def events(self, stop: threading.Event):
        if self._thread is None:
            self.start_recorder()
        yield from super().events(stop)


# ----------------------------------------------------------------------------- order inference
def feed_has_positions(update: dict) -> bool:
    """Does this TimingData update carry car positions?"""
    lines = update.get("Lines") if isinstance(update, dict) else None
    return isinstance(lines, dict) and any(isinstance(v, dict) and "Position" in v for v in lines.values())


def infer_positions(state: RaceState) -> None:
    """Order cars from laps and gaps when the feed sends no positions (no-auth live feed).

    Most laps completed first, then smallest gap to the leader (a missing gap counts as 0 only
    for cars on the leader's lap count), then grid slot, then car number. Written to
    ``DriverState.position`` so every engineer sees an ordinary tower. Only the runtime does this;
    ``state.py`` stays a plain reducer of what the feed said.
    """
    cars = list(state.drivers.values())
    if not cars:
        return
    top = max(d.laps for d in cars)

    def key(d):
        gap = d.gap_to_leader
        if gap is None:
            gap = 0.0 if d.laps == top else float("inf")
        return (-d.laps, gap, d.grid if d.grid else 99, int(d.number) if d.number.isdigit() else 999)

    for i, d in enumerate(sorted(cars, key=key), 1):
        d.position = i


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


class PitWallRuntime:
    """Follows one source, keeps the latest snapshot, logs calls and alerts, fans out to listeners."""

    def __init__(self, source: Source, *, team: TeamConfig | None = None, log_dir: Path | None = None,
                 models: bool | object = True, history=None, publish_every_s: float | None = None,
                 engineers=None, track: dict | None = None) -> None:
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
        self.state = RaceState(source.meta)
        self.wall = None
        self._wall_quali = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.sim_lock = threading.RLock()  # the loop applies an event under it; /api/ask reads a still state under it
        self._voice_lock = threading.Lock()
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
        self.snap_ms: list[float] = []
        self._last_pub_t: float | None = None
        self._last_pub_lap = -1
        self._last_calls: dict[str, tuple] = {}
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
        self._wall_t0 = time.time()
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
        if self.radio_worker is not None:
            self.radio_worker.stop()
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
            for e in self.source.events(self.stop_event):
                if self.wall is None:
                    self._build_wall()
                self.handle(e)
            if self.wall is not None:
                with self.sim_lock:
                    self.publish(force=True)
            self.status = "stopped" if self.stop_event.is_set() else "finished"
        except Exception as exc:
            log.exception("pit wall loop failed")
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            if self._log_fh:
                self._log_fh.flush()
            self._notify({"event": "status", "status": self.status})

    def handle(self, e: _events.Event) -> None:
        with self.sim_lock:
            self._handle(e)

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
        lap_changed = state.current_lap != self._last_pub_lap
        due = self._last_pub_t is None or state.t - self._last_pub_t >= self.publish_every_s
        if due or lap_changed:
            self.publish()

    # --------------------------------------------------------------- publishing
    def publish(self, force: bool = False) -> dict | None:
        state = self.state
        t0 = time.perf_counter()
        try:
            snap: Snapshot = self.wall.snapshot(state)
        except Exception:
            self.snapshot_errors += 1
            if self.snapshot_errors <= 3:
                log.exception("snapshot failed at t=%.1f", state.t)
            self._last_pub_t, self._last_pub_lap = state.t, state.current_lap
            return None
        d = snap.to_dict()
        d["extra"] = self._extra(state)
        self._log_changes(state, d, snap)
        d["extra"]["radio"] = dict(self.radio)
        d["extra"].update(self._feed_extra(state))
        payload = _json_bytes(d)
        ms = (time.perf_counter() - t0) * 1000
        self.snap_ms.append(ms)
        self._last_pub_t, self._last_pub_lap = state.t, state.current_lap
        self.n_snapshots += 1
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
                "track": self.track, "team_radio": self._team_radio(True),
                "wall_msgs": list(self.wall_msgs)}

    def _maybe_push_positions(self) -> None:
        """A light "pos" event: car positions (and radio changes) between snapshots, about 3 Hz on the wall clock."""
        state, now = self.state, time.monotonic()
        if self._pos_t is not None and (state.t - self._pos_t < POS_EVERY_S or now - self._pos_wall < POS_MIN_WALL_S):
            return
        self._pos_t, self._pos_wall = state.t, now
        self.positions = pos = self._positions()
        if not pos:
            return
        msg = {"t": round(state.t, 1), "speed": self.source.speed, "cars": pos}
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
        }

    def _say(self, snap: Snapshot, car: str) -> str | None:
        """The voice's radio message for car's current call (None if the voice is off or fails)."""
        if self.voice is False:
            return None
        try:
            if self.voice is None:
                from ..voice.api import Voice

                self.voice = Voice()
            call = next((c for c in snap.calls if c.car == car), None)
            if call is None or call.action == "NO_CALL":
                return None
            from ..voice.api import say

            with self._voice_lock:
                return say(call, snap, self.voice)
        except Exception:
            log.exception("voice failed; continuing without it")
            self.voice = False
            return None

    def _log_changes(self, state: RaceState, d: dict, snap: Snapshot | None = None) -> None:
        wall = round(time.time(), 3)
        for c in d["calls"]:
            key = _call_key(c)
            prev = self._last_calls.get(c["car"])
            self._last_calls[c["car"]] = key
            if prev == key or (prev is None and c["action"] == "NO_CALL"):
                continue
            car = state.drivers.get(c["car"])
            rec = {"kind": "call", "t": c["t"], "wall": wall, "lap": state.current_lap,
                   "car_lap": (car.laps + 1) if car else None, **{k: v for k, v in c.items() if k not in ("t",)}}
            radio = self._say(snap, c["car"]) if snap is not None else None
            if radio:
                rec["radio"] = radio
                self._wall_id += 1
                self.radio[c["car"]] = {"text": radio, "lap": state.current_lap, "action": c["action"], "id": self._wall_id}
                self.wall_msgs.append({"id": self._wall_id, "kind": "voice", "car": c["car"], "t": round(state.t, 1),
                                       "lap": state.current_lap, "text": radio, "action": c["action"]})
            self._record(rec, self.calls_log)
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

    def _record(self, rec: dict, mem: deque) -> None:
        mem.append(rec)
        if self._log_fh:
            self._log_fh.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
            self._log_fh.flush()

    # --------------------------------------------------------------- listeners
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=8)
        with self.lock:
            self.listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.listeners:
                self.listeners.remove(q)

    def _notify(self, msg: dict) -> None:
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
    def health(self) -> dict:
        ms = sorted(self.snap_ms)
        now = time.time()
        return {
            "status": self.status, "error": self.error, "mode": self.source.mode,
            "title": self.source.title, "speed": self.source.speed,
            "t": round(self.state.t, 1), "lap": self.state.current_lap, "total_laps": self.state.total_laps,
            "events": self.n_events, "snapshots": self.n_snapshots,
            "last_event_age_s": None if self.last_event_wall is None else round(now - self.last_event_wall, 1),
            "uptime_s": round(now - self.started_wall, 1),
            "inferred_order": self.inferred_order, "model": self.model_info,
            "viewers": len(self.listeners),
            "observe_errors": self.observe_errors, "snapshot_errors": self.snapshot_errors,
            "snapshot_ms_p50": round(ms[len(ms) // 2], 2) if ms else None,
            "snapshot_ms_max": round(ms[-1], 2) if ms else None,
            "call_log": str(self.log_path) if self.log_path else None,
            "recorder_error": getattr(self.source, "recorder_error", None),
            "radio": None if self.radio_worker is None else dict(self.radio_worker.stats, error=self.radio_worker.error),
            "track": None if self.track is None else ("provisional" if self.track.get("provisional") else "stored"),
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
            for m in self.wall_msgs:
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
        res = None
        with self.sim_lock:
            state = self.state
            if self.wall is None or self._wall_quali or not state.drivers:
                return {"ok": False, "error": "the pit wall has no race data yet"}
            if not car or car not in state.drivers:
                focus = self.team.focus(state)
                order = [d.number for d in state.running_order() if d.running]
                car = focus[0] if focus else (order[0] if order else next(iter(state.drivers)))
            tla_of = {n: d.tla for n, d in state.drivers.items()}
            q = whatif.parse_question(text, tla_of, car)
            snap = None
            if q.kind in ("whatif", "sc"):
                res = whatif.what_if(self.wall, state, car, q)
                answer, src = res["answer"], "whatif"
            else:
                snap = self.wall.snapshot(state)
                call = next((c for c in snap.calls if c.car == car), None) or Call(snap.t, car, "NO_CALL")
                nb = {"ahead": snap.cars.get(car, {}).get("rivals__ahead"), "behind": snap.cars.get(car, {}).get("rivals__behind")}
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

    def calls(self) -> dict:
        with self.lock:
            cur = (self.latest or {}).get("calls", [])
            hist = list(self.calls_log)
        return {"current": cur, "log": hist}

    def alerts(self) -> dict:
        with self.lock:
            cur = (self.latest or {}).get("alerts", [])
            hist = list(self.alerts_log)
        return {"active": cur, "log": hist}


# ----------------------------------------------------------------------------- shadow scoring
def load_call_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a half-written last line
    return rows


def shadow_score(calls: list[dict], final: RaceState, k: int = 2, cars: set[str] | None = None) -> dict:
    """Compare logged calls with the stops that really happened.

    A BOX / PREPARE_BOX / BOX_IF_SC call from a car's lap L (``car_lap``) counts as right when that
    car's real stop (its in-lap, red-flag stops excluded) is within L-k .. L+k. A STAY_OUT call is
    right when the car has no stop in L .. L+k. Recall: share of real stops that had a box call within
    +-k laps. NO_CALL is not scored. Calls are scored as logged; later calls never rewrite earlier ones.
    """
    stops: dict[str, list[int]] = {}
    for p in final.pit_events:
        if not p.under_red:
            stops.setdefault(p.driver, []).append(p.in_lap)
    box = [c for c in calls if c.get("kind") == "call" and c.get("action") in ("BOX", "PREPARE_BOX", "BOX_IF_SC")]
    stay = [c for c in calls if c.get("kind") == "call" and c.get("action") == "STAY_OUT"]
    if cars:
        box, stay = [c for c in box if c["car"] in cars], [c for c in stay if c["car"] in cars]

    def lap_of(c) -> int:
        return int(c.get("car_lap") or c.get("lap") or 0)

    box_hit = [any(abs(s - lap_of(c)) <= k for s in stops.get(c["car"], [])) for c in box]
    stay_ok = [not any(0 <= s - lap_of(c) <= k for s in stops.get(c["car"], [])) for c in stay]
    real = [(car, s) for car, laps in stops.items() if not cars or car in cars for s in laps]
    covered = [any(c["car"] == car and abs(s - lap_of(c)) <= k for c in box) for car, s in real]

    def rate(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    by_action: dict[str, dict] = {}
    for c, ok in zip(box, box_hit):
        a = by_action.setdefault(c["action"], {"calls": 0, "right": 0})
        a["calls"] += 1
        a["right"] += ok
    return {
        "k": k, "calls_logged": len([c for c in calls if c.get("kind") == "call"]),
        "box_calls": len(box), "box_precision": rate(box_hit), "by_action": by_action,
        "stay_out_calls": len(stay), "stay_out_accuracy": rate(stay_ok),
        "real_stops": len(real), "stop_recall": rate(covered),
        "note": "NO_CALL not scored" if not box and not stay else "",
    }


# ----------------------------------------------------------------------------- CLI
def _team_config(text: str | None) -> TeamConfig:
    if not text:
        return TeamConfig()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if parts and all(p.isdigit() for p in parts):
        return TeamConfig(cars=tuple(parts))
    return TeamConfig(team=text.strip())


SESSION_NAMES = {None: None, "race": "Race", "sprint": "Sprint", "qualifying": "Qualifying",
                 "sprint-qualifying": "Sprint Qualifying", "practice1": "Practice 1",
                 "practice2": "Practice 2", "practice3": "Practice 3"}


def build_source(a) -> Source:
    if a.live:
        out = Path(a.out) if a.out else Path(os.environ.get("PITSENSE_DATA", "data")) / "live" / (
            datetime.now().strftime("%Y%m%dT%H%M%S") + "-live.jsonl")
        return LiveSource(out, minutes=a.minutes, no_auth=a.no_auth)
    if a.file and a.follow:
        return FollowSource(Path(a.file), follow=True)
    if a.file:  # a finished recording: replay it at --speed
        from ..live import load_recording

        return ReplaySource(load_recording(Path(a.file), feeds=True), a.speed, title=Path(a.file).name)
    if not a.race:
        raise SystemExit("give --race (archive replay), --file (a recording) or --live")
    from .. import archive
    ref = archive.find_session(a.year, a.race, SESSION_NAMES.get(a.session) or ("Sprint" if a.sprint else "Race"))
    if not (ref.local_dir / "TimingData.jsonStream").exists():
        archive.download_session(ref)
    return ReplaySource(load_with_feeds(ref.local_dir), a.speed, ref=ref)


def load_with_feeds(session_dir: Path) -> _events.EventLog:
    """Timing, positions and radio (not CarData: the dashboard does not use telemetry channels)."""
    files = Path(session_dir).glob("*.jsonStream")
    return _events.load_archive_session(session_dir, tuple(f.stem for f in files if f.stem != "CarData.z"))


def _lan_ip() -> str:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
            return sk.getsockname()[0]
    except OSError:
        return "<this-computer-ip>"


def cmd_pitwall(a) -> None:
    from ..config import data_dir
    from ..web.server import serve

    src = build_source(a)
    log_dir = Path(a.log_dir) if a.log_dir else data_dir() / "pitwall"
    rt = PitWallRuntime(src, team=_team_config(a.team), log_dir=log_dir, models=not a.no_models)
    rt.start()
    server = serve(rt, host=a.host, port=a.port, token=a.token)
    port = server.server_address[1]
    q = f"?token={a.token}" if a.token else ""
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else server.server_address[0]}:{port}/{q}"
    print(f"Pit wall on {url}   ({src.mode}: {src.title})   call log: {rt.log_path}")
    if a.host in ("0.0.0.0", ""):
        print(f"On the same Wi-Fi open  http://{_lan_ip()}:{port}/{q}" + ("" if a.token else "   (no --token: anyone on the network can watch and ask)"))
    if not a.no_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        while True:
            time.sleep(0.5)
            if a.no_browser and rt.status in ("finished", "error"):  # headless: exit once the replay is done
                break
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        rt.stop()


def cmd_shadow_score(a) -> None:
    from .. import archive
    from ..state import replay

    ref = archive.find_session(a.year, a.race, "Sprint" if a.sprint else "Race")
    final = replay(_events.load_archive_session(ref.local_dir))
    calls = load_call_log(Path(a.log))
    res = shadow_score(calls, final, a.k, set(a.cars) if a.cars else None)
    res["race"] = ref.slug
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(f"{ref.slug}  calls from {a.log}  (+-{a.k} laps)")
    for key, v in res.items():
        if key not in ("race", "k"):
            print(f"  {key:<18} {v}")


def add_commands(sub) -> None:
    s = sub.add_parser("pitwall", help="follow a race and show the strategy screen in the browser")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", help="archive replay, e.g. 'hungary'")
    s.add_argument("--sprint", action="store_true")
    s.add_argument("--session", choices=[k for k in SESSION_NAMES if k],
                   help="session of the meeting (default race): qualifying and sprint-qualifying get the qualifying panel")
    s.add_argument("--speed", type=float, default=1.0, help="replay speed (x real time; 0 = as fast as possible)")
    s.add_argument("--file", help="a recording from `pitsense record`")
    s.add_argument("--follow", action="store_true", help="with --file: follow it as it grows instead of replaying")
    s.add_argument("--live", action="store_true", help="start the recorder and follow its file (needs pitsense[live])")
    s.add_argument("--out", help="with --live: recording file (default data/live/<time>-live.jsonl)")
    s.add_argument("--minutes", type=float, default=180, help="with --live: stop recording after this long")
    s.add_argument("--no-auth", action="store_true", help="with --live: skip F1 TV sign-in (no positions)")
    s.add_argument("--team", help="your team, e.g. 'ferrari', or car numbers '16,44'")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 lets phones on the same Wi-Fi connect (use --token)")
    s.add_argument("--token", help="require ?token=<this> on every request (first page load sets a cookie)")
    s.add_argument("--port", type=int, default=8765, help="0 = any free port")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--no-models", action="store_true", help="don't load a trained model bundle")
    s.add_argument("--log-dir", help="where the calls log goes (default data/pitwall)")
    s.set_defaults(fn=cmd_pitwall)

    s = sub.add_parser("shadow-score", help="compare a pit wall calls log with the stops that happened")
    s.add_argument("--log", required=True, help="calls-*.jsonl written by `pitsense pitwall`")
    s.add_argument("--year", type=int, default=2026)
    s.add_argument("--race", required=True)
    s.add_argument("--sprint", action="store_true")
    s.add_argument("--k", type=int, default=2, help="a call is right if the stop is within +-k laps")
    s.add_argument("--cars", nargs="+", help="only these car numbers")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_shadow_score)
