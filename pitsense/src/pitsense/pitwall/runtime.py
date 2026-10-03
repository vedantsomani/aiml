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

from .. import events as _events  # the runtime feeds the wall; engineers never see the log
from ..state import RaceState
from .types import Snapshot, TeamConfig

log = logging.getLogger("pitsense.pitwall")

PUBLISH_EVERY_S = 3.0  # session seconds between snapshots (wall seconds at 1x)
MAX_SPEED_TICK_S = 30.0  # snapshot spacing when replaying as fast as possible
LEAD_IN_S = 60.0  # replay starts pacing this long before the session start
LOG_KEEP = 500  # calls and alerts kept in memory for the API


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
                 idle_exit_s: float | None = None, mode: str | None = None) -> None:
        from ..live import RecordingTail

        self.path, self.follow, self.poll_s, self.idle_exit_s = Path(path), follow, poll_s, idle_exit_s
        self.tail = RecordingTail(self.path)
        self.title = self.path.name
        self.meta = {"source": str(self.path)}
        self.speed = 1.0
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
        super().__init__(out, follow=True, **kw)
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
                record(self.out, minutes=self.minutes, no_auth=self.no_auth)
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
def _json_bytes(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=str).encode()


def _call_key(c: dict) -> tuple:
    return (c["car"], c["action"], c.get("compound"))


class PitWallRuntime:
    """Follows one source, keeps the latest snapshot, logs calls and alerts, fans out to listeners."""

    def __init__(self, source: Source, *, team: TeamConfig | None = None, log_dir: Path | None = None,
                 models: bool | object = True, history=None, publish_every_s: float | None = None,
                 engineers=None) -> None:
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
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
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
        ctx = Context.for_race(prior, meta, history=history if start is not None else None,
                               race_start_utc=start, team=self.team, models=models)
        self.wall = PitWall(ctx, engineers=self.engineers)

    # --------------------------------------------------------------- loop
    def start(self) -> "PitWallRuntime":
        self.thread = threading.Thread(target=self.run, name="pitsense-pitwall", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
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
        state = self.state
        state.apply(e)
        self.n_events += 1
        self.last_event_wall = time.time()
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
        self._log_changes(state, d)
        payload = _json_bytes(d)
        ms = (time.perf_counter() - t0) * 1000
        self.snap_ms.append(ms)
        self._last_pub_t, self._last_pub_lap = state.t, state.current_lap
        self.n_snapshots += 1
        with self.lock:
            self.latest, self.latest_bytes = d, payload
        self._notify({"event": "snapshot", "data": payload})
        return d

    def _extra(self, state: RaceState) -> dict:
        rc = [{"t": round(m.t, 1), "message": m.message, "category": m.category}
              for m in state.rc if "BLUE FLAG" not in m.message][-6:]
        return {
            "mode": self.source.mode, "speed": self.source.speed, "title": self.source.title,
            "inferred_order": self.inferred_order,
            "model": self.model_info,
            "weather": dict(state.weather),
            "colours": {n: d.team_colour for n, d in state.drivers.items() if d.team_colour},
            "rc": rc,
            "track_status_since": round(state.track_status_since, 1),
            "team": self.team.team, "wall_time": time.time(),
        }

    def _log_changes(self, state: RaceState, d: dict) -> None:
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
            self._record(rec, self.calls_log)
        for a in d["alerts"]:
            key = (a["engineer"], a["code"], a.get("car"), a.get("since"))
            if key in self._seen_alerts:
                continue
            self._seen_alerts.add(key)
            self._record({"kind": "alert", "wall": wall, "lap": state.current_lap, **a}, self.alerts_log)

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
            "observe_errors": self.observe_errors, "snapshot_errors": self.snapshot_errors,
            "snapshot_ms_p50": round(ms[len(ms) // 2], 2) if ms else None,
            "snapshot_ms_max": round(ms[-1], 2) if ms else None,
            "call_log": str(self.log_path) if self.log_path else None,
            "recorder_error": getattr(self.source, "recorder_error", None),
        }

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


def build_source(a) -> Source:
    if a.live:
        out = Path(a.out) if a.out else Path(os.environ.get("PITSENSE_DATA", "data")) / "live" / (
            datetime.now().strftime("%Y%m%dT%H%M%S") + "-live.jsonl")
        return LiveSource(out, minutes=a.minutes, no_auth=a.no_auth)
    if a.file and a.follow:
        return FollowSource(Path(a.file), follow=True)
    if a.file:  # a finished recording: replay it at --speed
        from ..live import load_recording

        return ReplaySource(load_recording(Path(a.file)), a.speed, title=Path(a.file).name)
    if not a.race:
        raise SystemExit("give --race (archive replay), --file (a recording) or --live")
    from .. import archive
    ref = archive.find_session(a.year, a.race, "Sprint" if a.sprint else "Race")
    if not (ref.local_dir / "TimingData.jsonStream").exists():
        archive.download_session(ref)
    return ReplaySource(_events.load_archive_session(ref.local_dir), a.speed, ref=ref)


def cmd_pitwall(a) -> None:
    from ..config import data_dir
    from ..web.server import serve

    src = build_source(a)
    log_dir = Path(a.log_dir) if a.log_dir else data_dir() / "pitwall"
    rt = PitWallRuntime(src, team=_team_config(a.team), log_dir=log_dir, models=not a.no_models)
    rt.start()
    server = serve(rt, host=a.host, port=a.port)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"Pit wall on {url}   ({src.mode}: {src.title})   call log: {rt.log_path}")
    if not a.no_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        while True:
            time.sleep(0.5)
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
    s.add_argument("--speed", type=float, default=1.0, help="replay speed (x real time; 0 = as fast as possible)")
    s.add_argument("--file", help="a recording from `pitsense record`")
    s.add_argument("--follow", action="store_true", help="with --file: follow it as it grows instead of replaying")
    s.add_argument("--live", action="store_true", help="start the recorder and follow its file (needs pitsense[live])")
    s.add_argument("--out", help="with --live: recording file (default data/live/<time>-live.jsonl)")
    s.add_argument("--minutes", type=float, default=180, help="with --live: stop recording after this long")
    s.add_argument("--no-auth", action="store_true", help="with --live: skip F1 TV sign-in (no positions)")
    s.add_argument("--team", help="your team, e.g. 'ferrari', or car numbers '16,44'")
    s.add_argument("--host", default="127.0.0.1")
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
