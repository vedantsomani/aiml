"""Record a live session and turn the recording into an EventLog.

The recorder reuses FastF1's SignalR client (connection + F1 TV sign-in) but
writes our own JSONL: one line per message with the local receive time. That
receive time is what PitSense treats as "available_at" - the honest moment we
knew something.

    pitsense record --out data/live/2026-sepang-race.jsonl

Requires `pip install pitsense[live]`. Without an F1 TV login (`--no-auth`) the
feed omits car positions and pit-stop times since the 2025 Dutch GP; lap times,
gaps, tyres, track status and race control still arrive.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .events import Event, EventLog

TOPICS = [
    "Heartbeat", "ExtrapolatedClock", "SessionInfo", "SessionStatus", "SessionData", "DriverList",
    "LapCount", "TrackStatus", "TimingData", "TimingAppData", "TimingStats", "RaceControlMessages",
    "WeatherData", "PitStopSeries", "PitLaneTimeCollection", "PitStop", "TyreStintSeries",
    "TopThree", "TeamRadio", "CarData.z", "Position.z",
]


def _make_client(out: Path, *, no_auth: bool, timeout: int, deadline: float | None = None):
    from fastf1.livetiming.client import SignalRClient  # optional dependency
    from signalrcore.messages.completion_message import CompletionMessage

    class JsonlClient(SignalRClient):
        def __init__(self) -> None:
            super().__init__(str(out), filemode="a", timeout=timeout, no_auth=no_auth)
            self.topics = list(TOPICS)
            self.n_messages = 0
            self.interrupted = False
            self.deadline = deadline

        def _supervise(self) -> None:
            # FastF1's version only stops when the feed goes quiet; heartbeats keep it alive
            # forever, so also stop at the --minutes deadline. Print a status line every minute.
            self._t_last_message = time.time()
            next_status = time.time() + 60
            while True:
                now = time.time()
                if self.deadline is not None and now > self.deadline:
                    self.logger.info("Time limit reached; stopping.")
                    self._exit()
                    return
                if self.timeout and now - self._t_last_message > self.timeout:
                    self.logger.warning(f"No data for {self.timeout} s.")
                    self._exit()
                    return
                if now >= next_status:
                    age = now - self._t_last_message
                    self.logger.info(f"recording: {self.n_messages} messages, last one {age:.0f} s ago")
                    next_status = now + 60
                time.sleep(1)

        def _run(self) -> None:
            # Same as FastF1's _run, except: --no-auth omits the token factory (FastF1 passes
            # None, which signalrcore >= 1.0 rejects) and connecting gives up after 30 s.
            import requests
            from fastf1.internals.f1auth import get_auth_token
            from signalrcore.hub_connection_builder import HubConnectionBuilder

            self._output_file = open(self.filename, self.filemode, encoding="utf-8")
            r = requests.options(self._negotiate_url, headers=self.headers, timeout=30)
            if "AWSALBCORS" in r.cookies:
                self.headers.update({"Cookie": f"AWSALBCORS={r.cookies['AWSALBCORS']}"})
            options = {"verify_ssl": True, "headers": self.headers}
            if not self._no_auth:
                options["access_token_factory"] = get_auth_token
            self._connection = (
                HubConnectionBuilder().with_url(self._connection_url, options=options)
                .configure_logging(logging.WARNING).build()
            )
            self._connection.on_open(self._on_connect)
            self._connection.on_close(self._on_close)
            self._connection.on("feed", self._on_message)
            self._connection.start()
            t0 = time.time()
            while not self._is_connected:
                if time.time() - t0 > 30:
                    raise ConnectionError("no connection to the live timing feed after 30 s")
                time.sleep(0.1)
            self._connection.send("Subscribe", [self.topics], on_invocation=self._on_message)

        def start(self) -> None:
            # FastF1 swallows Ctrl+C inside start(); we need to know so we stop reconnecting
            self._run()
            try:
                self._supervise()
            except KeyboardInterrupt:
                self.interrupted = True
                self.logger.info("Stopping recording...")
                self._exit()

        def _write(self, row: dict) -> None:
            self._output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._output_file.flush()
            self.n_messages += 1

        def _on_message(self, msg) -> None:  # noqa: D401 - FastF1 callback
            self._t_last_message = time.time()
            recv = time.time()
            try:
                if isinstance(msg, CompletionMessage):
                    for topic, data in (msg.result or {}).items():
                        self._write({"recv": recv, "topic": topic, "kind": "snapshot", "data": data})
                elif isinstance(msg, list) and len(msg) >= 2:
                    self._write({"recv": recv, "topic": msg[0], "kind": "delta", "data": msg[1],
                                 "utc": msg[2] if len(msg) > 2 else None})
            except Exception:  # never let a bad message kill the recording
                self.logger.exception("could not write message")

    return JsonlClient()


def _cleanup(client) -> None:
    """After a failed attempt: stop a half-open connection and close the output file.

    FastF1's ``_exit`` does this on a clean stop, but not when connecting raises.
    """
    conn = getattr(client, "_connection", None)
    if conn is not None:
        try:
            conn.stop()
        except Exception:
            pass
    f = getattr(client, "_output_file", None)
    if f is not None and not f.closed:
        f.close()


def record(out: Path, *, minutes: float = 180.0, no_auth: bool = False, idle_timeout: int = 120) -> int:
    """Record until ``minutes`` have passed, reconnecting after drop-outs. Returns message count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + minutes * 60
    total = 0
    log = logging.getLogger("pitsense.record")
    while time.time() < deadline:
        client = _make_client(out, no_auth=no_auth, timeout=idle_timeout, deadline=deadline)
        try:
            client.start()  # blocks until idle timeout or Ctrl+C
        except KeyboardInterrupt:
            _cleanup(client)
            total += client.n_messages
            break
        except Exception as exc:  # connection failures: clean up, wait and retry
            _cleanup(client)
            log.warning("connection problem: %s; retrying in 10 s", exc)
            time.sleep(10)
        total += client.n_messages
        if client.interrupted:
            break
        if time.time() < deadline:
            log.info("feed went quiet or dropped; reconnecting (%d messages so far)", total)
            time.sleep(5)
    return total


def load_fastf1_recording(path: Path, meta: dict | None = None) -> EventLog:
    """Fallback: a file saved with FastF1's own recorder (`python -m fastf1.livetiming save f.txt`).

    Lines are Python-literal lists: [topic, data, utc]. The initial subscribe
    response has the data as a JSON string and an empty timestamp.
    """
    import ast
    from datetime import datetime

    def utc_seconds(text: str) -> float | None:
        if not text:
            return None
        text = text.rstrip("Z")
        if "." in text:
            head, frac = text.split(".", 1)
            text = f"{head}.{frac[:6]}"
        return datetime.fromisoformat(text).timestamp()

    parsed = []
    # FastF1 writes with the system encoding (cp1252 on Windows); don't die on accents
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                topic, data, ts = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            kind = "delta"
            if isinstance(data, str):
                data, kind = json.loads(data), "snapshot"
            parsed.append((utc_seconds(ts), topic, data, i, kind))
    known = [p[0] for p in parsed if p[0] is not None]
    if not known:
        return EventLog([], dict(meta or {}))
    t0 = min(known)
    events, last_t = [], 0.0
    for ts, topic, data, i, kind in parsed:
        if topic.endswith(".z"):
            continue
        t = last_t if ts is None else ts - t0  # snapshots inherit the preceding time
        last_t = max(last_t, t)
        events.append(Event(round(t, 3), topic, data, i, kind))
    events.sort(key=lambda e: (e.t, e.seq))
    return EventLog(events, dict(meta or {}, source=str(path)))


def load_recording(path: Path, meta: dict | None = None) -> EventLog:
    """Recorded JSONL -> EventLog. ``t`` = seconds since the first message was received."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return EventLog([], dict(meta or {}))
    t0 = min(r["recv"] for r in rows)
    events = []
    for i, r in enumerate(rows):
        topic = r["topic"]
        if topic.endswith(".z"):
            continue  # compressed telemetry; not used by the reducer yet
        events.append(Event(round(r["recv"] - t0, 3), topic, r["data"], i, r.get("kind", "delta")))
    # receive order is the truth for availability; keep it stable
    events.sort(key=lambda e: (e.t, e.seq))
    return EventLog(events, dict(meta or {}, source=str(path)))


class RecordingTail:
    """Follow a recording that is still being written.

    :meth:`poll` returns the events completed since the last call, in file order. A last line
    that is still being written is kept back until its newline arrives; a broken complete line
    is skipped (and counted in ``bad_lines``). ``t`` is seconds since the first message received,
    exactly as :func:`load_recording` computes it, so following and replaying a file agree.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.t0: float | None = None
        self.first_recv: float | None = None
        self.bad_lines = 0
        self._offset = 0
        self._partial = b""
        self._seq = 0

    def poll(self) -> list[Event]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []  # not created yet
        if size < self._offset:  # truncated or replaced: start over
            self._offset, self._partial = 0, b""
        if size == self._offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        self._offset += len(chunk)
        data = self._partial + chunk
        *lines, self._partial = data.split(b"\n")
        events: list[Event] = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                recv, topic = float(row["recv"]), row["topic"]
            except (ValueError, KeyError, TypeError):
                self.bad_lines += 1
                continue
            seq, self._seq = self._seq, self._seq + 1
            if self.t0 is None:
                self.t0 = self.first_recv = recv
            if topic.endswith(".z"):
                continue  # compressed telemetry; not used by the reducer
            events.append(Event(round(max(recv - self.t0, 0.0), 3), topic, row.get("data"), seq,
                                row.get("kind", "delta")))
        return events
