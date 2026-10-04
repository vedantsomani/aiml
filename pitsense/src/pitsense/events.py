"""One time-ordered event log per session.

Archive replay and live recordings both become an :class:`EventLog`, so every
downstream module (state, features, models) runs the same code in both modes.

``Event.t`` is the session clock in seconds: the moment the message was
published. Nothing may use an event before its ``t``.
"""

from __future__ import annotations

import base64
import bisect
import gzip
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterator

from .config import FEED_TOPICS, TOPIC_PRIORITY

_TS = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


@dataclass(frozen=True)
class Event:
    t: float  # seconds on the session clock when the message was published
    topic: str
    data: Any  # JSON payload; treat as read-only
    seq: int  # deterministic tie-breaker
    kind: str = "delta"  # "delta" (merge) or "snapshot" (replace topic state)


def parse_clock(text: str) -> float:
    m = _TS.match(text)
    if not m:
        raise ValueError(f"bad timestamp: {text[:20]!r}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def parse_stream_line(line: str) -> tuple[float, Any] | None:
    """'HH:MM:SS.mmm{json}' -> (seconds, payload). Returns None for blank lines."""
    line = line.lstrip("\ufeff").rstrip("\r\n")
    if not line.strip():
        return None
    m = _TS.match(line)
    if not m:
        raise ValueError(f"bad stream line: {line[:40]!r}")
    payload = line[m.end():]
    return parse_clock(line), json.loads(payload)


def decode_z(payload: Any) -> Any:
    """Payload of a ``.z`` topic (base64 of raw-deflate JSON) -> the JSON it holds.

    Anything that is not a string (already decoded) is returned as is.
    """
    if not isinstance(payload, str):
        return payload
    return json.loads(zlib.decompress(base64.b64decode(payload), -15))


def _sort_key(topic: str, t: float, line_no: int) -> tuple:
    return (t, TOPIC_PRIORITY.get(topic, 99), topic, line_no)


@dataclass
class EventLog:
    """Immutable-by-convention list of events, sorted by (t, topic priority, seq)."""

    events: list[Event]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._times = [e.t for e in self.events]

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    @property
    def start(self) -> float:
        return self.events[0].t if self.events else 0.0

    @property
    def end(self) -> float:
        return self.events[-1].t if self.events else 0.0

    def until(self, t: float) -> "EventLog":
        """Everything published at or before ``t`` - the only past that exists at ``t``."""
        i = bisect.bisect_right(self._times, t)
        return EventLog(self.events[:i], dict(self.meta))

    def topics(self) -> set[str]:
        return {e.topic for e in self.events}

    # --- clock mapping -------------------------------------------------
    def stream_start_utc(self) -> datetime | None:
        """UTC time of session-clock zero, estimated from Heartbeat/ExtrapolatedClock."""
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        diffs = []
        for e in self.events:
            if e.topic in ("Heartbeat", "ExtrapolatedClock") and isinstance(e.data, dict):
                utc = e.data.get("Utc")
                if utc:
                    diffs.append((_parse_utc(utc) - epoch).total_seconds() - e.t)
        if not diffs:
            return None
        return epoch + timedelta(seconds=median(diffs))

    def utc_at(self, t: float) -> datetime | None:
        start = self.stream_start_utc()
        return None if start is None else start + timedelta(seconds=t)

    # --- persistence ---------------------------------------------------
    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"meta": self.meta}) + "\n")
            for e in self.events:
                f.write(json.dumps([round(e.t, 3), e.topic, e.kind, e.data], separators=(",", ":")) + "\n")

    @staticmethod
    def load_jsonl(path: Path) -> "EventLog":
        opener = gzip.open if path.suffix == ".gz" else open
        events: list[Event] = []
        meta: dict = {}
        with opener(path, "rt", encoding="utf-8") as f:
            for n, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and "meta" in row:
                    meta = row["meta"]
                    continue
                t, topic, kind, data = row
                events.append(Event(float(t), topic, data, n, kind))
        events.sort(key=lambda e: _sort_key(e.topic, e.t, e.seq))
        return EventLog(events, meta)


def _parse_utc(text: str) -> datetime:
    text = text.rstrip("Z")
    if "." in text:
        head, frac = text.split(".", 1)
        text = f"{head}.{frac[:6]}"
    dt = datetime.fromisoformat(text)
    return dt.replace(tzinfo=timezone.utc)


def load_archive_session(
    session_dir: Path, topics: tuple[str, ...] | None = None, *, feeds: bool = False
) -> EventLog:
    """Build the event log of a downloaded session from its ``*.jsonStream`` files.

    The telemetry, position and radio feeds (``config.FEED_TOPICS``) are large and not used by
    the timing reducer, so they are loaded only with ``feeds=True`` (or when named in ``topics``).
    ``.z`` payloads are decoded; each event is stamped with its message's publish time.
    """
    session_dir = Path(session_dir)
    files = sorted(session_dir.glob("*.jsonStream"))
    if topics is not None:
        files = [f for f in files if f.stem in topics]
    elif not feeds:
        files = [f for f in files if f.stem not in FEED_TOPICS]
    keyed: list[tuple[tuple, Event]] = []
    for f in files:
        topic = f.stem
        text = f.read_bytes().decode("utf-8-sig")
        for line_no, line in enumerate(text.splitlines()):
            parsed = parse_stream_line(line)
            if parsed is None:
                continue
            t, data = parsed
            if topic.endswith(".z"):
                data = decode_z(data)
            keyed.append((_sort_key(topic, t, line_no), Event(t, topic, data, line_no)))
    keyed.sort(key=lambda kv: kv[0])
    events = [Event(e.t, e.topic, e.data, i, e.kind) for i, (_, e) in enumerate(keyed)]
    meta = {}
    info = session_dir / "session.json"
    if info.exists():
        meta = json.loads(info.read_text(encoding="utf-8"))
    return EventLog(events, meta)
