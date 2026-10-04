"""Car telemetry, car positions and team radio, stored as-of.

Three feed topics are not part of the timing reducer. They reach the race state through
``state.feeds`` (a :class:`Feeds`) and never enter ``RaceState.view()`` or the fingerprint:

* ``CarData.z``  per car: RPM, speed, gear, throttle, brake, DRS
* ``Position.z`` per car: X, Y, Z on the circuit and an OnTrack / OffTrack status
* ``TeamRadio``  captures: car, UTC, path of an mp3 in the session folder

``.z`` payloads are base64 raw-deflate JSON; :func:`pitsense.events.decode_z` unpacks them.
Every message holds a batch of entries with their own (earlier) ``Utc``. All entries of a
message become available at the message's publish time ``t`` (the event time), never before:
a sample's ``t`` column is the publish time and its ``utc`` column the time it was measured.

Memory is bounded: one numpy ring buffer per car and feed (default about ten minutes).
A radio transcript counts as known only ``RADIO_LATENCY_S`` after the message (docs/feeds.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .asof import LeakageError
from .events import Event, decode_z

# Seconds between a radio message being published and its transcript being usable. Live this is
# the mp3 download (~1 s) plus Whisper on the GPU (well under 1 s per clip) plus queueing behind
# other clips; replay uses the same fixed number, so a backtest never reads a transcript earlier
# than live would.
RADIO_LATENCY_S = 15.0

CHANNELS = {"rpm": "0", "speed": "2", "gear": "3", "throttle": "4", "brake": "5", "drs": "45"}
TELEMETRY_COLS = ("t", "utc") + tuple(CHANNELS)
POSITION_COLS = ("t", "utc", "x", "y", "z", "on_track")
DEFAULT_TELEMETRY_CAP = 3000  # samples per car (~10 min at ~5 Hz)
DEFAULT_POSITION_CAP = 3000


def parse_utc(text: str) -> float:
    """ISO UTC string ('2025-08-03T12:07:40.5377207Z') -> epoch seconds."""
    text = text.rstrip("Z")
    if "." in text:
        head, frac = text.split(".", 1)
        text = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def _car_key(c: str) -> int:
    return int(c) if c.isdigit() else 999


class Ring:
    """Fixed-capacity ring buffer of float rows. The oldest rows are overwritten."""

    def __init__(self, cols: int, cap: int) -> None:
        self.buf = np.full((cap, cols), np.nan)
        self.cap = cap
        self.n = 0  # rows ever appended

    def __len__(self) -> int:
        return min(self.n, self.cap)

    def append(self, row) -> None:
        self.buf[self.n % self.cap] = row
        self.n += 1

    def rows(self) -> np.ndarray:
        """Chronological copy of what is held."""
        if self.n <= self.cap:
            return self.buf[: self.n].copy()
        i = self.n % self.cap
        return np.concatenate([self.buf[i:], self.buf[:i]])


class TelemetryStore:
    """Per-car ring buffers of CarData and Position samples, filled message by message."""

    def __init__(self, telemetry_cap: int = DEFAULT_TELEMETRY_CAP, position_cap: int = DEFAULT_POSITION_CAP) -> None:
        self.telemetry_cap = telemetry_cap
        self.position_cap = position_cap
        self._tel: dict[str, Ring] = {}
        self._pos: dict[str, Ring] = {}
        self.now = 0.0  # publish time of the newest message applied
        self.n_messages = 0

    # ------------------------------------------------------------------ writes (publish time t)
    def add_car_data(self, t: float, payload: dict) -> None:
        self.now = max(self.now, t)
        self.n_messages += 1
        for entry in sorted(payload.get("Entries") or (), key=lambda e: e.get("Utc", "")):
            utc = parse_utc(entry["Utc"])
            for car, d in (entry.get("Cars") or {}).items():
                ch = d.get("Channels") or {}
                ring = self._tel.get(car)
                if ring is None:
                    ring = self._tel[car] = Ring(len(TELEMETRY_COLS), self.telemetry_cap)
                ring.append((t, utc, *(ch.get(k, np.nan) for k in CHANNELS.values())))

    def add_position(self, t: float, payload: dict) -> None:
        self.now = max(self.now, t)
        self.n_messages += 1
        for entry in sorted(payload.get("Position") or (), key=lambda e: e.get("Timestamp", "")):
            utc = parse_utc(entry["Timestamp"])
            for car, d in (entry.get("Entries") or {}).items():
                ring = self._pos.get(car)
                if ring is None:
                    ring = self._pos[car] = Ring(len(POSITION_COLS), self.position_cap)
                ring.append((t, utc, d.get("X", np.nan), d.get("Y", np.nan), d.get("Z", np.nan),
                             1.0 if d.get("Status") == "OnTrack" else 0.0))

    # ------------------------------------------------------------------ reads
    def _check(self, t: float | None) -> float:
        if t is None:
            return self.now
        if t > self.now + 1e-9:
            raise LeakageError(f"asked for t={t:.3f} but the newest feed message is at {self.now:.3f}")
        return t

    @staticmethod
    def _window(ring: Ring | None, cols: tuple[str, ...], t: float, last_s: float | None) -> dict[str, np.ndarray]:
        rows = np.empty((0, len(cols))) if ring is None else ring.rows()
        rows = rows[rows[:, 0] <= t]  # published by t
        if last_s is not None and len(rows):
            rows = rows[rows[:, 1] >= rows[-1, 1] - last_s]  # last_s of the car's own measured time
        return {c: rows[:, i] for i, c in enumerate(cols)}

    def cars(self) -> list[str]:
        return sorted(set(self._tel) | set(self._pos), key=_car_key)

    def telemetry(self, car: str, last_s: float | None = 30.0, t: float | None = None) -> dict[str, np.ndarray]:
        """Samples of one car from the last ``last_s`` seconds of its measured time (None: all held).

        Columns: t (published), utc (measured, epoch s), rpm, speed (km/h), gear, throttle (%),
        brake (0 / 100+), drs (raw code; >= 10 is open). Oldest first.
        """
        return self._window(self._tel.get(str(car)), TELEMETRY_COLS, self._check(t), last_s)

    def position_history(self, car: str, last_s: float | None = 30.0, t: float | None = None) -> dict[str, np.ndarray]:
        """Position samples of one car (columns t, utc, x, y, z, on_track), oldest first."""
        return self._window(self._pos.get(str(car)), POSITION_COLS, self._check(t), last_s)

    def latest_position(self, t: float | None = None) -> dict[str, dict[str, float]]:
        """Newest published position of every car: {car: {x, y, z, on_track, t, utc}}."""
        t = self._check(t)
        out = {}
        for car, ring in self._pos.items():
            h = self._window(ring, POSITION_COLS, t, 0.0)
            if len(h["t"]):
                out[car] = {c: float(h[c][-1]) for c in POSITION_COLS} | {"on_track": bool(h["on_track"][-1])}
        return dict(sorted(out.items(), key=lambda kv: _car_key(kv[0])))

    def latest_telemetry(self, t: float | None = None) -> dict[str, dict[str, float]]:
        """Newest published CarData sample of every car."""
        t = self._check(t)
        out = {}
        for car, ring in self._tel.items():
            h = self._window(ring, TELEMETRY_COLS, t, 0.0)
            if len(h["t"]):
                out[car] = {c: float(h[c][-1]) for c in TELEMETRY_COLS}
        return dict(sorted(out.items(), key=lambda kv: _car_key(kv[0])))

    def nbytes(self) -> int:
        return sum(r.buf.nbytes for r in (*self._tel.values(), *self._pos.values()))


@dataclass(frozen=True)
class RadioMessage:
    car: str
    t: float  # publish time (session clock)
    utc: float | None  # capture time, epoch s
    path: str  # as published, e.g. TeamRadio/MAXVER01_1_20250803_150810.mp3
    audio: str | None  # local file, if the session folder is known
    text: str | None  # None until the transcript is known (t + latency) or if none exists
    known_at: float  # t + latency: when the transcript counts as available

    def to_dict(self) -> dict:
        return {"car": self.car, "t": self.t, "utc": self.utc, "path": self.path, "audio": self.audio,
                "text": self.text, "known_at": self.known_at}


class TranscriptCache:
    """Transcripts cached as JSON next to each mp3: ``X.mp3`` -> ``X.json``."""

    def __init__(self, session_dir: Path | None) -> None:
        self.dir = None if session_dir is None else Path(session_dir)
        self._text: dict[str, str] = {}  # transcripts found so far (a file, once written, never changes)

    def audio_path(self, path: str) -> Path | None:
        return None if self.dir is None else self.dir / path

    def get(self, path: str) -> str | None:
        audio = self.audio_path(path)
        if audio is None:
            return None
        if path in self._text:
            return self._text[path]
        f = audio.with_suffix(".json")
        if not f.exists():
            return None
        try:
            text = json.loads(f.read_text(encoding="utf-8")).get("text")
        except (OSError, ValueError):
            return None
        if text is not None:
            self._text[path] = text
        return text


class RadioStore:
    """Team radio captures in publish order. Transcripts appear ``latency_s`` seconds later."""

    def __init__(self, session_dir: Path | None = None, latency_s: float = RADIO_LATENCY_S, keep: int = 2000) -> None:
        self.latency_s = latency_s
        self.keep = keep
        self.transcripts = TranscriptCache(session_dir)
        self._items: list[tuple[str, float, float | None, str]] = []  # car, t, utc, path
        self._seen: set[str] = set()
        self.now = 0.0

    def add(self, t: float, data: dict) -> None:
        self.now = max(self.now, t)
        caps = (data or {}).get("Captures") or ()
        if isinstance(caps, dict):
            caps = [caps[k] for k in sorted(caps, key=lambda k: int(k) if str(k).isdigit() else 0)]
        for c in caps:
            path = c.get("Path")
            if not path or path in self._seen:
                continue
            self._seen.add(path)
            utc = c.get("Utc")
            self._items.append((str(c.get("RacingNumber", "")), t, parse_utc(utc) if utc else None, path))
        if len(self._items) > self.keep:
            del self._items[: len(self._items) - self.keep]

    def messages(self, car: str | None = None, t: float | None = None, last_s: float | None = None) -> list[RadioMessage]:
        """Radio messages published up to ``t`` (default: now), oldest first.

        ``text`` is filled only once ``t >= message time + latency``.
        """
        if t is None:
            t = self.now
        elif t > self.now + 1e-9:
            raise LeakageError(f"asked for t={t:.3f} but the newest radio message is at {self.now:.3f}")
        out = []
        for c, mt, utc, path in self._items:
            if mt > t or (car is not None and c != str(car)) or (last_s is not None and mt < t - last_s):
                continue
            known = mt + self.latency_s
            audio = self.transcripts.audio_path(path)
            out.append(RadioMessage(c, mt, utc, path, None if audio is None else str(audio),
                                    self.transcripts.get(path) if t >= known else None, known))
        return out


class Feeds:
    """Everything the race state keeps outside the timing reducer. Created empty with the state."""

    def __init__(self, meta: dict | None = None, *, latency_s: float = RADIO_LATENCY_S) -> None:
        self.telemetry = TelemetryStore()
        session_dir = None
        path = (meta or {}).get("path")
        if (meta or {}).get("radio_dir"):  # live / followed recording: a folder next to the file
            session_dir = Path(meta["radio_dir"])
            latency_s = 0.0  # a transcript counts when its file exists: live latency is real latency
        elif path:
            from .config import raw_dir

            session_dir = raw_dir() / str(path).rstrip("/")
        self.radio = RadioStore(session_dir, latency_s)

    def advance(self, t: float) -> None:
        """Move both stores' clocks to the race state's time (any event), so default queries mean 'now'."""
        if t > self.telemetry.now:
            self.telemetry.now = t
        if t > self.radio.now:
            self.radio.now = t

    def apply(self, e: Event) -> None:
        data: Any = e.data
        if isinstance(data, str):
            data = decode_z(data)
        if not isinstance(data, dict):
            return
        if e.topic == "CarData.z":
            self.telemetry.add_car_data(e.t, data)
        elif e.topic == "Position.z":
            self.telemetry.add_position(e.t, data)
        elif e.topic == "TeamRadio":
            self.radio.add(e.t, data)
