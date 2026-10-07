"""Where the pit wall's events come from: an archive replay at N x speed (controllable), a recording followed as it
grows, or a live session (the recorder started and its file followed). Also the running order when the feed
sends no positions (no-auth live feed)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .. import events as _events  # the feeder: reads the log; exempt by name in tests/test_pitwall.py
from ..state import RaceState
from . import replay as _replay

log = logging.getLogger("pitsense.pitwall")


LEAD_IN_S = 60.0  # replay starts pacing this long before the session start

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
    """An EventLog replayed against the wall clock at ``speed`` x (0 = as fast as possible).

    Controllable while it runs: :meth:`pause`, :meth:`resume`, :meth:`set_speed` take effect before the next
    event; a seek is a :class:`Control` the runtime loop executes (it alone writes state), after which
    :meth:`jump` moves the read position. ``pos`` is the number of events handed out.
    """

    mode = "replay"
    seekable = True

    def __init__(self, log_: _events.EventLog, speed: float = 1.0, *, ref=None, title: str = "", start_utc=None) -> None:
        self.log, self.speed, self.ref = log_, float(speed), ref
        self.meta = dict(log_.meta)
        self.title = title or (ref.slug if ref else "replay")
        self.start_utc = start_utc if start_utc is not None else (ref.start_utc if ref else None)
        self.pos = 0
        self.paused = False
        self._ctl: deque = deque()
        self._anchor: tuple[float, float] | None = None  # (wall time, session time) the pacing is measured from
        self._t_pace = log_.start  # pacing begins here (LEAD_IN_S before the lights)
        for e in log_.events:
            if e.topic == "SessionStatus" and isinstance(e.data, dict) and e.data.get("Status") == "Started":
                self._t_pace = max(log_.start, e.t - LEAD_IN_S)
                break

    # --- controls (any thread)
    def _reanchor(self) -> None:
        self._anchor = None  # the loop re-anchors at the next event

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._reanchor()

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)
        self._reanchor()

    def request(self, ctl: _replay.Control) -> _replay.Control:
        self._ctl.append(ctl)
        return ctl

    def jump(self, pos: int) -> None:
        self.pos = max(0, min(int(pos), len(self.log.events)))
        self._reanchor()

    def events(self, stop: threading.Event):
        evs = self.log.events
        while not stop.is_set():
            if self._ctl:
                yield self._ctl.popleft()
                continue
            if self.paused:
                time.sleep(0.03)
                continue
            if self.pos >= len(evs):
                return
            e = evs[self.pos]
            if self.speed > 0 and e.t > self._t_pace:
                if self._anchor is None:
                    self._anchor = (time.monotonic(), max(e.t, self._t_pace))
                aw, at = self._anchor
                delay = aw + (e.t - at) / self.speed - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 0.05))
                    continue  # re-check controls, pause and speed
            self.pos += 1
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
    """Start the recorder (needs ``pitsense[live]``) and follow the file it writes.

    Failover: ``record`` reconnects after feed drop-outs itself; if the recorder thread dies anyway (an error
    outside the connection, a crash), a watchdog restarts it, appending to the same file, after a backoff of
    5 s doubling to 60 s, until the session's time is up. Restarts and the last error are shown in /api/health.
    """

    mode = "live"
    BACKOFF_S = (5.0, 60.0)  # first wait, longest wait

    def __init__(self, out: Path, *, minutes: float = 180.0, no_auth: bool = False, **kw) -> None:
        super().__init__(out, follow=True, **kw)  # makes self.radio
        self.out, self.minutes, self.no_auth = Path(out), minutes, no_auth
        self.recorder_error: str | None = None
        self.recorder_restarts = 0
        self._thread: threading.Thread | None = None
        self._deadline: float | None = None
        self._stop: threading.Event | None = None

    def _record(self, minutes: float) -> None:  # the recorder itself (tests replace it)
        from ..live import record

        record(self.out, minutes=minutes, no_auth=self.no_auth, radio=self.radio)

    def start_recorder(self) -> None:
        if self._deadline is None:
            try:
                import fastf1  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("live recording needs the optional dependency: pip install pitsense[live]") from exc
            self._deadline = time.time() + self.minutes * 60
        left = (self._deadline - time.time()) / 60

        def run() -> None:
            try:
                self._record(left)
            except Exception as exc:  # shown in /api/health; the watchdog restarts it
                self.recorder_error = f"{type(exc).__name__}: {exc}"
                log.exception("recorder stopped")

        self._thread = threading.Thread(target=run, name="pitsense-recorder", daemon=True)
        self._thread.start()

    def _watchdog(self, stop: threading.Event) -> None:
        wait = self.BACKOFF_S[0]
        while not stop.wait(1.0):
            t = self._thread
            if t is None or t.is_alive():
                continue
            if self._deadline is None or time.time() >= self._deadline - 5:
                return  # the session's time is up: a recorder that ended is done
            log.warning("recorder is not running (%s); restarting in %.0f s", self.recorder_error or "it ended early", wait)
            if stop.wait(wait):
                return
            self.recorder_restarts += 1
            self.start_recorder()
            wait = min(wait * 2, self.BACKOFF_S[1])

    def events(self, stop: threading.Event):
        if self._thread is None:
            self.start_recorder()
            threading.Thread(target=self._watchdog, args=(stop,), name="pitsense-recorder-watchdog", daemon=True).start()
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
