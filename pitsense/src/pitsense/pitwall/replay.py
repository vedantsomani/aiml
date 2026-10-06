"""Replay lab: checkpoints, bookmarks and the control sentinel for a seekable replay.

A replay can jump to any lap and still be leak-safe because a jump never "rewinds" anything: it puts back a
saved copy of the race state *and every engineer* (``PitWall``, memory included) taken at the first event of a
lap, or, where no copy exists yet, plays the event log forward from the nearest earlier copy through the same
code path as a straight run (same events, same snapshot cadence). So a jump to lap L gives exactly the
snapshot a straight play to lap L gives, and nothing from after L is ever visible to an engineer.

Checkpoints are shared through a :class:`ReplayIndex`, which a background pass (``PitWallRuntime(index=True)``)
fills ahead of the viewer. The timeline marks (``ReplayIndex.marks``) deliberately include things that happen
later in the race; they are read by the page only, never by an engineer.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field

from ..state import RaceState

# state kept by the runtime itself (besides state and wall) that a jump must put back
RUNTIME_ATTRS = (
    "session_info", "calls_log", "alerts_log", "inferred_order", "_feed_positions", "observe_errors",
    "snapshot_errors", "_last_pub_t", "_last_pub_lap", "_last_calls", "radio", "wall_msgs", "_wall_id",
    "track", "_track_done", "_pos_t", "positions", "_radio_lap", "_radio_list", "_radio_sig", "_seen_alerts",
    "_wall_quali", "n_events", "n_snapshots", "latest", "latest_bytes", "_prev_lap", "_last_alarms",
)

TRACK_KINDS = {"4": "SC", "6": "VSC", "5": "RED"}


class Control:
    """Sentinel a ReplaySource yields so that the runtime loop (the only writer of state) runs a command."""

    def __init__(self, kind: str, **args) -> None:
        self.kind, self.args = kind, args
        self.done = threading.Event()
        self.taken = False
        self.result: dict | None = None


@dataclass
class Checkpoint:
    idx: int  # events handled
    lap: int  # leader lap at this moment
    prev_lap: int  # leader lap before the event that made it ``lap`` (so this is the first event of ``lap``)
    t: float
    payload: object = field(repr=False, default=None)  # deep copy of (state, wall, runtime attrs)


class ReplayIndex:
    """Checkpoints and bookmarks of one replay. Thread-safe; written by the player and the indexing pass."""

    def __init__(self, every: int = 1) -> None:
        self.every = max(1, int(every))
        self.lock = threading.Lock()
        self.checkpoints: dict[int, Checkpoint] = {}  # idx -> checkpoint
        self.marks: dict[str, dict] = {}
        self.indexed_idx = 0  # how far the indexing pass has got
        self.indexed_lap = 0
        self.done = False
        self.error: str | None = None
        self.n_events = 0
        self.disabled = False  # a deep copy failed: only replay-from-start is possible

    # --- checkpoints
    def add(self, ck: Checkpoint) -> None:
        with self.lock:
            self.checkpoints.setdefault(ck.idx, ck)

    def wanted(self, lap: int, prev_lap: int) -> bool:
        """Should the first event of ``lap`` get a checkpoint?"""
        if self.disabled:
            return False
        return lap % self.every == 0 or prev_lap < 0 or lap - prev_lap > 1

    def first_at_or_after(self, lap: int) -> Checkpoint | None:
        """The checkpoint that is exactly the first event with leader lap >= ``lap`` (None if not saved yet)."""
        with self.lock:
            best = None
            for ck in self.checkpoints.values():
                if ck.lap >= lap > ck.prev_lap and (best is None or ck.idx < best.idx):
                    best = ck
            return best

    def before(self, lap: int, max_idx: int | None = None) -> Checkpoint | None:
        """The latest checkpoint whose lap is below ``lap`` (a start for playing forward to it)."""
        with self.lock:
            best = None
            for ck in self.checkpoints.values():
                if ck.lap < lap and (max_idx is None or ck.idx <= max_idx) and (best is None or ck.idx > best.idx):
                    best = ck
            return best

    def laps(self) -> list[int]:
        with self.lock:
            return sorted({ck.lap for ck in self.checkpoints.values()})

    # --- marks
    def add_mark(self, m: dict) -> None:
        with self.lock:
            self.marks.setdefault(m["id"], m)

    def mark_list(self) -> list[dict]:
        with self.lock:
            return sorted(self.marks.values(), key=lambda m: (m["t"], m["id"]))


def capture(rt) -> object:
    """Deep copy of everything a jump must restore. The Context (models, history) is shared, not copied."""
    memo = {id(rt.wall.ctx): rt.wall.ctx}
    attrs = {k: getattr(rt, k) for k in RUNTIME_ATTRS}
    return copy.deepcopy((rt.state, rt.wall, attrs), memo)


def restore(rt, payload) -> None:
    memo = {id(rt.wall.ctx): rt.wall.ctx} if rt.wall is not None else {}
    state, wall, attrs = copy.deepcopy(payload, memo)
    rt.state, rt.wall = state, wall
    for k, v in attrs.items():
        setattr(rt, k, v)


# ----------------------------------------------------------------------------- bookmarks
def _mark(kind: str, lap, t, label: str, car: str | None = None, key: str = "") -> dict:
    return {"id": f"{kind}:{car or ''}:{key or round(t)}", "kind": kind, "lap": int(lap or 0), "t": round(float(t), 1),
            "car": car, "label": label}


def scan_marks(log, index: ReplayIndex) -> None:
    """Track-status changes, real pit stops and rain from the timing alone (no engineers: a few seconds)."""
    st = RaceState(dict(log.meta))
    n_pits = 0
    rain = False
    prev_status = "1"
    index.n_events = len(log.events)
    for e in log.events:
        st.apply(e)
        if st.track_status != prev_status:
            kind = TRACK_KINDS.get(st.track_status)
            if kind:
                index.add_mark(_mark(kind, st.current_lap, e.t, {"SC": "Safety car", "VSC": "Virtual safety car",
                                                              "RED": "Red flag"}[kind], key=f"{round(e.t)}"))
            prev_status = st.track_status
        if len(st.pit_events) > n_pits:
            for p in st.pit_events[n_pits:]:
                d = st.drivers.get(p.driver)
                index.add_mark(_mark("pit", p.in_lap, p.in_t, f"{d.tla if d else p.driver} pits (lap {p.in_lap})",
                                     car=p.driver, key=f"{p.in_lap}"))
            n_pits = len(st.pit_events)
        r = (st.weather.get("Rainfall") or 0) > 0
        if r != rain:
            rain = r
            if r:
                index.add_mark(_mark("rain", st.current_lap, e.t, "Rain", key=f"{round(e.t)}"))


def marks_from_record(index: ReplayIndex, rec: dict, focus: list[str], state) -> None:
    """Our calls (focus cars) and mechanic alerts, from the call/alert records the runtime writes."""
    if rec.get("kind") == "call":
        if rec.get("action") == "NO_CALL" or (focus and rec["car"] not in focus):
            return
        tla = state.drivers[rec["car"]].tla if rec["car"] in state.drivers else rec["car"]
        index.add_mark(_mark("call", rec.get("lap"), rec.get("t", state.t), f"{tla}: {rec['action']}",
                             car=rec["car"], key=f"{rec['action']}:{round(rec.get('t', state.t))}"))
    elif rec.get("kind") == "alert" and str(rec.get("engineer", "")).startswith("mechanic"):
        car = rec.get("car")
        tla = state.drivers[car].tla if car in state.drivers else car
        index.add_mark(_mark("mechanic", rec.get("lap"), state.t, f"{tla or ''} {rec.get('message') or rec.get('code')}".strip(),
                             car=car, key=f"{rec.get('code')}:{round(state.t)}"))
