"""Deterministic reducer: events -> race state.

``RaceState.apply(event)`` is the only way state changes. Feed it an archive
replay or a live recording and you get the same states at the same times.

Besides the current timing tower it keeps three append-only records, each row
stamped with the time it became known:
    laps       one row per completed lap (``t_end``; lap time at ``lap_time_t``)
    pit_stops  one row per stop from the official pit-stop topics
    rc         race-control messages
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import DRY_COMPOUNDS, NOMINATIONS, TRACK_STATUS
from .events import Event, EventLog
from .merge import clone, deep_merge

_LAPS_DOWN = re.compile(r"^\+?(\d+)\s*L(?:AP)?S?$", re.IGNORECASE)


def relative_compound(compound: str, meta: dict) -> str:
    """A compound as SOFT/MEDIUM/HARD: 2018's Pirelli names via the weekend's nominations.

    The rank among the three compounds nominated before the race gives the name; the
    tyres used during the race never do. Other names and later seasons pass through.
    """
    nominated = NOMINATIONS.get((meta.get("year"), meta.get("meeting_name")))
    if nominated and compound in nominated:
        return DRY_COMPOUNDS[nominated.index(compound)]
    return compound


# --------------------------------------------------------------------------- parsing helpers
def parse_lap_time(text: Any) -> float | None:
    """'1:26.103' -> 86.103; '' -> None."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        if ":" in text:
            m, s = text.split(":", 1)
            return round(int(m) * 60 + float(s), 3)
        return round(float(text), 3)
    except ValueError:
        return None


def parse_gap(text: Any) -> tuple[float | None, int]:
    """Gap/interval string -> (seconds or None, laps_down).

    'LAP 23' (leader) -> (0.0, 0); '+1.234' -> (1.234, 0); '1L' -> (None, 1); '' -> (None, 0)
    """
    if not isinstance(text, str):
        return None, 0
    s = text.strip()
    if not s:
        return None, 0
    if s.upper().startswith("LAP"):
        return 0.0, 0
    m = _LAPS_DOWN.match(s)
    if m:
        return None, int(m.group(1))
    try:
        return float(s.lstrip("+")), 0
    except ValueError:
        return None, 0


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- records
@dataclass
class DriverState:
    number: str
    tla: str = ""
    team: str = ""
    team_colour: str = ""
    grid: int | None = None
    position: int | None = None
    gap_to_leader: float | None = None  # seconds; 0.0 for the leader
    laps_down: int = 0
    interval: float | None = None  # seconds to the car ahead
    laps: int = 0  # completed laps
    last_lap_time: float | None = None
    best_lap_time: float | None = None
    in_pit: bool = False
    pit_out: bool = False
    pit_stops: int = 0
    retired: bool = False
    stopped: bool = False
    compound: str | None = None
    tyre_new: bool | None = None
    tyre_age: int | None = None  # laps on the current set incl. earlier sessions (FastF1 'TyreLife')
    stint: int = 0  # number of tyre sets fitted so far (1 = starting set)
    stint_index: int = -1  # index of the current record in the feed's stint list
    stint_start_lap: int = 0  # race lap after which this set was fitted
    stint_start_age: int = 0  # laps already on the set when fitted
    compounds_used: list[str] = field(default_factory=list)
    last_in_lap: int | None = None
    last_out_lap: int | None = None
    pit_in_t: float | None = None
    pit_out_t: float | None = None

    @property
    def running(self) -> bool:
        return not (self.retired or self.stopped)


@dataclass
class LapRecord:
    driver: str
    lap: int
    t_start: float | None
    t_end: float  # when the line crossing was published
    lap_time: float | None
    lap_time_t: float | None  # when the lap time was published
    position: int | None
    gap_to_leader: float | None
    laps_down: int
    interval: float | None
    compound: str | None
    tyre_age: int | None
    stint: int
    is_in_lap: bool
    is_out_lap: bool
    track_status: str  # statuses seen during the lap, e.g. "1" or "1,4"
    lap_time_inferred: bool = False  # value omitted by the feed because it repeated


@dataclass
class PitStopRecord:
    driver: str
    lap: int | None
    lane_time: float | None  # pit entry -> pit exit, seconds
    stop_time: float | None  # stationary time, seconds (2024 US GP onwards)
    t: float  # when it was published
    source: str


@dataclass
class PitEvent:
    """A pit entry seen on the timing feed (NumberOfPitStops went up)."""

    driver: str
    in_lap: int  # the lap that ended in the pit lane
    in_t: float
    out_lap: int | None = None  # the lap that started from the pit lane
    out_t: float | None = None
    status_at_entry: str = "1"  # track status when the car entered (5 = red flag)

    @property
    def under_red(self) -> bool:
        return self.status_at_entry == "5"

    @property
    def lane_time(self) -> float | None:
        return None if self.out_t is None else self.out_t - self.in_t


@dataclass
class RCMessage:
    t: float
    category: str
    flag: str
    scope: str
    message: str
    lap: int | None
    driver: str | None


# --------------------------------------------------------------------------- reducer
class RaceState:
    """Current state of a session. Mutated only through :meth:`apply`."""

    def __init__(self, meta: dict | None = None) -> None:
        self.meta = dict(meta or {})
        self.t: float = 0.0
        self.topics: dict[str, Any] = {}
        self.drivers: dict[str, DriverState] = {}
        self.current_lap: int = 0
        self.total_laps: int | None = None
        self.track_status: str = "1"
        self.track_status_since: float = 0.0
        self.status_log: list[tuple[float, str]] = []
        self.session_status: str | None = None
        self.started_t: float | None = None
        self.finished_t: float | None = None
        self.weather: dict[str, float | None] = {}
        self.rc: list[RCMessage] = []
        self.laps: list[LapRecord] = []
        self.pit_stops: list[PitStopRecord] = []
        self.pit_events: list[PitEvent] = []
        self.new_laps: list[LapRecord] = []  # laps completed by the last applied event
        self._lap_status: dict[str, set[str]] = {}
        self._lap_start: dict[str, float] = {}
        self._pending_time: dict[str, LapRecord] = {}
        self._pit_series_count: dict[str, int] = {}
        self._plt_seen: set[tuple[str, str]] = set()
        self._has_pit_series = False

    # ------------------------------------------------------------------ public
    @property
    def started(self) -> bool:
        return self.started_t is not None

    def driver(self, number: str) -> DriverState:
        d = self.drivers.get(number)
        if d is None:
            d = DriverState(number=number)
            self.drivers[number] = d
            self._lap_status[number] = {self.track_status}
        return d

    def apply(self, e: Event) -> None:
        self.t = e.t
        self.new_laps = []
        if e.kind == "snapshot":
            self.topics[e.topic] = clone(e.data)
        else:
            self.topics[e.topic] = deep_merge(self.topics.get(e.topic), e.data)
        handler = getattr(self, f"_on_{e.topic.replace('.', '_')}", None)
        if handler is not None and isinstance(e.data, dict):
            handler(e.data, e)

    def running_order(self) -> list[DriverState]:
        """Cars with a position, sorted by position."""
        cars = [d for d in self.drivers.values() if d.position is not None]
        return sorted(cars, key=lambda d: d.position)

    def view(self) -> dict:
        """JSON-serialisable snapshot of everything known at ``self.t``."""
        order = self.running_order()
        behind: dict[str, float | None] = {}
        for ahead, back in zip(order, order[1:]):
            behind[ahead.number] = back.interval
        return {
            "t": round(self.t, 3),
            "lap": self.current_lap,
            "total_laps": self.total_laps,
            "track_status": self.track_status,
            "track_status_since": round(self.track_status_since, 3),
            "session_status": self.session_status,
            "weather": dict(sorted(self.weather.items())),
            "drivers": {
                n: {**asdict(d), "gap_behind": behind.get(n)}
                for n, d in sorted(self.drivers.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999)
            },
            "n_laps_recorded": len(self.laps),
            "n_pit_stops": len(self.pit_stops),
            "n_rc": len(self.rc),
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.view(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    # ------------------------------------------------------------------ topic handlers
    def _on_SessionStatus(self, data: dict, e: Event) -> None:
        status = data.get("Status")
        if status:
            self.session_status = status
            if status == "Started" and self.started_t is None:
                self.started_t = e.t
                for n, d in self.drivers.items():
                    self._lap_start[n] = e.t
                    if d.in_pit:  # starting from the pit lane: lap 1 is an out-lap
                        d.last_out_lap = 1
            if status in ("Finished", "Finalised") and self.finished_t is None:
                self.finished_t = e.t

    def _on_DriverList(self, data: dict, e: Event) -> None:
        for num, upd in data.items():
            if not isinstance(upd, dict) or not num.isdigit():
                continue
            d = self.driver(num)
            full = self.topics["DriverList"].get(num, {})
            d.tla = full.get("Tla", d.tla) or d.tla
            d.team = full.get("TeamName", d.team) or d.team
            d.team_colour = full.get("TeamColour", d.team_colour) or d.team_colour

    def _on_LapCount(self, data: dict, e: Event) -> None:
        if "CurrentLap" in data:
            self.current_lap = _to_int(data["CurrentLap"]) or self.current_lap
        if "TotalLaps" in data:
            self.total_laps = _to_int(data["TotalLaps"]) or self.total_laps

    def _on_TrackStatus(self, data: dict, e: Event) -> None:
        code = str(data.get("Status", self.track_status))
        if code != self.track_status:
            self.track_status = code
            self.track_status_since = e.t
            self.status_log.append((e.t, code))
            for s in self._lap_status.values():
                s.add(code)

    def _on_WeatherData(self, data: dict, e: Event) -> None:
        for key, value in data.items():
            self.weather[key] = _to_float(value)

    def _on_RaceControlMessages(self, data: dict, e: Event) -> None:
        msgs = data.get("Messages")
        if isinstance(msgs, list):
            items = msgs
        elif isinstance(msgs, dict):
            items = [v for k, v in sorted(msgs.items(), key=lambda kv: _to_int(kv[0]) or 0)]
        else:
            return
        for m in items:
            if not isinstance(m, dict):
                continue
            self.rc.append(
                RCMessage(
                    t=e.t,
                    category=str(m.get("Category", "")),
                    flag=str(m.get("Flag", "")),
                    scope=str(m.get("Scope", "")),
                    message=str(m.get("Message", "")),
                    lap=_to_int(m.get("Lap")),
                    driver=str(m["RacingNumber"]) if m.get("RacingNumber") else None,
                )
            )

    def _on_TimingAppData(self, data: dict, e: Event) -> None:
        lines = data.get("Lines")
        if not isinstance(lines, dict):
            return
        full_lines = self.topics["TimingAppData"].get("Lines", {})
        for num, upd in lines.items():
            if not isinstance(upd, dict):
                continue
            d = self.driver(num)
            if "GridPos" in upd:
                d.grid = _to_int(upd["GridPos"])
            if "Stints" not in upd:
                continue
            stints = (full_lines.get(num) or {}).get("Stints") or []
            if isinstance(stints, dict):  # defensive: should be a list after merge
                stints = [stints[k] for k in sorted(stints, key=lambda k: _to_int(k) or 0)]
            real = self._real_stints(stints)
            if not real:
                continue
            idx = real[-1]
            cur = stints[idx]
            if idx != d.stint_index:
                d.stint_index = idx
                d.stint_start_lap = self._stint_start_lap(d, idx)
            d.stint = len(real)
            start_age = _to_int(cur.get("StartLaps"))
            if start_age is not None:
                d.stint_start_age = start_age
            # every set fitted so far, not just the current one: when several records
            # arrive in one message (2018 publishes the first ones minutes after the
            # start, after any lap-1 stops), the earlier sets are never current
            for i in real:
                used = str(stints[i].get("Compound", "")).upper()
                if used and used not in ("UNKNOWN", "TEST_UNKNOWN"):
                    used = relative_compound(used, self.meta)
                    if used not in d.compounds_used:
                        d.compounds_used.append(used)
            compound = str(cur.get("Compound", "")).upper() or None
            if compound and compound not in ("UNKNOWN", "TEST_UNKNOWN"):
                d.compound = relative_compound(compound, self.meta)
            new = cur.get("New")
            d.tyre_new = None if new is None else str(new).lower() == "true"
            self._update_tyre_age(d)

    @staticmethod
    def _real_stints(stints: list) -> list[int]:
        """Indices of stint records that mean tyres were actually fitted.

        The feed adds a record at every stop (and for pit-lane starts) flagged
        TyresNotChanged=1 and may only correct the flag laps later. A record whose
        compound differs from the previous fitted set is a tyre change regardless.
        """
        out: list[int] = []
        prev_compound = None
        for i, s in enumerate(stints):
            if not isinstance(s, dict) or not s.get("Compound"):
                continue
            compound = str(s["Compound"]).upper()
            flagged_same = str(s.get("TyresNotChanged", "0")) == "1"
            changed = compound not in ("UNKNOWN", "TEST_UNKNOWN") and prev_compound is not None and compound != prev_compound
            if not flagged_same or changed or not out:
                out.append(i)
                prev_compound = compound if compound not in ("UNKNOWN", "TEST_UNKNOWN") else prev_compound
        return out

    def _stint_start_lap(self, d: DriverState, record_index: int) -> int:
        """Race lap after which the current set was fitted."""
        if not self.started or record_index == 0:
            return 0
        # every recorded pit event is already in the past (events apply in time order)
        stops = [p for p in self.pit_events if p.driver == d.number]
        if not stops:
            return d.laps
        # stint record k belongs to the k-th stop; with fewer stops (pit-lane start
        # placeholders shift the numbering) it belongs to the latest stop so far
        if len(stops) >= record_index:
            return stops[record_index - 1].in_lap
        return stops[-1].in_lap

    def _on_TimingData(self, data: dict, e: Event) -> None:
        lines = data.get("Lines")
        if not isinstance(lines, dict):
            return
        for num, upd in lines.items():
            if not isinstance(upd, dict):
                continue
            d = self.driver(num)
            if "Position" in upd:
                d.position = _to_int(upd["Position"])
            if "GapToLeader" in upd:
                d.gap_to_leader, d.laps_down = parse_gap(upd["GapToLeader"])
            iv = upd.get("IntervalToPositionAhead")
            if isinstance(iv, dict) and "Value" in iv:
                value, _ = parse_gap(iv["Value"])
                d.interval = None if (isinstance(iv["Value"], str) and iv["Value"].upper().startswith("LAP")) else value
            if "Retired" in upd:
                d.retired = bool(upd["Retired"])
            if "Stopped" in upd:
                d.stopped = bool(upd["Stopped"])
            if "InPit" in upd:
                now_in = bool(upd["InPit"])
                if now_in and not d.in_pit and self.started:
                    d.pit_in_t = e.t
                d.in_pit = now_in
            if "NumberOfPitStops" in upd:
                n = _to_int(upd["NumberOfPitStops"]) or 0
                if n > d.pit_stops and self.started:
                    d.last_in_lap = d.laps + 1
                    self.pit_events.append(
                        PitEvent(num, d.last_in_lap, d.pit_in_t if d.in_pit and d.pit_in_t else e.t,
                                 status_at_entry=self.track_status)
                    )
                d.pit_stops = n
            if "PitOut" in upd:
                now_out = bool(upd["PitOut"])
                if now_out and not d.pit_out and self.started:
                    d.pit_out_t = e.t
                    d.last_out_lap = d.laps + 1
                    for pe in reversed(self.pit_events):
                        if pe.driver == num:
                            if pe.out_t is None:
                                pe.out_t, pe.out_lap = e.t, d.last_out_lap
                            break
                d.pit_out = now_out
            llt = upd.get("LastLapTime")
            has_time = isinstance(llt, dict) and parse_lap_time(llt.get("Value")) is not None
            if "NumberOfLaps" in upd:
                n = _to_int(upd["NumberOfLaps"])
                if n is not None and n > d.laps:
                    for lap in range(d.laps + 1, n + 1):
                        d.laps = lap
                        self._update_tyre_age(d)
                        rec = self._complete_lap(d, lap, e.t)
                        if not has_time and lap > 1 and d.last_lap_time is not None and lap == n:
                            # The feed only sends changed values: a lap time equal to the
                            # previous one is omitted. Assume unchanged; a later value overrides.
                            rec.lap_time, rec.lap_time_t, rec.lap_time_inferred = d.last_lap_time, e.t, True
            if has_time:
                value = parse_lap_time(llt["Value"])
                d.last_lap_time = value
                pending = self._pending_time.get(num)
                # A value sent apart from the lap count completes the pending lap, unless
                # more than half of it has passed since that lap ended: then it is the next
                # lap's time, sent just before its count (seen in 2018, never in 2025-26).
                if (pending is not None and (pending.lap_time is None or pending.lap_time_inferred)
                        and e.t - pending.t_end <= value / 2):
                    pending.lap_time, pending.lap_time_t, pending.lap_time_inferred = value, e.t, False
            blt = upd.get("BestLapTime")
            if isinstance(blt, dict) and "Value" in blt:
                value = parse_lap_time(blt["Value"])
                if value is not None:
                    d.best_lap_time = value

    def _update_tyre_age(self, d: DriverState) -> None:
        if d.compound is None:
            return
        d.tyre_age = d.stint_start_age + max(0, d.laps - d.stint_start_lap)

    def _complete_lap(self, d: DriverState, lap: int, t: float) -> LapRecord:
        statuses = self._lap_status.get(d.number) or {self.track_status}
        rec = LapRecord(
            driver=d.number,
            lap=lap,
            t_start=self._lap_start.get(d.number),
            t_end=t,
            lap_time=None,
            lap_time_t=None,
            position=d.position,
            gap_to_leader=d.gap_to_leader,
            laps_down=d.laps_down,
            interval=d.interval,
            compound=d.compound,
            tyre_age=d.tyre_age,
            stint=d.stint,
            is_in_lap=d.last_in_lap == lap,
            is_out_lap=d.last_out_lap == lap,
            track_status=",".join(sorted(statuses)),
        )
        self.laps.append(rec)
        self.new_laps.append(rec)
        self._pending_time[d.number] = rec
        self._lap_start[d.number] = t
        self._lap_status[d.number] = {self.track_status}
        return rec

    def _on_PitStopSeries(self, data: dict, e: Event) -> None:
        self._has_pit_series = True
        full = (self.topics["PitStopSeries"] or {}).get("PitTimes") or {}
        for num in (data.get("PitTimes") or {}):
            stops = full.get(num) or []
            if isinstance(stops, dict):
                stops = [stops[k] for k in sorted(stops, key=lambda k: _to_int(k) or 0)]
            seen = self._pit_series_count.get(num, 0)
            for s in stops[seen:]:
                p = (s or {}).get("PitStop") or {}
                self.pit_stops.append(
                    PitStopRecord(
                        driver=str(p.get("RacingNumber", num)),
                        lap=_to_int(p.get("Lap")),
                        lane_time=_to_float(p.get("PitLaneTime")),
                        stop_time=_to_float(p.get("PitStopTime")),
                        t=e.t,
                        source="PitStopSeries",
                    )
                )
            self._pit_series_count[num] = len(stops)

    def _on_PitLaneTimeCollection(self, data: dict, e: Event) -> None:
        # Older seasons only publish this topic; skip it when PitStopSeries exists.
        if self._has_pit_series:
            return
        for num, p in (data.get("PitTimes") or {}).items():
            if num == "_deleted" or not isinstance(p, dict) or "Duration" not in p:
                continue
            key = (num, str(p.get("Lap")))
            if key in self._plt_seen:
                continue
            self._plt_seen.add(key)
            self.pit_stops.append(
                PitStopRecord(
                    driver=str(p.get("RacingNumber", num)),
                    lap=_to_int(p.get("Lap")),
                    lane_time=_to_float(p.get("Duration")),
                    stop_time=None,
                    t=e.t,
                    source="PitLaneTimeCollection",
                )
            )


def replay(log: EventLog, until: float | None = None) -> RaceState:
    """Fold the log into a state, stopping after the last event at or before ``until``."""
    state = RaceState(log.meta)
    for e in log.events:
        if until is not None and e.t > until:
            break
        state.apply(e)
    return state


def track_status_name(code: str) -> str:
    return TRACK_STATUS.get(code, code)


def is_dry(compound: str | None) -> bool:
    return compound in DRY_COMPOUNDS
