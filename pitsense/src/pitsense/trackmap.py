"""Track outline per circuit, learned from finished races' Position data.

For a finished race this picks the fastest clean lap of one car and resamples its path into a
closed polyline, finds the pit lane from a green-flag pit stop, and takes the start/finish line
as the start of the lap. The result is cached as JSON per (circuit_key, race), and
:func:`track_asof` returns the outline from the latest race that **finished before** a given
time, so the map is available before the race starts and never uses the race it is drawn for.

    pitsense tracks build --year 2025 2026
    track_asof(ref.circuit_key, ref.start_utc)  ->  {"x": [...], "y": [...], "start": {...}, "pit": {...}}

Coordinates are the feed's X/Y (1/10 m). Start/finish is where the timing feed published the
line crossing: within about a second of the true line (a car covers 50-80 m per second).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .archive import SessionRef, load_ref
from .config import data_dir
from .events import decode_z, load_archive_session, parse_stream_line
from .feeds import parse_utc
from .state import replay

N_OUTLINE = 400  # points of the closed lap polyline
N_PIT = 60  # points of the pit-lane polyline
MIN_SAMPLES = 120  # position samples a lap must have (about 5 Hz -> 25 s)
MAX_GAP_S = 2.0  # a lap with a longer hole in its samples is skipped
CLOSE_FRAC = 0.03  # first and last point of a lap must be this close, as a fraction of its length


def tracks_dir() -> Path:
    return data_dir() / "feeds" / "tracks"


def load_positions(session_dir: Path) -> dict[str, np.ndarray]:
    """Whole Position.z stream of a finished session: {car: array of rows (utc, x, y, on_track)}."""
    f = Path(session_dir) / "Position.z.jsonStream"
    rows: dict[str, list] = {}
    for line in f.read_bytes().decode("utf-8-sig").splitlines():
        parsed = parse_stream_line(line)
        if parsed is None:
            continue
        for entry in decode_z(parsed[1]).get("Position", ()):
            utc = parse_utc(entry["Timestamp"])
            for car, d in (entry.get("Entries") or {}).items():
                if "X" in d and "Y" in d:
                    rows.setdefault(car, []).append((utc, d["X"], d["Y"], d.get("Status") == "OnTrack"))
    return {c: np.array(sorted(r), dtype=float) for c, r in rows.items()}


def _resample(xy: np.ndarray, n: int, closed: bool) -> np.ndarray:
    """Points evenly spaced along the path's length (stationary points dropped)."""
    keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1.0]
    xy = xy[keep]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    grid = np.linspace(0.0, s[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])])


def _length(xy: np.ndarray) -> float:
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def _best_lap(final, pos: dict[str, np.ndarray], utc_of) -> tuple[np.ndarray, str, int] | None:
    laps = sorted(
        (l for l in final.laps if l.lap_time and l.lap > 1 and not l.is_in_lap and not l.is_out_lap
         and l.track_status == "1" and l.driver in pos),
        key=lambda l: l.lap_time,
    )
    for lap in laps[:40]:
        u1 = utc_of(lap.t_end)
        a = pos[lap.driver]
        seg = a[(a[:, 0] >= u1 - lap.lap_time) & (a[:, 0] <= u1) & (a[:, 3] > 0)]
        if len(seg) < MIN_SAMPLES or np.diff(seg[:, 0]).max() > MAX_GAP_S:
            continue
        xy = seg[:, 1:3]
        if np.hypot(*(xy[0] - xy[-1])) > CLOSE_FRAC * _length(xy):
            continue
        return xy, lap.driver, lap.lap
    return None


def _pit_lane(final, pos: dict[str, np.ndarray], utc_of) -> np.ndarray | None:
    best = None
    for pe in final.pit_events:
        if pe.out_t is None or pe.under_red or pe.status_at_entry != "1" or pe.driver not in pos:
            continue
        a = pos[pe.driver]
        seg = a[(a[:, 0] >= utc_of(pe.in_t) - 2) & (a[:, 0] <= utc_of(pe.out_t) + 2)]
        if len(seg) < 30:
            continue
        if best is None or pe.lane_time < best[0]:
            best = (pe.lane_time, seg[:, 1:3])
    if best is None:
        return None
    xy = best[1]
    return _resample(xy, N_PIT, closed=False) if _length(xy) > 200 else None


def build_outline(session_dir: Path) -> dict | None:
    """Outline of the circuit from one finished, downloaded race (needs Position.z). None if unusable."""
    session_dir = Path(session_dir)
    ref = load_ref(session_dir)
    log = load_archive_session(session_dir)  # timing only
    final = replay(log)
    start = log.stream_start_utc()
    if start is None or not (session_dir / "Position.z.jsonStream").exists():
        return None
    base = start.timestamp()
    utc_of = lambda t: base + t  # noqa: E731
    pos = load_positions(session_dir)
    lap = _best_lap(final, pos, utc_of)
    if lap is None:
        return None
    xy, driver, lap_no = lap
    line = _resample(xy, N_OUTLINE, closed=True)
    pit = _pit_lane(final, pos, utc_of)
    heading = np.degrees(np.arctan2(*(line[3] - line[0])[::-1]))  # direction of travel at the line
    end = ref.start_utc + timedelta(hours=3)
    if final.finished_t is not None and final.started_t is not None:
        end = ref.start_utc + timedelta(seconds=final.finished_t - final.started_t)
    return {
        "circuit_key": ref.circuit_key,
        "circuit": ref.circuit,
        "race_id": ref.slug,
        "start_utc": ref.start_utc.isoformat(),
        "end_utc": end.isoformat(),
        "source": {"driver": driver, "lap": lap_no},
        "length_units": round(_length(np.vstack([line, line[:1]])), 1),
        "x": [round(float(v), 1) for v in line[:, 0]],
        "y": [round(float(v), 1) for v in line[:, 1]],
        "start": {"x": round(float(line[0, 0]), 1), "y": round(float(line[0, 1]), 1), "heading_deg": round(float(heading), 1)},
        "pit": None if pit is None else {
            "x": [round(float(v), 1) for v in pit[:, 0]], "y": [round(float(v), 1) for v in pit[:, 1]]},
    }


def build_tracks(refs: list[SessionRef], *, force: bool = False) -> list[Path]:
    """Build and cache the outline of every given race that has Position.z. Returns files written."""
    out = []
    for ref in refs:
        f = tracks_dir() / str(ref.circuit_key) / f"{ref.slug}.json"
        if f.exists() and not force:
            continue
        if not (ref.local_dir / "Position.z.jsonStream").exists():
            continue
        outline = build_outline(ref.local_dir)
        if outline is None:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(outline, separators=(",", ":")), encoding="utf-8")
        out.append(f)
    return out


def track_asof(circuit_key: int, before_utc: datetime, root: Path | None = None) -> dict | None:
    """Outline of a circuit from the latest cached race that finished before ``before_utc``."""
    folder = (root or tracks_dir()) / str(circuit_key)
    best = None
    for f in sorted(folder.glob("*.json")) if folder.exists() else ():
        d = json.loads(f.read_text(encoding="utf-8"))
        end = datetime.fromisoformat(d["end_utc"])
        if end < before_utc and (best is None or end > datetime.fromisoformat(best["end_utc"])):
            best = d
    return best


def add_commands(sub) -> None:
    def cmd(a) -> None:
        from . import archive

        refs = []
        for y in a.year:
            refs += archive.races(y)
        files = build_tracks(refs, force=a.force)
        print(f"{len(files)} outline(s) written to {tracks_dir()}")

    s = sub.add_parser("tracks", help="build track outlines from downloaded Position.z data")
    s.add_argument("--year", type=int, nargs="+", default=[2025, 2026])
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd)
