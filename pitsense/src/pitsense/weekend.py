"""A race weekend's other sessions: which exist, and how to download them.

Practice, sprint qualifying, sprint and qualifying all start before the race of the same
meeting, so their streams are known when the race starts (see ``before``).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import archive
from .archive import SessionRef

PRACTICE = ("Practice 1", "Practice 2", "Practice 3")
QUALI = ("Qualifying", "Sprint Qualifying")


def meeting_sessions(ref: SessionRef) -> list[SessionRef]:
    """Every session of ref's meeting, in start order."""
    out = [r for r in archive.fetch_index(ref.year) if r.meeting_key == ref.meeting_key]
    return sorted(out, key=lambda r: r.start_utc)


def before(ref: SessionRef) -> list[SessionRef]:
    """Sessions of the same meeting that started before ``ref`` did (known at its start)."""
    return [r for r in meeting_sessions(ref) if r.start_utc < ref.start_utc]


def quali_sessions(year: int) -> list[SessionRef]:
    """Qualifying and Sprint Qualifying sessions of a season that have started."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return [r for r in archive.fetch_index(year) if r.session_name in QUALI and r.start_utc <= now]


def download_weekend(race: SessionRef, topics: tuple[str, ...], *, force: bool = False,
                     feed_topics: tuple[str, ...] = ()) -> list[SessionRef]:
    """Download every non-race session of the race's meeting (Position.z etc. only for quali)."""
    got = []
    for ref in meeting_sessions(race):
        if ref.session_key == race.session_key:
            continue
        extra = feed_topics if ref.session_name in QUALI else ()
        archive.download_session(ref, topics + extra, force=force)
        got.append(ref)
    return got


# --------------------------------------------------------------------------- tyre sets
# Dry sets each driver gets for a weekend, by compound (Pirelli's usual split). Assumption, not
# a feed value: the nominated mix changes by race, and the feed never says which sets a team
# returned. Override per call (``allocation=``) or per race (``ctx.meta["tyre_allocation"]``).
ALLOCATION = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}  # 13 dry sets
ALLOCATION_SPRINT = {"SOFT": 7, "MEDIUM": 3, "HARD": 2}  # 12 dry sets at a sprint weekend
DRY = ("SOFT", "MEDIUM", "HARD")
_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}


def allocation(sprint: bool = False, returned: dict[str, int] | None = None,
               override: dict[str, int] | None = None) -> dict[str, int]:
    """Dry sets per compound at the start of the weekend, minus ``returned`` sets."""
    base = dict(override or (ALLOCATION_SPRINT if sprint else ALLOCATION))
    for c, n in (returned or {}).items():
        base[c] = max(0, base.get(c, 0) - int(n))
    return base


@dataclass
class TyreSet:
    compound: str
    laps: int  # laps run on it so far (the feed's TotalLaps)
    first: str = ""  # session where it first ran ("" = not seen new: origin unknown)
    mounted: bool = False  # on the car now / already used in the race itself

    def to_list(self) -> list:
        return [self.compound, self.laps, self.first]


def _real_stints(stints: list) -> list[dict]:
    """Stint records that mean a tyre was fitted (see ``RaceState._real_stints``)."""
    from .state import RaceState

    if isinstance(stints, dict):
        stints = [stints[k] for k in sorted(stints, key=lambda k: int(k) if str(k).isdigit() else 0)]
    return [stints[i] for i in RaceState._real_stints(stints)]


def add_stints(sets: list[TyreSet], stints: list, session: str, meta: dict | None = None) -> None:
    """Fold one car's stint records into its set list (dry compounds only).

    ``New`` "true" is a set nobody had run: a new set. A used one (``New`` "false") is matched
    to the car's set of that compound whose laps equal the stint's ``StartLaps``; with no match
    it is a used set of unknown origin. Wet compounds are not dry-set allocation and are skipped.
    """
    from .state import relative_compound

    for s in _real_stints(stints):
        compound = relative_compound(str(s.get("Compound", "")).upper(), meta or {})
        if compound not in DRY:
            continue
        start, total = _int(s.get("StartLaps")), _int(s.get("TotalLaps"))
        if str(s.get("New", "")).lower() == "true":
            sets.append(TyreSet(compound, max(total, 0), session))
            continue
        cands = [t for t in sets if t.compound == compound and t.laps <= start and not t.mounted]
        if cands:  # the closest earlier age: the set this stint continues
            best = max(cands, key=lambda t: t.laps)
            best.laps = max(total, best.laps)
        else:
            sets.append(TyreSet(compound, max(total, start), ""))


def _int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def session_stints(session_dir) -> dict[str, list]:
    """Final TimingAppData stints per car of a downloaded session ({} if the stream is missing)."""
    from pathlib import Path

    from .events import load_archive_session
    from .merge import deep_merge

    d = Path(session_dir)
    if not (d / "TimingAppData.jsonStream").exists():
        return {}
    topic = None
    for e in load_archive_session(d, ("TimingAppData",)).events:
        topic = deep_merge(topic, e.data) if e.kind != "snapshot" else e.data
    lines = (topic or {}).get("Lines") or {}
    return {n: (v.get("Stints") or []) for n, v in lines.items() if isinstance(v, dict)}


def weekend_sets(ref: SessionRef) -> dict[str, list[TyreSet]]:
    """Every set each car ran in this meeting's sessions that started before ``ref``.

    Read from those sessions' own streams, all published before ``ref`` starts. A session that
    is not downloaded contributes nothing (see ``pitsense fetch --weekend``).
    """
    out: dict[str, list[TyreSet]] = {}
    meta = {"year": ref.year, "meeting_name": ref.meeting_name}
    for s in before(ref):
        for car, stints in session_stints(s.local_dir).items():
            add_stints(out.setdefault(car, []), stints, s.session_name, meta)
    return out


def have_sessions(ref: SessionRef) -> list[str]:
    """Names of the earlier sessions of the weekend that are on disk."""
    return [s.session_name for s in before(ref) if (s.local_dir / "TimingAppData.jsonStream").exists()]


def is_sprint_weekend(ref: SessionRef) -> bool:
    return any(s.session_name == "Sprint" for s in meeting_sessions(ref))


def remaining(sets: list[TyreSet], alloc: dict[str, int]) -> dict[str, int]:
    """New sets left per compound: allocation minus the sets already run."""
    out = {}
    for c in DRY:
        used_new = sum(1 for t in sets if t.compound == c and t.first)
        out[c] = max(0, alloc.get(c, 0) - used_new)
    return out


def used_str(sets: list[TyreSet]) -> str:
    """Used sets still free to fit, e.g. "S12 S5 M20": compound letter + laps on it."""
    free = [t for t in sets if not t.mounted]
    free.sort(key=lambda t: (DRY.index(t.compound), t.laps))
    return " ".join(f"{_LETTER[t.compound]}{t.laps}" for t in free)
