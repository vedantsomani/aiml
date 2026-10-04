"""A race weekend's other sessions: which exist, and how to download them.

Practice, sprint qualifying, sprint and qualifying all start before the race of the same
meeting, so their streams are known when the race starts (see ``before``).
"""

from __future__ import annotations

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
