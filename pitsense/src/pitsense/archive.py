"""Download raw F1 live-timing streams from the public static archive.

Each topic is stored exactly as published (``Topic.jsonStream``: one timestamped
message per line). These are the same messages the live feed sends, which is
why replaying them exercises the same code path as a live race.

Never read the ``Topic.json`` keyframes for replay: they hold the *final* state
of the session, i.e. the future.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .config import ARCHIVE_BASE, DEFAULT_TOPICS, raw_dir

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "pitsense/0.1 (+https://github.com/)"


@dataclass(frozen=True)
class SessionRef:
    year: int
    meeting_key: int
    meeting_name: str
    location: str
    country: str
    circuit_key: int
    circuit: str
    round_index: int  # order within the season's index (testing excluded)
    session_key: int
    session_name: str  # "Race", "Sprint", "Qualifying", ...
    session_type: str
    start_local: str  # e.g. 2026-07-26T15:00:00 (local track time)
    gmt_offset: str  # e.g. 02:00:00 or -05:00:00
    path: str  # archive path, ends with "/"

    @property
    def start_utc(self) -> datetime:
        local = datetime.fromisoformat(self.start_local)
        sign = -1 if self.gmt_offset.startswith("-") else 1
        h, m, s = (int(x) for x in self.gmt_offset.lstrip("-+").split(":"))
        offset = timedelta(hours=h, minutes=m, seconds=s) * sign
        return (local - offset).replace(tzinfo=timezone.utc)

    @property
    def slug(self) -> str:
        name = _slugify(self.meeting_name)
        return f"{self.year}-{self.round_index:02d}-{name}-{_slugify(self.session_name)}"

    @property
    def local_dir(self) -> Path:
        return raw_dir() / self.path.rstrip("/")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SessionRef":
        return SessionRef(**d)


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _get(url: str, *, retries: int = 4, timeout: float = 60.0) -> requests.Response | None:
    """GET with retries. Returns None for 403/404 (topic not published)."""
    delay = 1.0
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code in (403, 404):
            return None
        if r.status_code >= 500 and attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r
    return None


def _decode(content: bytes) -> str:
    return content.decode("utf-8-sig")


def fetch_index(year: int, *, refresh: bool = False) -> list[SessionRef]:
    """All sessions of a season, in calendar order (pre-season testing excluded)."""
    cache = raw_dir() / str(year) / "Index.json"
    stale = (
        not cache.exists()
        or refresh
        # the current season's index grows during the year; refresh daily
        or (year >= datetime.now(timezone.utc).year and time.time() - cache.stat().st_mtime > 86400)
    )
    if stale:
        r = _get(f"{ARCHIVE_BASE}{year}/Index.json")
        if r is None:
            raise RuntimeError(f"No archive index for {year}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
    data = json.loads(_decode(cache.read_bytes()))

    refs: list[SessionRef] = []
    round_index = 0
    for meeting in data.get("Meetings", []):
        if "test" in meeting.get("Name", "").lower():
            continue
        round_index += 1
        for s in meeting.get("Sessions", []):
            if not s.get("Path"):
                continue
            refs.append(
                SessionRef(
                    year=year,
                    meeting_key=int(meeting["Key"]),
                    meeting_name=meeting.get("Name", ""),
                    location=meeting.get("Location", ""),
                    country=(meeting.get("Country") or {}).get("Name", ""),
                    circuit_key=int((meeting.get("Circuit") or {}).get("Key", -1)),
                    circuit=(meeting.get("Circuit") or {}).get("ShortName", ""),
                    round_index=round_index,
                    session_key=int(s.get("Key", -1)),
                    session_name=s.get("Name", ""),
                    session_type=s.get("Type", ""),
                    start_local=s.get("StartDate", ""),
                    gmt_offset=s.get("GmtOffset", "00:00:00"),
                    path=s["Path"],
                )
            )
    return refs


def races(year: int, *, include_sprints: bool = False, refresh: bool = False) -> list[SessionRef]:
    """Grand Prix race sessions of a season that have already started."""
    now = datetime.now(timezone.utc)
    out = []
    for ref in fetch_index(year, refresh=refresh):
        if ref.session_type != "Race":
            continue
        if ref.session_name == "Sprint" and not include_sprints:
            continue
        if ref.start_utc > now:
            continue
        out.append(ref)
    return out


def find_session(year: int, query: str, session_name: str = "Race") -> SessionRef:
    """Find a session by fuzzy meeting name, e.g. ('2026', 'hungary')."""
    q = _slugify(query)
    candidates = [
        r
        for r in fetch_index(year)
        if r.session_name.lower() == session_name.lower()
        and any(
            q in _slugify(x)
            for x in (r.meeting_name, r.location, r.country, r.circuit, str(r.round_index))
        )
    ]
    if not candidates:
        names = sorted({r.meeting_name for r in fetch_index(year)})
        raise LookupError(f"No {session_name} matching {query!r} in {year}. Meetings: {names}")
    if len(candidates) > 1:
        exact = [r for r in candidates if _slugify(r.meeting_name).startswith(q)]
        if len(exact) == 1:
            return exact[0]
        raise LookupError(f"{query!r} is ambiguous: {[c.meeting_name for c in candidates]}")
    return candidates[0]


def download_session(
    ref: SessionRef, topics: tuple[str, ...] = DEFAULT_TOPICS, *, force: bool = False
) -> Path:
    """Download raw topic streams for one session into data/raw/<path>/."""
    out = ref.local_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "session.json").write_text(json.dumps(ref.to_dict(), indent=2), encoding="utf-8")
    for topic in topics:
        target = out / f"{topic}.jsonStream"
        missing = out / f"{topic}.missing"
        if not force and (target.exists() or missing.exists()):
            continue
        r = _get(f"{ARCHIVE_BASE}{ref.path}{topic}.jsonStream")
        if r is None:
            missing.write_text("not published for this session\n", encoding="utf-8")
            continue
        tmp = target.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(target)
    return out


def load_ref(session_dir: Path) -> SessionRef:
    return SessionRef.from_dict(json.loads((session_dir / "session.json").read_text(encoding="utf-8")))


def downloaded_sessions() -> list[SessionRef]:
    """All sessions already on disk, oldest first."""
    refs = [load_ref(p.parent) for p in raw_dir().glob("*/*/*/session.json")]
    return sorted(refs, key=lambda r: r.start_utc)
