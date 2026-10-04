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

from .config import ARCHIVE_BASE, ARCHIVE_MIRRORS, DEFAULT_TOPICS, raw_dir
from .events import parse_stream_line

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "pitsense/0.1 (+https://github.com/)"

# Sessions whose streams are published but which the season's Index.json leaves out
# (checked against the archive in Oct 2026): 2018 starts at round 2, 2024 at round 10.
# Each session's own SessionInfo stream supplies the fields an index entry would have.
INDEX_GAPS: dict[int, tuple[str, ...]] = {
    2018: ("2018/2018-03-25_Australian_Grand_Prix/2018-03-25_Race/",),
    2024: tuple(
        f"2024/{p}/"
        for p in (
            "2024-03-02_Bahrain_Grand_Prix/2024-03-02_Race",
            "2024-03-09_Saudi_Arabian_Grand_Prix/2024-03-09_Race",
            "2024-03-24_Australian_Grand_Prix/2024-03-24_Race",
            "2024-04-07_Japanese_Grand_Prix/2024-04-07_Race",
            "2024-04-21_Chinese_Grand_Prix/2024-04-20_Sprint",
            "2024-04-21_Chinese_Grand_Prix/2024-04-21_Race",
            "2024-05-05_Miami_Grand_Prix/2024-05-04_Sprint",
            "2024-05-05_Miami_Grand_Prix/2024-05-05_Race",
            "2024-05-19_Emilia_Romagna_Grand_Prix/2024-05-19_Race",
            "2024-05-26_Monaco_Grand_Prix/2024-05-26_Race",
            "2024-06-09_Canadian_Grand_Prix/2024-06-09_Race",
        )
    ),
}

# The archive gives Mugello (2020 Tuscan GP, meeting 1053) the circuit key Jeddah has
# used since 2021. A key of its own keeps Jeddah's first pit-loss prior from coming
# from Mugello. Negative: not an official key. Meeting key -> circuit key.
CIRCUIT_KEY_FIXES: dict[int, int] = {1053: -149}

# Type "Race" sessions that are sprints. 2021's sprints were called "Sprint Qualifying".
SPRINT_NAMES = ("Sprint", "Sprint Qualifying")


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


def _get_archive(path: str) -> tuple[bytes, str] | None:
    """(content, base URL) of an archive file, from the first source that publishes it."""
    for base in (ARCHIVE_BASE, *ARCHIVE_MIRRORS):
        r = _get(f"{base}{path}")
        if r is not None:
            return r.content, base
    return None


def _save(target: Path, content: bytes, base: str) -> None:
    """Write atomically; note the source in SOURCES.json when it wasn't the archive."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.replace(target)
    sources = target.parent / "SOURCES.json"
    if base != ARCHIVE_BASE or sources.exists():
        known = json.loads(sources.read_text(encoding="utf-8")) if sources.exists() else {}
        if base == ARCHIVE_BASE:
            known.pop(target.name, None)  # re-fetched from the archive itself
        else:
            known[target.name] = base
        sources.write_text(json.dumps(known, indent=1, sort_keys=True), encoding="utf-8")


def _session_info(path: str) -> dict | None:
    """The first SessionInfo message of a session: the same fields as its index entry.

    Session metadata (names, keys, scheduled start), published before the session.
    """
    local = raw_dir() / path.rstrip("/") / "SessionInfo.jsonStream"
    if not local.exists():
        got = _get_archive(f"{path}SessionInfo.jsonStream")
        if got is None:
            return None
        _save(local, *got)
    for line in _decode(local.read_bytes()).splitlines():
        parsed = parse_stream_line(line)
        if parsed is not None and isinstance(parsed[1], dict) and parsed[1].get("Path") == path:
            return parsed[1]
    return None


def _first_start(meeting: dict) -> str:
    return min((s.get("StartDate") or "" for s in meeting.get("Sessions", [])), default="")


def _fill_gaps(year: int, meetings: list[dict]) -> list[dict]:
    """Add the INDEX_GAPS sessions the index lacks, keeping meetings in calendar order."""
    gaps = INDEX_GAPS.get(year, ())
    if not gaps:
        return meetings
    meetings = [dict(m, Sessions=list(m.get("Sessions", []))) for m in meetings]
    known = {s.get("Path") for m in meetings for s in m["Sessions"]}
    for path in gaps:
        if path in known:
            continue
        info = _session_info(path)
        if info is None:
            continue
        session = {k: info.get(k) for k in ("Key", "Type", "Name", "StartDate", "EndDate", "GmtOffset", "Path")}
        meeting = info.get("Meeting") or {}
        home = next((m for m in meetings if m.get("Key") == meeting.get("Key")), None)
        if home is None:
            home = {**meeting, "Sessions": []}
            start = session.get("StartDate") or ""
            i = next((k for k, m in enumerate(meetings) if _first_start(m) > start), len(meetings))
            meetings.insert(i, home)
        home["Sessions"].append(session)
        home["Sessions"].sort(key=lambda s: s.get("StartDate") or "")
    return meetings


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
        got = _get_archive(f"{year}/Index.json")
        if got is None:
            raise RuntimeError(f"No archive index for {year}")
        _save(cache, *got)
    data = json.loads(_decode(cache.read_bytes()))

    refs: list[SessionRef] = []
    round_index = 0
    for meeting in _fill_gaps(year, data.get("Meetings", [])):
        if "test" in meeting.get("Name", "").lower():
            continue
        round_index += 1
        circuit_key = int((meeting.get("Circuit") or {}).get("Key", -1))
        for s in meeting.get("Sessions", []):
            # sessions without data, and ones filed under another season: the 2021 index
            # lists a 2022 FOM track test whose path starts with "../uat/"
            if not s.get("Path") or not s["Path"].startswith(f"{year}/"):
                continue
            refs.append(
                SessionRef(
                    year=year,
                    meeting_key=int(meeting["Key"]),
                    meeting_name=meeting.get("Name", ""),
                    location=meeting.get("Location", ""),
                    country=(meeting.get("Country") or {}).get("Name", ""),
                    circuit_key=CIRCUIT_KEY_FIXES.get(int(meeting["Key"]), circuit_key),
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
    """Grand Prix race sessions of a season that have already started (and sprints if asked).

    Selected by name: Type "Race" also covers 2021's sprints and one-off test sessions.
    """
    now = datetime.now(timezone.utc)
    out = []
    for ref in fetch_index(year, refresh=refresh):
        if ref.session_type != "Race":
            continue
        if ref.session_name != "Race" and not (include_sprints and ref.session_name in SPRINT_NAMES):
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
        got = _get_archive(f"{ref.path}{topic}.jsonStream")
        if got is None:
            missing.write_text("not published for this session\n", encoding="utf-8")
            continue
        _save(target, *got)
    return out


def radio_paths(session_dir: Path) -> list[str]:
    """Mp3 paths (relative to the session folder) named by a downloaded TeamRadio stream."""
    f = session_dir / "TeamRadio.jsonStream"
    if not f.exists():
        return []
    seen: dict[str, None] = {}
    for line in _decode(f.read_bytes()).splitlines():
        parsed = parse_stream_line(line)
        if parsed is None or not isinstance(parsed[1], dict):
            continue
        caps = parsed[1].get("Captures") or ()
        for c in caps.values() if isinstance(caps, dict) else caps:
            if isinstance(c, dict) and c.get("Path"):
                seen[c["Path"]] = None
    return list(seen)


def download_radio(ref: SessionRef, *, jobs: int = 3) -> tuple[int, int]:
    """Download the mp3s a session's TeamRadio stream names, next to the stream.

    Returns (downloaded, not published). A missing clip leaves ``X.mp3.missing`` so it is
    not asked for again. Run :func:`download_session` with ``RADIO_TOPICS`` first.
    """
    from concurrent.futures import ThreadPoolExecutor

    out = ref.local_dir
    todo = [p for p in radio_paths(out) if not (out / p).exists() and not (out / (p + ".missing")).exists()]

    def one(path: str) -> bool:
        got = _get_archive(f"{ref.path}{path}")
        if got is None:
            marker = out / (path + ".missing")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("not published\n", encoding="utf-8")
            return False
        _save(out / path, *got)
        return True

    with ThreadPoolExecutor(max(1, min(jobs, 3))) as pool:
        ok = list(pool.map(one, todo))
    return sum(ok), len(ok) - sum(ok)


def load_ref(session_dir: Path) -> SessionRef:
    return SessionRef.from_dict(json.loads((session_dir / "session.json").read_text(encoding="utf-8")))


def downloaded_sessions() -> list[SessionRef]:
    """All sessions already on disk, oldest first."""
    refs = [load_ref(p.parent) for p in raw_dir().glob("*/*/*/session.json")]
    return sorted(refs, key=lambda r: r.start_utc)
