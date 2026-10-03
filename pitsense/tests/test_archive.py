"""Season index quirks and the mirror fallback, on synthetic snippets (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pitsense import archive
from pitsense.config import ARCHIVE_BASE, ARCHIVE_MIRRORS

YEAR = 1999  # a past season: the cached index is never refreshed


def _session(name: str, day: str, path: str | None, *, type_: str = "Race", key: int = 1) -> dict:
    return {"Key": key, "Type": type_, "Name": name, "StartDate": f"{day}T15:00:00", "GmtOffset": "02:00:00", "Path": path}


def _meeting(key: int, name: str, day: str, sessions: list[dict], circuit: int = 7) -> dict:
    slug = name.replace(" ", "_")
    base = f"{YEAR}/{day}_{slug}/"
    for s in sessions:
        if s["Path"] == "":
            s["Path"] = f"{base}{s['StartDate'][:10]}_{s['Name'].replace(' ', '_')}/"
    return {"Key": key, "Name": name, "Location": "Somewhere", "Country": {"Name": "Nowhere"},
            "Circuit": {"Key": circuit, "ShortName": name.split()[0]}, "Sessions": sessions}


@pytest.fixture
def season(tmp_path, monkeypatch):
    """A cached index: a test, two Grands Prix (one with a 2021-style sprint) and a stray session."""
    monkeypatch.setenv("PITSENSE_DATA", str(tmp_path))
    meetings = [
        _meeting(10, "Pre-Season Test", f"{YEAR}-03-01", [_session("Day 1", f"{YEAR}-03-01", "", type_="Practice")]),
        _meeting(12, "Beta Grand Prix", f"{YEAR}-04-11", [
            _session("Sprint Qualifying", f"{YEAR}-04-10", ""),  # 2021's name for a sprint race
            _session("Race", f"{YEAR}-04-11", ""),
            _session("High Speed Track Test", f"{YEAR + 1}-03-24", f"../uat/static/{YEAR + 1}/x/y/"),
            _session(None, f"{YEAR}-04-11", None),  # duplicate entry without data
        ], circuit=149),
        _meeting(13, "Gamma Grand Prix", f"{YEAR}-05-02", [_session("Race", f"{YEAR}-05-02", "")]),
    ]
    index = tmp_path / "raw" / str(YEAR) / "Index.json"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"\xef\xbb\xbf" + json.dumps({"Year": YEAR, "Meetings": meetings}).encode())
    return tmp_path


def test_races_are_grands_prix_only(season):
    refs = archive.races(YEAR)
    assert [(r.meeting_name, r.session_name, r.round_index) for r in refs] == [
        ("Beta Grand Prix", "Race", 1), ("Gamma Grand Prix", "Race", 2)]
    sprints = [r.session_name for r in archive.races(YEAR, include_sprints=True) if r.session_name != "Race"]
    assert sprints == ["Sprint Qualifying"]


def test_sessions_filed_under_another_season_are_skipped(season):
    names = [r.session_name for r in archive.fetch_index(YEAR)]
    assert "High Speed Track Test" not in names and None not in names
    assert all(r.local_dir.is_relative_to(season / "raw" / str(YEAR)) for r in archive.fetch_index(YEAR))


def test_index_gap_is_filled_from_the_sessions_own_session_info(season, monkeypatch):
    path = f"{YEAR}/{YEAR}-03-21_Alpha_Grand_Prix/{YEAR}-03-21_Race/"
    monkeypatch.setitem(archive.INDEX_GAPS, YEAR, (path,))
    info = {"Meeting": {"Key": 11, "Name": "Alpha Grand Prix", "Location": "Elsewhere", "Country": {"Name": "Erewhon"},
                        "Circuit": {"Key": 3, "ShortName": "Alpha"}},
            "ArchiveStatus": {"Status": "Generating"}, "Key": 501, "Type": "Race", "Name": "Race",
            "StartDate": f"{YEAR}-03-21T16:10:00", "EndDate": f"{YEAR}-03-21T18:10:00", "GmtOffset": "11:00:00", "Path": path}
    local = season / "raw" / path / "SessionInfo.jsonStream"
    local.parent.mkdir(parents=True)
    local.write_text("﻿00:00:00.000" + json.dumps(info) + "\n", encoding="utf-8")
    monkeypatch.setattr(archive, "_get", lambda url, **kw: pytest.fail(f"network: {url}"))

    refs = archive.races(YEAR)
    assert [(r.meeting_name, r.round_index) for r in refs] == [
        ("Alpha Grand Prix", 1), ("Beta Grand Prix", 2), ("Gamma Grand Prix", 3)]
    alpha = refs[0]
    assert (alpha.meeting_key, alpha.session_key, alpha.circuit_key, alpha.path) == (11, 501, 3, path)
    assert alpha.start_utc.isoformat() == f"{YEAR}-03-21T05:10:00+00:00"


def test_circuit_key_fix(season, monkeypatch):
    monkeypatch.setitem(archive.CIRCUIT_KEY_FIXES, 12, -149)
    keys = {r.meeting_name: r.circuit_key for r in archive.races(YEAR)}
    assert keys == {"Beta Grand Prix": -149, "Gamma Grand Prix": 7}


def test_unpublished_files_come_from_the_mirror_and_are_noted(season, monkeypatch):
    mirror = ARCHIVE_MIRRORS[0]

    def fake_get(url, **kw):
        if url.startswith(mirror) and "LapCount" in url:
            return SimpleNamespace(content=b'00:00:01.000{"CurrentLap":1}\n')
        return None  # 403/404 everywhere else

    monkeypatch.setattr(archive, "_get", fake_get)
    ref = archive.races(YEAR)[0]
    out = archive.download_session(ref, topics=("LapCount", "TimingData"))
    assert (out / "LapCount.jsonStream").read_bytes() == b'00:00:01.000{"CurrentLap":1}\n'
    assert (out / "TimingData.missing").exists() and not (out / "TimingData.jsonStream").exists()
    assert json.loads((out / "SOURCES.json").read_text(encoding="utf-8")) == {"LapCount.jsonStream": mirror}

    # fetched again from the archive itself: the mirror note goes away
    monkeypatch.setattr(archive, "_get", lambda url, **kw: SimpleNamespace(content=b"") if url.startswith(ARCHIVE_BASE) else None)
    archive.download_session(ref, topics=("LapCount",), force=True)
    assert json.loads((out / "SOURCES.json").read_text(encoding="utf-8")) == {}
