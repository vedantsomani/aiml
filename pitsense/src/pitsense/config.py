"""Paths and constants. Override the data folder with the PITSENSE_DATA env var."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("PITSENSE_DATA", "data")).resolve()


def raw_dir() -> Path:
    return data_dir() / "raw"


def bench_dir() -> Path:
    return data_dir() / "bench"


def live_dir() -> Path:
    return data_dir() / "live"


ARCHIVE_BASE = "https://livetiming.formula1.com/static/"
# Tried in order when the archive does not publish a file. The archive answers 403 for
# everything from 2022; FastF1's mirror has it, and its 2021 files are byte-identical to
# the archive's. Files taken from a mirror are listed in that folder's SOURCES.json.
ARCHIVE_MIRRORS: tuple[str, ...] = ("https://livetiming-mirror.fastf1.dev/static/",)

# Topics we download for every session. TimingData is ~7 MB per race; the rest are small.
DEFAULT_TOPICS: tuple[str, ...] = (
    "SessionInfo",
    "SessionStatus",
    "DriverList",
    "LapCount",
    "TrackStatus",
    "TimingData",
    "TimingAppData",
    "RaceControlMessages",
    "WeatherData",
    "PitStopSeries",
    "PitLaneTimeCollection",
    "Heartbeat",
    "ExtrapolatedClock",
)

# Opt-in feed topics (`pitsense fetch --telemetry` / `--radio`). CarData.z and Position.z are
# ~8-10 MB each per race (base64 raw-deflate JSON); TeamRadio is a few KB plus one mp3 per message.
# They never enter the benchmark: load_archive_session leaves them out unless feeds=True.
TELEMETRY_TOPICS: tuple[str, ...] = ("CarData.z", "Position.z")
RADIO_TOPICS: tuple[str, ...] = ("TeamRadio",)
FEED_TOPICS: tuple[str, ...] = TELEMETRY_TOPICS + RADIO_TOPICS

# When several messages share a timestamp, apply them in this order (lower first).
# Deterministic ordering is what makes replay reproducible.
TOPIC_PRIORITY: dict[str, int] = {
    "SessionInfo": 0,
    "SessionStatus": 1,
    "DriverList": 2,
    "ExtrapolatedClock": 3,
    "Heartbeat": 4,
    "TrackStatus": 5,
    "LapCount": 6,
    "TimingAppData": 7,
    "TimingData": 8,
    "PitStopSeries": 9,
    "PitLaneTimeCollection": 10,
    "RaceControlMessages": 11,
    "WeatherData": 12,
    "TeamRadio": 13,
    "CarData.z": 14,
    "Position.z": 15,
}

TRACK_STATUS = {
    "1": "GREEN",
    "2": "YELLOW",
    "4": "SC",
    "5": "RED",
    "6": "VSC",
    "7": "VSC_ENDING",
}

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")  # softest first

# From 2019 the feed names the weekend's three dry compounds SOFT/MEDIUM/HARD, softest
# first. 2018 uses Pirelli's own names (HYPERSOFT ... SUPERHARD), so a 2018 "SOFT" can
# be the hardest tyre of a weekend. These are the nominations, softest first, as Pirelli
# announced them weeks before each race: RaceFans, "Pirelli announces final F1 tyre
# selections of 2018" (23 Aug 2018); Russia from Autosport, "Hypersoft F1 tyres part of
# Pirelli's selection for Russian GP"; Singapore and Germany also checked on Wikipedia.
# A pre-race fact, unlike the tyres used in the race. As a transcription check only,
# every dry compound used in each 2018 race is one of its three nominations.
NOMINATIONS: dict[tuple[int, str], tuple[str, str, str]] = {
    (2018, "Australian Grand Prix"): ("ULTRASOFT", "SUPERSOFT", "SOFT"),
    (2018, "Bahrain Grand Prix"): ("SUPERSOFT", "SOFT", "MEDIUM"),
    (2018, "Chinese Grand Prix"): ("ULTRASOFT", "SOFT", "MEDIUM"),
    (2018, "Azerbaijan Grand Prix"): ("ULTRASOFT", "SUPERSOFT", "SOFT"),
    (2018, "Spanish Grand Prix"): ("SUPERSOFT", "SOFT", "MEDIUM"),
    (2018, "Monaco Grand Prix"): ("HYPERSOFT", "ULTRASOFT", "SUPERSOFT"),
    (2018, "Canadian Grand Prix"): ("HYPERSOFT", "ULTRASOFT", "SUPERSOFT"),
    (2018, "French Grand Prix"): ("ULTRASOFT", "SUPERSOFT", "SOFT"),
    (2018, "Austrian Grand Prix"): ("ULTRASOFT", "SUPERSOFT", "SOFT"),
    (2018, "British Grand Prix"): ("SOFT", "MEDIUM", "HARD"),
    (2018, "German Grand Prix"): ("ULTRASOFT", "SOFT", "MEDIUM"),
    (2018, "Hungarian Grand Prix"): ("ULTRASOFT", "SOFT", "MEDIUM"),
    (2018, "Belgian Grand Prix"): ("SUPERSOFT", "SOFT", "MEDIUM"),
    (2018, "Italian Grand Prix"): ("SUPERSOFT", "SOFT", "MEDIUM"),
    (2018, "Singapore Grand Prix"): ("HYPERSOFT", "ULTRASOFT", "SOFT"),
    (2018, "Russian Grand Prix"): ("HYPERSOFT", "ULTRASOFT", "SOFT"),
    (2018, "Japanese Grand Prix"): ("SUPERSOFT", "SOFT", "MEDIUM"),
    (2018, "United States Grand Prix"): ("ULTRASOFT", "SUPERSOFT", "SOFT"),
    (2018, "Mexican Grand Prix"): ("HYPERSOFT", "ULTRASOFT", "SUPERSOFT"),
    (2018, "Brazilian Grand Prix"): ("SUPERSOFT", "SOFT", "MEDIUM"),
    (2018, "Abu Dhabi Grand Prix"): ("HYPERSOFT", "ULTRASOFT", "SUPERSOFT"),
}
