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
}

TRACK_STATUS = {
    "1": "GREEN",
    "2": "YELLOW",
    "4": "SC",
    "5": "RED",
    "6": "VSC",
    "7": "VSC_ENDING",
}

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
