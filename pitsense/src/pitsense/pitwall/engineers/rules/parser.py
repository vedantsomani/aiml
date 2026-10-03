"""Typed parser for race-control messages.

``parse(message, category, flag)`` turns one feed line into an :class:`RCEvent`.
Wording is matched on normalised text (upper case, single spaces), so small
differences between seasons do not matter. Anything not recognised becomes
``kind="unknown"`` and is counted by the caller; the parser never raises.

Kinds
    penalty / penalty_served   pen = time | drive_through | stop_go, seconds, reason
    reprimand, warning
    investigation / investigation_after / investigation_closed   (stewards)
    noted                      an incident was noted
    track_limits_deleted / track_limits_reinstated   one lap time of ``car``
    bw_flag                    black-and-white flag for ``car``
    blue_flag, yellow, double_yellow, flag_clear, track_clear
    sc_deployed, sc_in_this_lap, sc_pit_lane, vsc_deployed, vsc_ending
    red_flag, race_resume
    pit_exit_open/closed, pit_entry_open/closed, pit_lane_clear, pit_lane_yellow
    drs_enabled/disabled, overtake_enabled/disabled
    chequered, rain_risk (value = percent), grip_low/grip_normal
    start (formation lap, start, aborted start), info (operational notices)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TLA = r"(?: \([A-Z]{2,4}\))?"
_REASON_CLEAN = re.compile(r"\s*\(\d{1,2}:\d{2}:\d{2}\)\s*$|\s*\(\d+(?:ST|ND|RD|TH) OFFENCE\)\s*$")

_PEN_TIME = re.compile(r"(\d+) SECOND (TIME|STOP/GO) PENALTY FOR CAR (\d+)" + _TLA + r"\s*(?:-\s*(.*))?$")
_PEN_DT = re.compile(r"(DRIVE THROUGH|STOP-AND-GO|STOP AND GO) PENALTY FOR CAR (\d+)" + _TLA + r"\s*(?:-\s*(.*))?$")
_REPRIMAND = re.compile(r"(REPRIMAND|WARNING)(?: \([A-Z ]+\))? FOR CAR (\d+)" + _TLA + r"\s*(?:-\s*(.*))?$")
_CAR_NUMS = re.compile(r"\bCARS? (\d+)" + _TLA + r"(?: AND (\d+)" + _TLA + r")?")
_TL_DELETED = re.compile(r"CAR (\d+)" + _TLA + r" (?:TIME [\d:.]+|LAP(?: \d+)?) DELETED - TRACK LIMITS")
_TL_REINSTATED = re.compile(r"CAR (\d+)" + _TLA + r" (?:TIME [\d:.]+|LAP(?: \d+)?) WILL BE REINSTATED")
_BW = re.compile(r"BLACK AND WHITE FLAG FOR CAR (\d+)" + _TLA + r"\s*(?:-\s*(.*))?$")
_BLUE = re.compile(r"WAVED BLUE FLAG FOR CAR (\d+)")
_RAIN = re.compile(r"RISK OF RAIN FOR .*? IS (\d+) ?%")

_EXACT = {
    "SAFETY CAR DEPLOYED": "sc_deployed",
    "SAFETY CAR IN THIS LAP": "sc_in_this_lap",
    "SAFETY CAR THROUGH THE PIT LANE": "sc_pit_lane",
    "VSC DEPLOYED": "vsc_deployed",
    "VIRTUAL SAFETY CAR DEPLOYED": "vsc_deployed",
    "VSC ENDING": "vsc_ending",
    "VIRTUAL SAFETY CAR ENDING": "vsc_ending",
    "CHEQUERED FLAG": "chequered",
    "TRACK CLEAR": "track_clear",
    "PIT EXIT OPEN": "pit_exit_open",
    "GREEN LIGHT - PIT EXIT OPEN": "pit_exit_open",
    "PIT EXIT CLOSED": "pit_exit_closed",
    "PIT LANE ENTRY OPEN": "pit_entry_open",
    "PIT ENTRY OPEN": "pit_entry_open",
    "PIT LANE ENTRY CLOSED": "pit_entry_closed",
    "PIT ENTRY CLOSED": "pit_entry_closed",
    "PIT LANE CLEAR": "pit_lane_clear",
    "YELLOW IN PIT LANE": "pit_lane_yellow",
    "DRS ENABLED": "drs_enabled",
    "DRS DISABLED": "drs_disabled",
    "OVERTAKE ENABLED": "overtake_enabled",
    "OVERTAKE DISABLED": "overtake_disabled",
    "RED FLAG": "red_flag",
    "RED FLAG - RACE SUSPENDED": "red_flag",
    "LOW GRIP CONDITIONS": "grip_low",
    "NORMAL GRIP CONDITIONS": "grip_normal",
    "LOW GRIP DELTA ACTIVE": "grip_low",
    "NORMAL GRIP DELTA ACTIVE": "grip_normal",
}
_PREFIX = (
    ("DRS ENABLED IN ZONE", "drs_enabled"),
    ("DRS DISABLED IN ZONE", "drs_disabled"),
    ("RACE WILL RESUME", "race_resume"),
    ("GREEN LIGHT", "info"),
    ("FORMATION LAP", "start"),
    ("EXTRA FORMATION LAP", "start"),
    ("STANDING START", "start"),
    ("ROLLING START", "start"),
    ("RACE START", "start"),
    ("ABORTED START", "start"),
    ("STARTING PROCEDURE", "start"),
    ("RACE WILL START", "start"),
    ("RESUMPTION ORDER", "info"),
    ("PUSH CARS", "info"),
    ("LAPPED CAR", "info"),
    ("ALL CARS", "info"),
    ("ALL PASS HOLDERS", "info"),
    ("MARSHALS ON TRACK", "info"),
    ("RECOVERY VEHICLE", "info"),
    ("MEDICAL CAR", "info"),
    ("AWNINGS", "info"),
    ("SAFETY CAR LIGHTS", "info"),
    ("SAFETY CAR WILL", "info"),
    ("TRACK SURFACE SLIPPERY", "info"),
    ("AIR TEMPERATURE", "info"),
)


@dataclass(frozen=True)
class RCEvent:
    kind: str
    car: str | None = None  # the (first) car concerned
    cars: tuple[str, ...] = ()  # all cars named
    pen: str | None = None  # time | drive_through | stop_go (penalties)
    seconds: int | None = None
    reason: str | None = None
    value: float | None = None
    text: str = ""


def normalise(message: str) -> str:
    t = re.sub(r"\s+", " ", str(message or "").upper().replace("�", "-")).strip()
    t = re.sub(r"^CORRECTION:+ ?", "", t)
    return re.sub(r"\s*\(\d{1,2}:\d{2}:\d{2}\)$", "", t)  # trailing incident time stamp


def _reason(r: str | None) -> str | None:
    if not r:
        return None
    return _REASON_CLEAN.sub("", r.strip()).strip() or None


def _cars(text: str) -> tuple[str, ...]:
    m = _CAR_NUMS.search(text)
    return tuple(g for g in m.groups() if g) if m else ()


def parse(message: str, category: str = "", flag: str = "") -> RCEvent:
    """Classify one race-control message. Never raises."""
    try:
        return _parse(normalise(message), (category or "").upper(), (flag or "").upper())
    except Exception:  # defensive: an odd line must never stop a replay
        return RCEvent("unknown", text=str(message)[:200])


def _split_reason(body: str) -> str | None:
    return _reason(body.split(" - ", 1)[1]) if " - " in body else None


def _parse(t: str, category: str, flag: str) -> RCEvent:
    # ---- stewards (corrections arrive without the "FIA STEWARDS" prefix)
    if not t.startswith("FIA STEWARDS:") and ("UNDER INVESTIGATION" in t or "REVIEWED NO FURTHER" in t):
        t = "FIA STEWARDS: " + t
    if t.startswith("FIA STEWARDS:"):
        body = t[len("FIA STEWARDS:"):].strip()
        served = body.startswith("PENALTY SERVED -")
        if served:
            body = body[len("PENALTY SERVED -"):].strip()
        m = _PEN_TIME.search(body)
        if m:
            sec, kind, car, reason = m.groups()
            return RCEvent("penalty_served" if served else "penalty", car, (car,),
                           "time" if kind == "TIME" else "stop_go", int(sec), _reason(reason), text=t)
        m = _PEN_DT.search(body)
        if m:
            kind, car, reason = m.groups()
            return RCEvent("penalty_served" if served else "penalty", car, (car,),
                           "drive_through" if kind.startswith("DRIVE") else "stop_go", None, _reason(reason), text=t)
        m = _REPRIMAND.search(body)
        if m:
            kind, car, reason = m.groups()
            return RCEvent("reprimand" if kind == "REPRIMAND" else "warning", car, (car,), reason=_reason(reason), text=t)
        cars = _cars(body)
        c0 = cars[0] if cars else None
        reason = _split_reason(body)
        if "WILL BE INVESTIGATED AFTER THE RACE" in body:
            return RCEvent("investigation_after", c0, cars, reason=reason, text=t)
        if "UNDER INVESTIGATION" in body:
            return RCEvent("investigation", c0, cars, reason=reason, text=t)
        if "NO FURTHER INVESTIGATION" in body or "NO FURTHER ACTION" in body or "NO INFRINGEMENT" in body:
            return RCEvent("investigation_closed", c0, cars, reason=reason, text=t)
        if "NOTED" in body:
            return RCEvent("noted", c0, cars, reason=reason, text=t)
        return RCEvent("unknown", c0, cars, text=t)

    # ---- flags
    m = _BW.search(t)
    if m:
        return RCEvent("bw_flag", m.group(1), (m.group(1),), reason=_reason(m.group(2)), text=t)
    m = _BLUE.search(t)
    if m:
        return RCEvent("blue_flag", m.group(1), (m.group(1),), text=t)
    if flag == "DOUBLE YELLOW" or t.startswith("DOUBLE YELLOW"):
        return RCEvent("double_yellow", text=t)
    if flag == "YELLOW" or t.startswith("YELLOW IN TRACK"):
        return RCEvent("yellow", text=t)
    if t.startswith("CLEAR IN TRACK SECTOR"):
        return RCEvent("flag_clear", text=t)

    # ---- track limits and incidents
    m = _TL_DELETED.search(t)
    if m:
        return RCEvent("track_limits_deleted", m.group(1), (m.group(1),), text=t)
    m = _TL_REINSTATED.search(t)
    if m:
        return RCEvent("track_limits_reinstated", m.group(1), (m.group(1),), text=t)
    if "NOTED" in t and ("INCIDENT" in t or t.startswith("LAP ")):
        cars = _cars(t)
        return RCEvent("noted", cars[0] if cars else None, cars, reason=_split_reason(t), text=t)

    m = _RAIN.search(t)
    if m:
        return RCEvent("rain_risk", value=float(m.group(1)), text=t)

    kind = _EXACT.get(t)
    if kind:
        return RCEvent(kind, text=t)
    if "REINSTATED" in t or "DELETED" in t:
        cars = _cars(t)
        return RCEvent("info", cars[0] if cars else None, cars, text=t)
    for prefix, kind in _PREFIX:
        if t.startswith(prefix):
            if kind == "info" and t.startswith("GREEN LIGHT") and "PIT EXIT" in t:
                return RCEvent("pit_exit_open", text=t)
            return RCEvent(kind, text=t)
    if "PADDING MATERIAL" in t:
        return RCEvent("info", text=t)
    return RCEvent("unknown", text=t)
