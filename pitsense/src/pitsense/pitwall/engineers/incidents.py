"""Incidents: spot an unscheduled stop (puncture, wing damage, a car in trouble) before it happens.

Per car, as of now (scalars only):

* ``damage_prob``              an unplanned stop or retirement is likely at the end of this lap or the next
* ``unscheduled_stop_prob_3``  chance of such a stop within 3 laps that tyre age does not explain
* ``incident_reason``          the strongest evidence in a few words ("" when there is none)

Evidence (each also a key, so it can be inspected and used as a benchmark feature):

* ``pace_drop``    s: the last clean lap against the car's own stint (median of its previous clean laps on the set),
                   minus what the rest of the field lost on the same lap (so fuel, rain and track evolution cancel)
* ``pace_drop_prev`` the same for the lap before
* ``neigh_drop``   s: the car's gap to the cars around it on the last lap, against its own usual gap
* ``tel_slow``, ``tel_power``  0-1: ``mechanic_telemetry`` (car slowing in a sector, full-throttle speed / speed-trap drop)
* ``rc_collision``, ``rc_incident``  race control noted / is investigating an incident naming the car (collision, off track)
* ``rc_debris``    debris / recovery vehicle on track in the last minutes (any car; puncture risk)
* ``rc_x_pace``, ``pace_both``  interactions: an incident message on a car that is also slow; both pace views agree
* ``radio_dmg``    0-1: team radio about damage, a puncture or vibration (``mechanic_radio``)

The two probabilities are small logistic models on these. Weights were fitted on 2025 races only
(``python -m pitsense incidents-eval``, ``bench/incidents.py``); 2026 was scored once. Tyre age is not an input on
purpose: the labels leave out stops that tyre age explains, so the models learn what is *not* tyre wear.
Sector times and the speed trap are not in the race state; the telemetry checks stand in for them, and
need a ``feeds=True`` archive (without it those inputs are 0).
"""

from __future__ import annotations

import math
import re
import statistics

from ..engineer import Engineer
from ..memory import is_clean
from ..types import Alert

FEATURES = ("pace_drop", "pace_drop_prev", "neigh_drop", "tel_slow", "tel_power", "rc_collision", "rc_incident", "rc_debris", "radio_dmg", "rc_x_pace", "pace_both")
ALERT_DAMAGE = 0.02  # damage_prob at which an alert is raised (chosen on 2025, bench/incidents.py)
RC_CAR_S = 300.0  # an incident message counts this long
RC_DEBRIS_S = 240.0
PREV_LAPS = 5

# Logistic weights on the clipped features (clip_features), fitted on 2025 by bench/incidents.py.
BIAS = {'damage': -5.372, 'stop3': -4.917}
WEIGHTS = {
    "damage": {"pace_drop": 0.836, "pace_drop_prev": 0.063, "neigh_drop": 0.381, "tel_slow": 0.173, "tel_power": 0.018, "rc_collision": 0.61, "rc_incident": 0.186, "rc_debris": 0.13, "radio_dmg": -0.013, "rc_x_pace": 0.227, "pace_both": 0.648},
    "stop3": {"pace_drop": 0.943, "pace_drop_prev": 0.047, "neigh_drop": 0.072, "tel_slow": 0.16, "tel_power": 0.011, "rc_collision": 0.444, "rc_incident": 0.199, "rc_debris": 0.379, "radio_dmg": -0.018, "rc_x_pace": 0.202, "pace_both": 0.524},
}

_CARS = re.compile(r"\bCARS? (\d+)(?: \([A-Z]{3}\))?(?: AND (\d+))?")
_NOTED = re.compile(r"\b(?:NOTED|UNDER INVESTIGATION)\b")
_NOT_DAMAGE = re.compile(r"NO FURTHER|AFTER THE RACE|PIT LANE|YELLOW FLAG|TRACK LIMITS|FALSE START|PENALTY|REPRIMAND|BLUE FLAG|"
                         r"UNSAFE RELEASE|DRIVING|SPEEDING|JUMP START|INFRINGEMENT")
_COLLISION = re.compile(r"COLLISION|CONTACT|FORCING ANOTHER|SPUN|SPIN|OFF THE TRACK|STOPPED|BROKEN DOWN")
_DEBRIS = re.compile(r"DEBRIS|RECOVERY VEHICLE|OBJECT ON TRACK|MARSHAL")


def classify_rc(message: str) -> tuple[str, tuple[str, ...]]:
    """("collision" | "incident" | "debris" | "", car numbers named) for one race-control line."""
    u = re.sub(r"\s+", " ", message.upper())
    if _DEBRIS.search(u) and not _NOT_DAMAGE.search(u):
        return "debris", ()
    if "INCIDENT" not in u or not _NOTED.search(u) or _NOT_DAMAGE.search(u):
        return "", ()
    cars = tuple(c for m in _CARS.finditer(u) for c in m.groups() if c)
    if not cars:
        return "", ()
    return ("collision" if _COLLISION.search(u) else "incident"), cars


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def clip_features(f: dict) -> dict[str, float]:
    """Model inputs: the evidence scaled to roughly 0-1."""
    pd, nd = max(0.0, min(f["pace_drop"], 8.0)) / 4.0, max(0.0, min(f["neigh_drop"], 8.0)) / 4.0
    rc = max(f["rc_collision"], f["rc_incident"])
    return {
        "rc_x_pace": rc * max(pd, nd), "pace_both": min(pd, nd),
        "pace_drop": max(0.0, min(f["pace_drop"], 8.0)) / 4.0,
        "pace_drop_prev": max(0.0, min(f["pace_drop_prev"], 8.0)) / 4.0,
        "neigh_drop": max(0.0, min(f["neigh_drop"], 8.0)) / 4.0,
        "tel_slow": f["tel_slow"], "tel_power": f["tel_power"],
        "rc_collision": f["rc_collision"], "rc_incident": f["rc_incident"], "rc_debris": f["rc_debris"],
        "radio_dmg": f["radio_dmg"],
    }


def probabilities(f: dict, bias: dict | None = None, weights: dict | None = None) -> tuple[float, float]:
    bias, weights = bias or BIAS, weights or WEIGHTS
    x = clip_features(f)
    out = []
    for k in ("damage", "stop3"):
        out.append(_sig(bias[k] + sum(weights[k][n] * x[n] for n in FEATURES)))
    return out[0], out[1]


def _label(f: dict) -> tuple[str, float]:
    """(reason text, strength) of the strongest evidence."""
    cands = [
        (f["rc_collision"], "race control: collision involving the car"),
        (f["radio_dmg"], "radio reports damage / puncture"),
        (f["tel_slow"], "car slowing on track"),
        (f["rc_incident"], "race control: incident noted"),
        (f["tel_power"], "speed / power loss on the straights"),
        (min(1.0, f["pace_drop"] / 4.0), f"sudden pace loss {f['pace_drop']:.1f} s against its own stint"),
        (min(1.0, f["neigh_drop"] / 4.0), f"{f['neigh_drop']:.1f} s slower than the cars around it"),
        (0.5 * f["rc_debris"], "debris on track"),
    ]
    s, text = max(cands, key=lambda c: c[0])
    return (text, s) if s >= 0.2 else ("", 0.0)


class IncidentsEngineer(Engineer):
    name = "incidents"
    requires = ("mechanic_telemetry", "mechanic_radio")
    features = ("damage_prob", "unscheduled_stop_prob_3", "pace_drop", "neigh_drop")
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._rc_seen = 0
        self._rc_car: dict[str, list[tuple[float, str]]] = {}  # car -> [(t, kind)]
        self._rc_debris: float | None = None
        self._delta_cache: dict[tuple[str, int], float | None] = {}

    def observe(self, state) -> None:
        rc = state.rc
        while self._rc_seen < len(rc):
            m = rc[self._rc_seen]
            self._rc_seen += 1
            kind, cars = classify_rc(m.message)
            if kind == "debris":
                self._rc_debris = m.t
            elif kind:
                for c in cars:
                    self._rc_car.setdefault(c, []).append((m.t, kind))

    # ------------------------------------------------------------------ pace evidence
    def _delta(self, car: str, lap: int) -> float | None:
        """Lap time minus the median of the car's previous clean laps on the same set (needs 3); None if it says nothing."""
        key = (car, lap)
        if key in self._delta_cache:
            return self._delta_cache[key]
        by = self.memory.index.by_driver.get(car, {})
        rec = by.get(lap)
        res = None
        if is_clean(rec):
            prev = [by[n].lap_time for n in range(lap - 1, 0, -1) if n in by and is_clean(by[n]) and by[n].stint == rec.stint]
            prev = prev[:PREV_LAPS]
            if len(prev) >= 3:
                res = rec.lap_time - statistics.median(prev)
        # a lap already published stays as it is
        if rec is not None:
            self._delta_cache[key] = res
        return res

    def _field_delta(self, lap: int, exclude: str) -> float | None:
        ds = [d for c in self.memory.index.by_driver if c != exclude for d in [self._delta(c, lap)] if d is not None]
        return statistics.median(ds) if len(ds) >= 5 else None

    def _rel(self, car: str, lap: int) -> float | None:
        """Lap time minus the median of the cars within two places on the same lap."""
        by = self.memory.index.by_driver
        rec = by.get(car, {}).get(lap)
        if not is_clean(rec) or rec.position is None:
            return None
        ts = [o[lap].lap_time for c, o in by.items() if c != car and lap in o and o[lap].position is not None
              and abs(o[lap].position - rec.position) <= 2 and is_clean(o[lap])]
        return rec.lap_time - statistics.median(ts) if len(ts) >= 2 else None

    def pace(self, car: str) -> tuple[float, float, float]:
        """(pace_drop, pace_drop_prev, neigh_drop) from the car's latest laps."""
        by = self.memory.index.by_driver.get(car, {})
        if not by:
            return 0.0, 0.0, 0.0
        last = max(by)
        out = []
        for lap in (last, last - 1):
            d, fd = self._delta(car, lap), self._field_delta(lap, car)
            out.append(0.0 if d is None or fd is None else d - fd)
        nd = 0.0
        r = self._rel(car, last)
        if r is not None:
            prior = [x for x in (self._rel(car, n) for n in range(last - 1, last - 1 - PREV_LAPS, -1)) if x is not None]
            if len(prior) >= 3:
                nd = r - statistics.median(prior)
        return out[0], out[1], nd

    # ------------------------------------------------------------------ outputs
    def evidence(self, state, number, view) -> dict:
        tel = view.car("mechanic_telemetry", number)
        rad = view.car("mechanic_radio", number)
        pd, pdp, nd = self.pace(number)
        ev = [e for e in self._rc_car.get(number, ()) if 0 <= state.t - e[0] <= RC_CAR_S]
        dbr = self._rc_debris is not None and 0 <= state.t - self._rc_debris <= RC_DEBRIS_S
        dmg = rad.get("radio_issue") in ("damage", "puncture", "vibration")
        return {
            "pace_drop": round(pd, 3), "pace_drop_prev": round(pdp, 3), "neigh_drop": round(nd, 3),
            "tel_slow": float(tel.get("slow_car") or 0.0), "tel_power": float(tel.get("power_loss") or 0.0),
            "rc_collision": 1.0 if any(k == "collision" for _, k in ev) else 0.0,
            "rc_incident": 1.0 if any(k == "incident" for _, k in ev) else 0.0,
            "rc_debris": 1.0 if dbr else 0.0,
            "radio_dmg": float(rad.get("radio_risk") or 0.0) if dmg else 0.0,
        }

    def car(self, state, number, view):
        d = state.drivers.get(number)
        f = self.evidence(state, number, view)
        out = bool(d is not None and not d.running)
        pd, p3 = probabilities(f)
        reason, _ = _label(f)
        if out:
            pd, p3, reason = 0.0, 0.0, ""
        return {"damage_prob": round(pd, 4), "unscheduled_stop_prob_3": round(max(p3, pd), 4), "incident_reason": reason, **f}

    def alerts(self, state, view):
        out = []
        for n in sorted(state.drivers, key=lambda x: int(x) if x.isdigit() else 999):
            v = view.car(self.name, n)
            if v["damage_prob"] < ALERT_DAMAGE:
                continue
            d = state.drivers[n]
            out.append(Alert(state.t, self.name, "damage_risk", "warn",
                             f"Car {n}{f' ({d.tla})' if d.tla else ''}: unscheduled stop likely ({v['incident_reason'] or 'several signs'})",
                             car=n, data={"damage_prob": v["damage_prob"]}))
        return out
