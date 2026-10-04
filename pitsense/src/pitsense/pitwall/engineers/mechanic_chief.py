"""Chief mechanic: one verdict per car from telemetry, radio, lap times and race control.

Evidence, each 0-1, with the time it became true (``since``):

* telemetry   ``mechanic_telemetry``: the worst check score; counted in full only while a
              level-triggered check is active, at half weight below that (it corroborates).
* radio       ``mechanic_radio``: the decayed risk of the strongest problem message.
* lap time    the last clean lap is >= 2.5 s slower than the median of the car's previous
              clean laps (rising to 1 at 6 s), or two clean laps in a row are >= 1.5 s slower.
              No pit, no SC/VSC/yellow, not lap 1, not in traffic (< 1.2 s behind the car ahead).
* race ctrl   a race-control message that names the car as stopped, slow or broken down.

``risk = 1 - prod(1 - w * e)`` with weights telemetry 1.0, radio 0.8, race control 0.9 and
lap time 0.5 (a slow lap alone has many innocent causes, so it never raises an alert by
itself). The alert is level-triggered: it is on while ``risk >= ALERT_RISK`` and its ``since``
is the earliest evidence behind it, so asking more or less often changes nothing.

Values per car: ``risk``, ``issue`` (power_loss / brake_issue / gearbox / slow_car / damage /
puncture / vibration / ... or ""), ``since``, ``why`` (one line), ``tel``, ``radio``, ``lap``,
``rc`` (the four evidence values), ``out`` (retired or stopped).
"""

from __future__ import annotations

import re

from ..engineer import Engineer
from ..memory import is_clean
from ..types import Alert

ALERT_RISK = 0.75  # chosen on 2025 (bench/mechanics.py): telemetry at its own alert level, or corroborated evidence
WEIGHTS = {"tel": 1.0, "radio": 0.8, "rc": 0.9, "lap": 0.5}
RC_WINDOW_S = 600.0

_RADIO_TO_CHECK = {"power": "power_loss", "brakes": "brake_issue", "gearbox": "gearbox", "retire": "slow_car",
                   "leak_fire": "power_loss", "electrical": "power_loss"}
_RC_CAR = re.compile(r"\bCARS? (\d+)")
_RC_STOP = re.compile(r"\b(?:STOPPED|STOPPING|STATIONARY|BROKEN DOWN|BREAKDOWN|STRANDED|SLOW(?:ING)? (?:CAR|ON)|CAR SLOW|SLOW CAR|SLOWLY ON)\b")
_RC_NOT = re.compile(r"UNNECESSARILY SLOW|DRIVING SLOWLY|SLOWLY (?:AT|IN|ON THE) (?:THE )?(?:PIT|FORMATION)|STOPPED (?:IN|AT) (?:THE )?(?:PIT|GRID)|STOP[- ]GO|STOP AND GO|STOPPED THE (?:RACE|SESSION)|PIT STOP")


def rc_car_problem(message: str) -> str | None:
    """Car number if the race-control line says that car stopped / is slow, else None."""
    u = re.sub(r"\s+", " ", message.upper())
    if _RC_NOT.search(u) or not _RC_STOP.search(u):
        return None
    m = _RC_CAR.search(u)
    return m.group(1) if m else None


class MechanicChief(Engineer):
    name = "mechanic_chief"
    requires = ("mechanic_telemetry", "mechanic_radio")
    features = ()
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._rc_seen = 0
        self._rc: dict[str, float] = {}  # car -> time of the latest "stopped / slow" message

    def observe(self, state) -> None:
        rc = state.rc
        while self._rc_seen < len(rc):
            m = rc[self._rc_seen]
            self._rc_seen += 1
            car = rc_car_problem(m.message)
            if car:
                self._rc[car] = m.t

    # ------------------------------------------------------------------ lap-time evidence
    def lap_drop(self, car: str) -> tuple[float, float | None]:
        """(evidence 0-1, time it started) from the car's last clean laps, as of now."""
        laps = self.memory.index.by_driver.get(car, {})
        if len(laps) < 5:
            return 0.0, None
        nums = sorted(laps)
        clean = [laps[n] for n in nums if is_clean(laps[n]) and (laps[n].interval is None or laps[n].interval >= 1.2)]
        if len(clean) < 5:
            return 0.0, None
        last = clean[-1]
        # the lap must be among the car's latest two laps, else the evidence has gone stale
        if nums[-1] - last.lap > 1:
            return 0.0, None
        prev = sorted(l.lap_time for l in clean[-6:-1])
        med = prev[len(prev) // 2]
        d1 = last.lap_time - med
        e = min(1.0, max(0.0, (d1 - 2.5) / 3.5))
        since = last.t_end if e > 0 else None
        if len(clean) >= 3:
            d2 = clean[-2].lap_time - sorted(l.lap_time for l in clean[-7:-2])[len(clean[-7:-2]) // 2]
            if d1 >= 1.5 and d2 >= 1.5 and clean[-1].lap - clean[-2].lap <= 2:
                e2 = min(1.0, max(0.0, (min(d1, d2) - 1.0) / 2.5))
                if e2 > e:
                    e, since = e2, clean[-2].t_end
        return round(e, 3), since

    # ------------------------------------------------------------------ outputs
    def car(self, state, number, view):
        tel = view.car("mechanic_telemetry", number)
        rad = view.car("mechanic_radio", number)
        d = state.drivers.get(number)
        out_flag = bool(d is not None and not d.running)
        e_tel = tel.get("mech_risk") or 0.0
        tel_active = bool(tel.get("mech_active"))
        e_tel_eff = e_tel if tel_active else 0.5 * e_tel
        e_rad = rad.get("radio_risk") or 0.0
        e_lap, lap_since = self.lap_drop(number)
        rc_t = self._rc.get(number)
        e_rc = 1.0 if rc_t is not None and state.t - rc_t <= RC_WINDOW_S else 0.0
        parts = {"tel": e_tel_eff, "radio": e_rad, "rc": e_rc, "lap": e_lap}
        prod = 1.0
        for k, e in parts.items():
            prod *= 1.0 - WEIGHTS[k] * e
        risk = 1.0 - prod
        # which problem: the strongest of telemetry / radio
        issue = ""
        if tel.get("mech_issue") and e_tel_eff >= max(e_rad, 0.3):
            issue = tel["mech_issue"]
        elif rad.get("radio_issue") and e_rad >= 0.3:
            issue = rad["radio_issue"]
            issue = {"power": "power_loss", "brakes": "brake_issue"}.get(issue, issue)
        elif tel.get("mech_issue"):
            issue = tel["mech_issue"]
        elif e_rc:
            issue = "slow_car"
        elif e_lap >= 0.3:
            issue = "pace_loss"
        # corroboration: radio and telemetry name the same check
        agree = bool(rad.get("radio_issue")) and _RADIO_TO_CHECK.get(rad["radio_issue"]) == tel.get("mech_issue") and e_tel >= 0.3
        if agree:
            risk = min(1.0, risk + 0.1)
        # since: the earliest of the evidence that is strong enough to count
        sinces = []
        if tel_active and tel.get("mech_since") is not None:
            sinces.append(tel["mech_since"])
        elif e_tel >= 0.3 and tel.get("mech_since") is not None:
            sinces.append(tel["mech_since"])
        if e_rad >= 0.3 and rad.get("radio_since") is not None:
            sinces.append(rad["radio_since"])
        if e_rc:
            sinces.append(rc_t)
        if e_lap > 0 and lap_since is not None:
            sinces.append(lap_since)
        why = []
        if e_tel >= 0.3:
            why.append(f"telemetry {e_tel:.1f}" + (f" ({tel['mech_issue'].replace('_', ' ')})" if tel.get("mech_issue") else ""))
        if e_rad >= 0.3:
            lap = rad.get("radio_lap")
            why.append(f"radio '{rad['radio_quote'][:60]}'" + (f" at lap {lap}" if lap else ""))
        if e_rc:
            why.append("race control: stopped/slow")
        if e_lap > 0:
            why.append(f"lap time drop {e_lap:.1f}")
        return {
            "risk": round(risk, 3), "issue": issue, "since": min(sinces) if sinces else None, "why": "; ".join(why),
            "tel": round(e_tel, 3), "radio": round(e_rad, 3), "lap": e_lap, "rc": e_rc, "out": out_flag,
            "alert": bool(risk >= ALERT_RISK and not out_flag and (issue or why)),
        }

    def alerts(self, state, view):
        out = []
        for n in sorted(state.drivers, key=lambda x: int(x) if x.isdigit() else 999):
            v = view.car(self.name, n)
            if not v["alert"]:
                continue
            d = state.drivers[n]
            label = {"power_loss": "power loss", "brake_issue": "brake issue", "slow_car": "car slow / stopping",
                     "pace_loss": "pace loss", "gearbox": "gearbox trouble"}.get(v["issue"], v["issue"].replace("_", " ") or "problem")
            out.append(Alert(
                state.t, self.name, f"mech_{v['issue'] or 'problem'}", "critical" if v["risk"] >= 0.85 else "warn",
                f"Car {n}{f' ({d.tla})' if d.tla else ''}: {label} suspected ({v['why']})",
                car=n, since=v["since"], data={"risk": v["risk"], "issue": v["issue"]}))
        return out
