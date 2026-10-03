"""Sporting / rules engineer: what the regulations allow, require and punish right now.

Reads race control (``state.rc``) through a typed parser (``parser.py``),
keeps penalty and investigation books (``book.py``) and applies the per-season
regulation table (``regs.py``). Docs: docs/engineers/rules.md.
"""

from __future__ import annotations

from ....config import DRY_COMPOUNDS
from ...engineer import Engineer
from ...types import Alert
from .book import RCBook
from .parser import parse
from .regs import RegRule, rule_for

__all__ = ["RulesEngineer", "RCBook", "parse", "RegRule", "rule_for"]


class RulesEngineer(Engineer):
    name = "rules"
    # Lean set, chosen on 2025 (docs/engineers/rules.md): the noisy counts add variance and no skill.
    features = (
        "must_stop", "penalty_s_pending", "drive_through_pending", "stint_laps_left_f",
        "sc_ending", "pit_lane_open",
    )

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self.book = RCBook()
        self.rule: RegRule | None = rule_for(ctx.meta)
        self._n = 0  # race-control messages consumed so far

    # ------------------------------------------------------------------ bookkeeping
    def _sync(self, state) -> None:
        rc = state.rc
        while self._n < len(rc):
            m = rc[self._n]
            self._n += 1
            self.book.add(m.t, parse(m.message, m.category, m.flag))

    def observe(self, state) -> None:
        self._sync(state)

    # ------------------------------------------------------------------ race values
    def sc_phase(self, state) -> str:
        b, since = self.book, state.track_status_since
        s = state.track_status
        if s == "5":
            return "red"
        if s == "4":
            return "sc_ending" if b.sc_in_t is not None and b.sc_in_t >= since else "sc"
        if s == "6":
            return "vsc_ending" if b.vsc_ending_t is not None and b.vsc_ending_t >= since else "vsc"
        if s == "7":
            return "vsc_ending"
        return "none"

    def race(self, state, view):
        self._sync(state)
        b = self.book
        used = {c for d in state.drivers.values() for c in d.compounds_used}
        phase = self.sc_phase(state)
        return {
            "race_dry": bool(used) and used <= set(DRY_COMPOUNDS),
            "sc_phase": phase,
            "sc_ending": phase in ("sc_ending", "vsc_ending"),
            "pit_lane_open": b.pit_entry_open,
            "pit_exit_open": b.pit_exit_open,
            "n_sc": b.n_sc + b.n_vsc,
            "n_red": b.n_red,
            "rc_rain_risk": b.rain_risk,
            "rain_risk_f": -1.0 if b.rain_risk is None else b.rain_risk,  # model input: -1 = no forecast yet
            "drs_enabled": b.drs_enabled,
            "low_grip": b.grip_low,
            "chequered": b.chequered_t is not None,
            "rc_messages": b.total,
            "rc_unknown": b.unknown,
            "reg_min_stops": self.rule.min_stops if self.rule else None,
            "reg_max_stint_laps": self.rule.max_stint_laps if self.rule else None,
        }

    # ------------------------------------------------------------------ car values
    def car(self, state, number, view):
        d = state.drivers.get(number)
        if d is None:
            return {}
        self._sync(state)
        b, rule = self.book, self.rule
        race_dry = view.race(self.name)["race_dry"]
        dry = set(d.compounds_used) & set(DRY_COMPOUNDS)
        stops = sum(1 for p in state.pit_events if p.driver == number and not p.under_red)
        total = state.total_laps or 0

        stint_left = None
        if rule is not None and rule.max_stint_laps:
            stint_left = max(rule.max_stint_laps - max(d.laps - d.stint_start_lap, 0), 0)

        need = False
        if race_dry:
            if len(dry) < 2 and (rule is None or rule.two_compounds):
                need = True
            if rule is not None and rule.min_stops and stops < rule.min_stops:
                need = True
        if stint_left is not None and total and (total - d.laps) > stint_left:
            need = True

        return {
            "must_stop": bool(need),
            "stint_laps_left": stint_left,
            "stint_laps_left_f": 999 if stint_left is None else stint_left,  # model input: 999 = no limit
            "penalty_s_pending": b.time_pending_s(number),
            "drive_through_pending": b.drive_pending(number),
            "penalties_issued": b.issued[number],
            "under_investigation": b.open_inv[number] > 0,
            "post_race_investigation": b.after_inv[number] > 0,
            "incidents_noted": b.noted[number],
            "track_limits_deleted": b.tl_deleted[number],
            "black_white_flag": number in b.bw,
            "reprimands": b.reprimands[number],
        }

    # ------------------------------------------------------------------ alerts
    def alerts(self, state, view):
        self._sync(state)
        b, t = self.book, state.t
        out: list[Alert] = []
        for car in sorted(b.pending):
            for pen, sec, reason, t0 in b.pending[car]:
                what = f"{sec} s time penalty" if pen == "time" else f"{sec or ''} s stop-go penalty".strip() if pen == "stop_go" and sec else pen.replace("_", "-") + " penalty"
                out.append(Alert(t, self.name, "penalty_pending", "warn",
                                 f"Car {car}: {what} still to serve" + (f" ({reason.lower()})" if reason else ""),
                                 car=car, since=t0, data={"pen": pen, "seconds": sec}))
        if not b.pit_entry_open:
            out.append(Alert(t, self.name, "pit_lane_closed", "warn", "Pit entry is closed",
                             since=b.pit_entry_since))
        phase = self.sc_phase(state)
        if phase in ("sc_ending", "vsc_ending"):
            since = b.sc_in_t if phase == "sc_ending" else b.vsc_ending_t
            out.append(Alert(t, self.name, "sc_ending", "info",
                             "Safety car coming in this lap" if phase == "sc_ending" else "VSC ending",
                             since=since))
        if phase == "red":
            out.append(Alert(t, self.name, "red_flag", "critical", "Red flag: race suspended", since=b.red_t))
        for car in sorted(c for c, n in b.open_inv.items() if n > 0):
            out.append(Alert(t, self.name, "under_investigation", "info", f"Car {car} under investigation", car=car))
        return out
