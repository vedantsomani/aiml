"""Sporting / rules engineer: what the regulations allow, require and punish right now.

STUB (owner: rules engineer). Contract keys (docs/ENGINEERING.md) are all
present; the stub knows the two-dry-compound rule and the track status. To
build: race-control messages as structured events (penalties and whether
they're served, pit lane closed, SC / VSC phases, red-flag rules), per-season
regulation parameters, constraints for the strategy engineer.
"""

from __future__ import annotations

from ...config import DRY_COMPOUNDS
from ..engineer import Engineer

_PHASES = {"4": "sc", "5": "red", "6": "vsc", "7": "vsc_ending"}


class RulesEngineer(Engineer):
    name = "rules"

    def race(self, state, view):
        used = {c for d in state.drivers.values() for c in d.compounds_used}
        return {
            "race_dry": bool(used) and used <= set(DRY_COMPOUNDS),
            "sc_phase": _PHASES.get(state.track_status, "none"),
            "pit_lane_open": None,
        }

    def car(self, state, number, view):
        d = state.drivers.get(number)
        if d is None:
            return {}
        mine = set(d.compounds_used) & set(DRY_COMPOUNDS)
        return {
            "must_stop": bool(view.race(self.name)["race_dry"] and len(mine) < 2),
            "penalty_s_pending": None,
            "drive_through_pending": None,
        }
