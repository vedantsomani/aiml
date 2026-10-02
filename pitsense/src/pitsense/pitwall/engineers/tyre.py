"""Tyre & performance engineer: how fast each car is on its tyres, and how fast that fades.

STUB (owner: tyre engineer). Contract keys (docs/ENGINEERING.md) are all
present; the stub fills the cheap ones from this stint's clean laps and leaves
the rest None. To build: fuel-corrected pace, degradation, tyre-cliff risk, and
pace on a fresh set of each compound.
"""

from __future__ import annotations

from statistics import median

import numpy as np

from ..engineer import Engineer
from ..memory import is_clean


class TyreEngineer(Engineer):
    name = "tyre"

    def car(self, state, number, view):
        d = state.drivers.get(number)
        if d is None:
            return {}
        laps = self.memory.index.by_driver.get(number, {})
        clean = [x for k, x in sorted(laps.items()) if k > d.stint_start_lap and is_clean(x)]
        times = [x.lap_time for x in clean]
        slope = float(np.polyfit([x.lap for x in clean], times, 1)[0]) if len(times) >= 4 else None
        pace = round(median(times), 3) if times else None
        return {
            "stint_pace": pace,
            "stint_clean_laps": len(times),
            "pace_s": pace,  # next clean lap on these tyres; stub: stint median (no fuel or wear)
            "deg_s_per_lap": round(slope, 4) if slope is not None else None,  # stub: raw slope, fuel not removed
            "fresh_soft_s": None,
            "fresh_medium_s": None,
            "fresh_hard_s": None,
            "cliff_risk": None,
        }

    def race(self, state, view):
        return {"fuel_s_per_lap": None}
