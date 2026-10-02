"""Strategy engineer: simulate the rest of the race and rank the plans.

STUB (owner: strategy engineer) - no simulation yet.
To build: a lap-by-lap race simulator that tests every candidate plan against
the same simulated futures, and Plan A / Plan B with the triggers that switch
between them. Slow, so it runs on the pit wall but not per benchmark row.
"""

from __future__ import annotations

from ..engineer import Engineer


class StrategyEngineer(Engineer):
    name = "strategy"
    requires = ("tyre", "pitstop", "rivals", "rules", "weather", "models")
    in_bench = False
