"""The virtual pit wall: one engineer per role on a race-strategy team.

    engineers/tyre      tyres and pace: degradation, cliff, fuel-corrected pace
    engineers/pitstop   pit loss now, where a car rejoins
    engineers/rivals    when rivals will stop, undercut / overcut threats
    engineers/rules     sporting regulations: compound rule, SC/VSC, penalties
    engineers/weather   rain risk, slick / intermediate crossover
    engineers/models    live predictions from the cross-race models (pre-trained)
    engineers/strategy  race simulation, Plan A / Plan B with triggers
    engineers/head      head of strategy: one call per car, with the reasons

Every engineer follows the same RaceState as the rest of PitSense and answers
with plain values as of now (see ``engineer.py``). ``PitWall`` runs them in
dependency order; the benchmark puts their values into every decision row, so
the corrupted-future test checks all of them.
"""

from .engineer import Context, Engineer, View
from .memory import RaceMemory
from .types import Alert, Call, Plan, PlanStop, Reason, Snapshot, TeamConfig
from .wall import PitWall

__all__ = [
    "Alert", "Call", "Context", "Engineer", "PitWall", "Plan", "PlanStop", "RaceMemory",
    "Reason", "Snapshot", "TeamConfig", "View",
]
