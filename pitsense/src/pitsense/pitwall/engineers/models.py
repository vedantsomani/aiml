"""Models engineer: predictions from the cross-race models, live.

STUB (owner: ML platform engineer). The benchmark trains models per test race
on earlier races only; on the pit wall the same models come pre-trained in
``ctx.models`` (trained before this race started - ``Context.for_race`` checks).
To build: a trained-model bundle (``pitsense train``) and predictions from rows
built exactly like the benchmark's, so live and backtest numbers agree.
"""

from __future__ import annotations

from ..engineer import Engineer


class ModelsEngineer(Engineer):
    name = "models"
    in_bench = False  # the benchmark trains its own per race; this is for the live pit wall

    def car(self, state, number, view):
        return {"pit_prob_1": None, "pit_prob_3": None, "rejoin_pred": None}
