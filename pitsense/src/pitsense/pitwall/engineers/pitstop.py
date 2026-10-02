"""Pit-stop engineer: what a stop costs right now, and where the car would rejoin.

STUB (owner: pit-stop engineer) around the v0.1 logic in ``pitloss.py``: the
circuit prior blended with this race's measured stops, and the gap-minus-pit-loss
rejoin estimate.
"""

from __future__ import annotations

from ...pitloss import cars_within, shrink
from ..engineer import Engineer


class PitStopEngineer(Engineer):
    name = "pitstop"

    def race(self, state, view):
        losses = self.memory.in_race_losses(state)
        p = self.ctx.prior
        green, sc, vsc = shrink(p.green, losses["green"]), shrink(p.sc, losses["sc"]), shrink(p.vsc, losses["vsc"])
        status = state.track_status
        now = sc if status == "4" else vsc if status in ("6", "7") else green
        return {
            "loss_green": round(green, 3),
            "loss_sc": round(sc, 3),
            "loss_vsc": round(vsc, 3),
            "loss_now": round(now, 3),
            "stops_measured": sum(len(v) for v in losses.values()),
        }

    def car(self, state, number, view):
        order = [x for x in state.running_order() if x.running]
        i = next((k for k, x in enumerate(order) if x.number == number), None)
        if i is None:
            return {}
        within, margin = cars_within(order, i, view.race(self.name)["loss_now"])
        pos = order[i].position
        return {
            "rejoin_if_box_now": pos + within if pos is not None else None,
            "cars_within_loss": within,
            "margin_s": margin,
        }
