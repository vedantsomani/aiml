"""Models engineer: the cross-race models' predictions, live.

``ctx.models`` is a ``modelstore.TrainedBundle`` trained once on races that ended
before this one started (``Context.for_race`` refuses it otherwise). For each car
this engineer builds the decision row exactly as the benchmark does
(``bench.features.base_row`` plus every bench engineer's values) and asks the
bundle:

* ``pit_prob_1`` / ``pit_prob_3``: the row at the car's latest lap end (``lap_end``);
* ``p_stop_le_1/2/3/5/8``, ``laps_to_stop_exp``, ``laps_to_stop_med``: the survival model's distribution of
  the laps to the car's next stop (``bench.laps_to_stop``), from the same lap-end row;
* ``rejoin_pred``: the row of a stop made now (``pit_entry`` at the next lap), the
  position the car would rejoin in.

Without a bundle, or when no decision is due (car in the pits, last lap, red flag),
the values are None. Everything is as of ``state.t``.
"""

from __future__ import annotations

import numpy as np

from ...bench.features import base_row, bench_engineers
from ..engineer import Engineer

STOP_KEYS = ("p_stop_le_1", "p_stop_le_2", "p_stop_le_3", "p_stop_le_5", "p_stop_le_8", "laps_to_stop_exp",
             "laps_to_stop_med")
KEYS = ("pit_prob_1", "pit_prob_3", "rejoin_pred") + STOP_KEYS
_NONE = dict.fromkeys(KEYS)


class ModelsEngineer(Engineer):
    name = "models"
    requires = ("tyre", "pitstop", "rivals", "rules", "weather")  # the bench engineers whose values are model inputs
    in_bench = False  # the benchmark trains its own per race; this is for the live pit wall

    # ------------------------------------------------------------------ rows
    def decision_rows(self, state, number, view):
        """(lap_end row, pit_entry row) for car ``number`` as of now; each None if no decision is due."""
        d = state.drivers.get(number)
        mine = self.memory.index.by_driver.get(number, {})
        if d is None or not mine:
            return None, None
        rec = mine[max(mine)]  # the car's latest completed lap
        extra = None

        def build(kind, **kw):
            nonlocal extra
            row = base_row(self.ctx.prior, self.memory, state, rec, driver=number, kind=kind, **kw)
            if row is None:
                return None
            if extra is None:  # engineer values do not depend on the kind of row
                extra = {}
                for cls in bench_engineers():
                    for k, v in view.car(cls.name, number).items():
                        extra[f"{cls.name}__{k}"] = v
                    for k, v in view.race(cls.name).items():
                        extra[f"{cls.name}__{k}"] = v
            return {**row, **extra}

        lap_end = build("lap_end")
        pit_entry = build("pit_entry", in_lap=rec.lap + 1)  # a stop made now, on the next lap
        return lap_end, pit_entry

    # ------------------------------------------------------------------ values
    def car(self, state, number, view):
        bundle = self.ctx.models
        if bundle is None:
            return dict(_NONE)
        lap_end, pit_entry = self.decision_rows(state, number, view)
        out = dict(_NONE)
        if lap_end is not None:
            for h, key in ((1, "pit_prob_1"), (3, "pit_prob_3")):
                out[key] = _one(bundle, f"pit_within_{h}", lap_end)
            out.update(_stop_distribution(bundle, lap_end))
        if pit_entry is not None:
            out["rejoin_pred"] = _one(bundle, "position_after_stop", pit_entry)
        return out


def _one(bundle, task: str, row: dict):
    if task not in bundle.models:
        return None
    import pandas as pd

    df = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in row.items()}])
    p = float(bundle.predict(task, df)[0])
    return p if np.isfinite(p) else None


def _stop_distribution(bundle, row: dict) -> dict:
    """``p_stop_le_k`` (k = 1, 2, 3, 5, 8), expected and median laps to the next stop; None without the model."""
    if "laps_to_stop" not in bundle.models:
        return {}
    import pandas as pd

    from ...bench.laps_to_stop import summarize

    df = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in row.items()}])
    s = summarize(np.asarray(bundle.models["laps_to_stop"].predict_cdf(df), float))
    out = {k: float(v[0]) for k, v in s.items()}
    return out if all(np.isfinite(v) for v in out.values()) else {}
