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
    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._memo: dict[tuple, dict] = {}  # row content -> predictions (a model is a pure function of its row)
        self._pre: tuple | None = None  # (view, {car: values}) from prefetch

    def prefetch(self, state, numbers, view) -> None:
        """Predict for every car in one batch per task (the snapshot asks for them all)."""
        if self.ctx.models is None:
            return
        rows = {n: self.decision_rows(state, n, view) for n in numbers}
        self._fill(self.ctx.models, [p[0] for p in rows.values() if p[0] is not None],
                   [p[1] for p in rows.values() if p[1] is not None])
        self._pre = (view, {n: self._values(*pair) for n, pair in rows.items()})

    def car(self, state, number, view):
        bundle = self.ctx.models
        if bundle is None:
            return dict(_NONE)
        if self._pre is not None and self._pre[0] is view and number in self._pre[1]:
            return dict(self._pre[1][number])
        lap_end, pit_entry = self.decision_rows(state, number, view)
        self._fill(bundle, [lap_end] if lap_end is not None else [], [pit_entry] if pit_entry is not None else [])
        return self._values(lap_end, pit_entry)

    def _values(self, lap_end, pit_entry) -> dict:
        out = dict(_NONE)
        if lap_end is not None:
            out.update(self._memo[("lap_end", _key(lap_end))])
        if pit_entry is not None:
            out.update(self._memo[("pit_entry", _key(pit_entry))])
        return out

    def _fill(self, bundle, lap_end: list[dict], pit_entry: list[dict]) -> None:
        """Memoise the predictions of every row; rows not seen before go through the models in one batch per task."""
        new: dict[tuple, dict] = {}
        for kind, rows in (("lap_end", lap_end), ("pit_entry", pit_entry)):
            for r in rows:
                k = (kind, _key(r))
                if k not in self._memo:
                    new[k] = r
        le = [k for k in new if k[0] == "lap_end"]
        pe = [k for k in new if k[0] == "pit_entry"]
        if le:
            df = _frame([new[k] for k in le])
            p1, p3 = _batch(bundle, "pit_within_1", df), _batch(bundle, "pit_within_3", df)
            stop = _stop_batch(bundle, df)
            for i, k in enumerate(le):
                self._memo[k] = {"pit_prob_1": p1[i], "pit_prob_3": p3[i], **stop[i]}
        if pe:
            rej = _batch(bundle, "position_after_stop", _frame([new[k] for k in pe]))
            for i, k in enumerate(pe):
                self._memo[k] = {"rejoin_pred": rej[i]}


def _key(row: dict) -> tuple:
    return tuple((k, None if v is None or v != v else v) for k, v in row.items())


def _frame(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame([{k: (np.nan if v is None else v) for k, v in r.items()} for r in rows])


def _batch(bundle, task: str, df) -> list:
    if task not in bundle.models:
        return [None] * len(df)
    p = bundle.predict(task, df)
    return [float(x) if np.isfinite(x) else None for x in p]


def _stop_batch(bundle, df):
    """Per row: ``p_stop_le_k`` (k = 1, 2, 3, 5, 8), expected and median laps to the next stop ({} if not finite);
    {} without the model."""
    if "laps_to_stop" not in bundle.models:
        return [{}] * len(df)
    from ...bench.laps_to_stop import summarize

    s = summarize(np.asarray(bundle.models["laps_to_stop"].predict_cdf(df), float))
    out = []
    for i in range(len(df)):
        d = {k: float(v[i]) for k, v in s.items()}
        out.append(d if all(np.isfinite(v) for v in d.values()) else {})
    return out
