"""Pit-stop engineer benchmark: rejoin position after a stop, and the stop's time loss.

``position_after_stop`` models (registered in ``registry.TASK_MODELS``) are scored
next to the core ones. ``pit_loss`` is a regression task: at pit entry, predict
what the stop will cost, measured on the finished race by the time-aligned method
(``pitloss.measure_stop``) and scored overall and by track status during the stop.

Labels read the finished race and live here, never in ``pitwall/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..pitloss import LapIndex, measure_stop
from .evaluate import EvalResult, evaluate_task
from .metrics import regression_scores
from .tasks import Task

PS = (
    "cars_within_loss", "cars_within_stopping", "passers_expected", "margin_s", "rejoin_gap_ahead_s",
    "rejoin_gap_behind_s", "loss_if_box_now", "stationary_s", "loss_now", "loss_now_sd", "pass_through",
    "phase_age_s", "stops_measured", "cars_in_pit_lane",
)
LOSS_COLS = (
    "pitstop__loss_now", "pitstop__loss_green", "pitstop__loss_sc", "pitstop__loss_vsc", "pitstop__loss_now_sd",
    "pitstop__loss_if_box_now", "pitstop__stationary_s", "pitstop__stationary_field_s", "pitstop__lane_time_s",
    "pitstop__loss_prior_green", "pitstop__loss_drift_s", "pitstop__stops_measured", "pitstop__cars_in_pit_lane",
    "pitstop__pass_through", "pitstop__phase_age_s", "sc", "vsc", "status_age_s", "track_temp", "position",
    "n_running", "lap", "race_frac", "tyre_age", "pit_loss_now",
)


# ----------------------------------------------------------------------------- labels (the finished race)
def pitstop_labels(rows: list[dict], final) -> None:
    """``y_pit_loss`` / ``y_pit_cond``: the stop's measured loss, for pit entries whose stop is typical."""
    idx = LapIndex(final.laps)
    log = list(final.status_log)
    cache: dict[int, object] = {}
    for row in rows:
        if row.get("kind") != "pit_entry":
            continue
        i = row["pit_event"]
        if i not in cache:
            cache[i] = measure_stop(idx, final.pit_events[i], log)
        m = cache[i]
        ok = m is not None and m.typical
        row["y_pit_loss"] = m.loss if ok else float("nan")
        row["y_pit_cond"] = m.condition if ok else ""


# ----------------------------------------------------------------------------- position after a stop
def _base(d: pd.DataFrame) -> np.ndarray:
    """The engineer's rejoin position; where the whole field drives through the pit lane, nobody changes places."""
    x = d["pitstop__rejoin_if_box_now"].to_numpy(float).copy()
    through = d["pitstop__pass_through"].to_numpy(bool)
    x[through] = d["position"].to_numpy(float)[through]
    return x


class PitstopRule:
    """The pit-stop engineer's own answer, no training."""

    name = "pitstop_rule"

    def fit(self, df, target):
        return self

    def predict(self, df):
        return np.clip(_base(df), 1, df["n_running"].to_numpy(float))


class PitstopGBM:
    """Learns the correction to the engineer's rejoin position from the engineer's own values only.

    Small and heavily regularised: on 2025 this beat adding the base features (which overfit ~600 rows).
    """

    name = "pitstop_gbm"

    def fit(self, df, target):
        self.cols = [f"pitstop__{k}" for k in PS]
        self.model = HistGradientBoostingRegressor(
            max_iter=80, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=50,
            l2_regularization=10.0, loss="absolute_error", random_state=0,
        )
        self.model.fit(df[self.cols].astype(float), df[target] - _base(df))
        return self

    def predict(self, df):
        pred = _base(df) + np.round(self.model.predict(df[self.cols].astype(float)))
        return np.clip(pred, 1, df["n_running"].to_numpy(float))


REJOIN_MODELS = (PitstopRule, PitstopGBM)


# ----------------------------------------------------------------------------- pit loss
def loss_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["kind"] == "pit_entry") & df["y_pit_loss"].notna()].copy()


class V01Prior:
    """v0.1: the circuit/season prior blended with this race's stops by lap-number measurement (``pit_loss_now``)."""

    name = "v01_prior_shrink"

    def fit(self, df, target):
        return self

    def predict(self, df):
        return df["pit_loss_now"].to_numpy(float)


class PitstopLoss:
    """The engineer's loss now plus the team's stationary time against the field."""

    name = "pitstop_loss"

    def fit(self, df, target):
        return self

    def predict(self, df):
        return df["pitstop__loss_if_box_now"].to_numpy(float)


class PitstopLossGBM:
    """Small, regularised correction to the engineer's loss from its values and the track state."""

    name = "pitstop_loss_gbm"

    def fit(self, df, target):
        self.model = HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=30,
            l2_regularization=5.0, loss="absolute_error", random_state=0,
        )
        self.cols = [c for c in LOSS_COLS if df[c].notna().sum() > 1]
        self.model.fit(self._x(df), df[target] - df["pitstop__loss_if_box_now"])
        return self

    def _x(self, df):
        return df[self.cols].astype(float)

    def predict(self, df):
        return df["pitstop__loss_if_box_now"].to_numpy(float) + self.model.predict(self._x(df))


LOSS_MODELS = (V01Prior, PitstopLoss, PitstopLossGBM)


def _evaluate_loss(df: pd.DataFrame, *, test_year: int, min_train_races: int) -> EvalResult:
    """Overall and by what the track was doing during the stop."""
    plain = Task("pit_loss", "regression", "y_pit_loss", loss_rows, LOSS_MODELS, keep=("y_pit_cond", "pitstop__loss_condition"))
    r = evaluate_task(plain, df, test_year, min_train_races)
    P, names = r.predictions, [m.name for m in LOSS_MODELS]
    parts = []
    for split, g in [("all", P)] + [(c, P[P["y_pit_cond"] == c]) for c in ("green", "sc", "vsc")]:
        if len(g):
            parts += [{"split": split, "model": n, **regression_scores(g["y_pit_loss"], g[n])} for n in names]
    board = pd.DataFrame(parts)
    return EvalResult("pit_loss", board, r.per_race, P, {})


def pitstop_tasks(**_) -> list[Task]:
    return [Task("pit_loss", "regression", "y_pit_loss", loss_rows, LOSS_MODELS, evaluate=_evaluate_loss)]
