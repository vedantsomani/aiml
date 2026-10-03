"""Rival-strategist benchmark (owner: rivals): undercut task, pit_within models, team-habit ablation.

Labels read the finished race (the future); they only score.

* ``y_uc``        lap-end rows: 1 if the car directly behind (``rivals__behind``) stops within 5 laps,
                  before this car, and is ahead of it once both have stopped (positions 3 laps after the
                  later of the two stops; if this car never stops, 3 laps after the chaser's stop).
                  0 if the chaser doesn't stop first in the window, or stops first and stays behind. NaN
                  when no car is behind or positions can't be compared (a car retired).
* ``y_uc_first``  the chaser stops first within 5 laps (1/0).
* ``y_uc_chance`` the same label for this car over the car *ahead* (``rivals__ahead``);
  ``y_uc_first_c`` its "stops first" part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..pitwall.engineers.tyre import TyreEngineer
from ..state import RaceState
from .features import NUMERIC_FEATURES
from .models import BaseRate
from .tasks import Task

UC_HORIZON = 5
UC_SETTLE = 3  # laps after the later stop at which positions are compared
TEAM_COLS = ("rivals__team_cover_rate", "rivals__cover")
SIGNAL_COLS = ("sc", "vsc", "ahead_pit", "behind_pit", "mate_pit", "cover", "cliff", "must", "stuck",
               "threat", "stops", "rem", "ratio")


# ----------------------------------------------------------------------------- labels
class _Race:
    def __init__(self, final: RaceState) -> None:
        self.stops: dict[str, list[int]] = {}
        for pe in final.pit_events:
            if not pe.under_red:
                self.stops.setdefault(pe.driver, []).append(pe.in_lap)
        for v in self.stops.values():
            v.sort()
        self.pos: dict[str, dict[int, int | None]] = {}
        for x in final.laps:
            self.pos.setdefault(x.driver, {})[x.lap] = x.position

    def next_stop(self, drv: str, L: int) -> int | None:
        return next((s for s in self.stops.get(drv, ()) if s > L), None)

    def over(self, att: str, dfd: str, L: int) -> tuple[float, float]:
        """(att stops first within the window, att ends ahead of dfd); NaN when it can't be told."""
        pb = self.next_stop(att, L)
        if pb is None or pb > L + UC_HORIZON:
            return 0.0, 0.0
        pa = self.next_stop(dfd, L)
        if pa is not None and pa <= pb:
            return 0.0, 0.0
        z = max(pb, pa if pa is not None else pb) + UC_SETTLE
        for lap in (z, z - 1, z - 2):
            a, b = self.pos.get(att, {}).get(lap), self.pos.get(dfd, {}).get(lap)
            if a is not None and b is not None:
                return 1.0, float(a < b)
        return 1.0, float("nan")


def rivals_labels(rows: list[dict], final: RaceState) -> None:
    race = _Race(final)
    nan = float("nan")
    for row in rows:
        if row.get("kind", "lap_end") != "lap_end":
            continue
        drv, L = row["driver"], row["lap"]
        beh, ahd = row.get("rivals__behind"), row.get("rivals__ahead")
        if beh:
            row["y_uc_first"], row["y_uc"] = race.over(beh, drv, L)
        else:
            row["y_uc_first"], row["y_uc"] = nan, nan
        if ahd:
            row["y_uc_first_c"], row["y_uc_chance"] = race.over(drv, ahd, L)
        else:
            row["y_uc_first_c"], row["y_uc_chance"] = nan, nan


# ----------------------------------------------------------------------------- tasks
def undercut_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["kind"] == "lap_end") & (df["y_retire_3"] == 0) & (df["track_status"] == "1")]
    d = d[d["rivals__behind"].notna() & d["y_uc"].notna() & d["rivals__gap_behind"].notna()]
    return d.copy()


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan for c in cols},
                        index=df.index).astype(float)


def _k(target: str) -> int:
    return int(target.rsplit("_", 1)[1])


# ----------------------------------------------------------------------------- pit_within models
class RivalsPit:
    """The engineer's own ``pit_prob_k`` (no training)."""

    name = "rivals_pit"
    col = "rivals__pit_prob_{k}"

    def fit(self, df, target):
        self.k = _k(target)
        self.p = float(df[target].mean())
        return self

    def predict(self, df):
        v = _num(df, [self.col.format(k=self.k)]).iloc[:, 0].to_numpy()
        return np.clip(np.where(np.isnan(v), self.p, v), 1e-4, 1 - 1e-4)


class RivalsPrior(RivalsPit):
    """Only the history prior (stint lengths by circuit and compound), no in-race signals."""

    name = "rivals_prior"
    col = "rivals__prior_pit_{k}"


class _Logit:
    """Logistic regression on the history prior's logit plus the in-race signals, trained on earlier races."""

    name = ""
    team = True

    def _x(self, df):
        x = _num(df, [f"rivals__{c}" for c in SIGNAL_COLS])
        x["lp"] = _num(df, [f"rivals__lp_{self.k}"]).iloc[:, 0]
        x["late"] = (_num(df, ["laps_remaining"]).iloc[:, 0] <= self.k + 1).astype(float)
        if not self.team:
            x = x.drop(columns=["rivals__cover"])
        return x

    def fit(self, df, target):
        self.k = _k(target)
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=0.5, max_iter=3000))
        self.model.fit(self._x(df), df[target].astype(int))
        return self

    def predict(self, df):
        return self.model.predict_proba(self._x(df))[:, 1]


class RivalsLogit(_Logit):
    name = "rivals_logit"


class RivalsLogitNoTeam(_Logit):
    name = "rivals_logit_noteam"
    team = False


# inputs of the learned models: everything gbm_hazard sees (base + tyre engineer's keys) plus the rivals' keys
BASE_COLS = ("tyre_age", "laps_in_stint", "race_frac", "laps_remaining", "must_stop", "sc", "vsc", "position",
             "interval_ahead", "gap_behind", "n_pitted_recent", "ahead_pitted_recent", "behind_pitted_recent",
             "pit_stops", "cmp_soft", "cmp_medium", "cmp_hard")
RIVAL_COLS = ("rivals__pit_prob_1", "rivals__pit_prob_3", "rivals__pit_prob_5", "rivals__ratio", "rivals__mate_pit",
              "rivals__undercut_threat", "rivals__undercut_chance")
TYRE_COLS = tuple(f"tyre__{k}" for k in TyreEngineer.features)


class _GBM:
    name = ""
    team = True
    params = dict(max_iter=150, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=200,
                  l2_regularization=10.0, random_state=0)  # as gbm_hazard (2025 favoured nothing larger)

    def _cols(self, df):
        return NUMERIC_FEATURES + list(TYRE_COLS) + list(RIVAL_COLS + (TEAM_COLS if self.team else ()))

    def fit(self, df, target):
        self.cols = self._cols(df)
        self.model = HistGradientBoostingClassifier(**self.params).fit(_num(df, self.cols), df[target].astype(int))
        return self

    def predict(self, df):
        return self.model.predict_proba(_num(df, self.cols))[:, 1]


class RivalsGBM(_GBM):
    """Boosted trees on the core situation, the tyre engineer's keys and the rivals' probabilities and signals."""

    name = "rivals_gbm"


class RivalsGBMNoTeam(_GBM):
    """The same without the team-habit columns (ablation)."""

    name = "rivals_gbm_noteam"
    team = False


class RivalsBlend:
    """Average (in logit space) of the engineer's ``pit_prob_k`` and a boosted model trained on earlier races."""

    name = "rivals_blend"

    def fit(self, df, target):
        self.a = RivalsPit().fit(df, target)
        self.b = RivalsGBM().fit(df, target)
        return self

    def predict(self, df):
        def lg(p):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return np.log(p / (1 - p))

        return 1 / (1 + np.exp(-0.5 * (lg(self.a.predict(df)) + lg(self.b.predict(df)))))


# ----------------------------------------------------------------------------- undercut models
class RivalsThreat:
    """The engineer's ``undercut_threat`` (no training)."""

    name = "rivals_threat"
    col = "rivals__undercut_threat"

    def fit(self, df, target):
        self.p = float(df[target].mean())
        return self

    def predict(self, df):
        v = _num(df, [self.col]).iloc[:, 0].to_numpy()
        return np.clip(np.where(np.isnan(v), self.p, v), 1e-4, 1 - 1e-4)


class GapLogit:
    """Logistic regression on the gap to the car behind and tyre age only (no engineer values)."""

    name = "gap_logit"
    cols = ("gap_behind", "tyre_age", "behind_tyre_age", "laps_remaining", "pit_loss_now")

    def _x(self, df):
        x = _num(df, self.cols)
        x["gap_small"] = (x["gap_behind"] < 3).astype(float)
        return x

    def fit(self, df, target):
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=0.5, max_iter=3000))
        self.model.fit(self._x(df), df[target].astype(int))
        return self

    def predict(self, df):
        return self.model.predict_proba(self._x(df))[:, 1]


class UndercutGBM(_GBM):
    """Boosted trees on the core situation plus margins, gaps and pit probabilities."""

    name = "undercut_gbm"
    params = dict(max_iter=150, learning_rate=0.03, max_leaf_nodes=6, min_samples_leaf=150,
                  l2_regularization=10.0, random_state=0)
    extra = ("rivals__uc_margin_threat", "rivals__uc_first_threat", "rivals__gap_behind", "behind_tyre_age",
             "pit_loss_now", "rivals__team_cover_rate")

    def _cols(self, df):
        return list(BASE_COLS + RIVAL_COLS + self.extra)


UC_MODELS = (BaseRate, GapLogit, RivalsThreat, UndercutGBM)


def rivals_tasks(**_) -> list[Task]:
    return [Task("undercut_5", "binary", "y_uc", undercut_rows, UC_MODELS,
                 keep=("rivals__gap_behind", "rivals__uc_margin_threat"))]
