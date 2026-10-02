"""Baselines and first contenders.

Every model is fit only on races that finished before the test race starts
(the evaluator enforces that). A model is a class with a ``name`` and
``fit(train_df, target) -> self`` / ``predict(test_df) -> np.ndarray``; add it
to a task's model list (bench/tasks.py). The first model of a task is its
reference baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns


# ----------------------------------------------------------------------------- pit within k laps
class BaseRate:
    """Climatology: the share of decision points followed by a stop."""

    name = "base_rate"

    def fit(self, df: pd.DataFrame, target: str) -> "BaseRate":
        self.p = float(df[target].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.p)


class TyreAgeLogit:
    """Logistic regression on tyre age, compound and race progress only."""

    name = "tyre_age_logit"
    cols = ["tyre_age", "laps_in_stint", "race_frac", "laps_remaining", "must_stop",
            "cmp_soft", "cmp_medium", "cmp_hard", "cmp_intermediate", "cmp_wet"]

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.cols].astype(float).copy()
        x["tyre_age_sq"] = x["tyre_age"] ** 2
        return x

    def fit(self, df: pd.DataFrame, target: str) -> "TyreAgeLogit":
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=1.0, max_iter=2000))
        self.model.fit(self._x(df), df[target].astype(int))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._x(df))[:, 1]


class GBMHazard:
    """Gradient boosting on all as-of features (tyres, gaps, rivals' stops, track status).

    Hyperparameters were chosen on 2025 races only (2026 stays a clean test set):
    small, heavily regularised trees beat larger ones and isotonic recalibration.
    """

    name = "gbm_hazard"

    def fit(self, df: pd.DataFrame, target: str) -> "GBMHazard":
        self.cols = feature_columns()
        self.model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=200,
            l2_regularization=10.0, random_state=0,
        )
        self.model.fit(df[self.cols].astype(float), df[target].astype(int))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.cols].astype(float))[:, 1]


PIT_MODELS = (BaseRate, TyreAgeLogit, GBMHazard)


# ----------------------------------------------------------------------------- position after a stop
class NoChange:
    name = "no_change"

    def fit(self, df: pd.DataFrame, target: str) -> "NoChange":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["position"].to_numpy(float)


class GapMinusPitLoss:
    """Broadcast-style 'pit window': everyone within one pit loss behind gets past."""

    name = "gap_minus_pitloss"

    def fit(self, df: pd.DataFrame, target: str) -> "GapMinusPitLoss":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["position"] + df["cars_within_pitloss"]).to_numpy(float)


class RejoinGBM:
    """Learns the correction to gap-minus-pit-loss (traffic, cars stopping too, SC)."""

    name = "rejoin_gbm"

    def fit(self, df: pd.DataFrame, target: str) -> "RejoinGBM":
        self.cols = feature_columns()
        self.model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, loss="absolute_error", random_state=0,
        )
        y = df[target] - df["position"] - df["cars_within_pitloss"]
        self.model.fit(df[self.cols].astype(float), y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        base = (df["position"] + df["cars_within_pitloss"]).to_numpy(float)
        pred = base + np.round(self.model.predict(df[self.cols].astype(float)))
        return np.clip(pred, 1, df["n_running"].to_numpy(float))


REJOIN_MODELS = (NoChange, GapMinusPitLoss, RejoinGBM)
