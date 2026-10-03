"""Tyre & pace benchmark (owner: tyre): labels, tasks and models for the tyre engineer.

Labels read the finished race (the future); they are only used to score.

* ``y_next_lap``    lap-end rows: time of lap L+1 if it is clean (timed, green, not an
                    in/out-lap) and on the same set.
* ``y_lap_5``       lap-end rows: time of lap L+5 if laps L+1..L+5 are all clean and on
                    the same set (no stop in between).
* ``y_cliff_3``     lap-end rows: 1 if the car's tyres fall off within 3 laps, 0 if they
                    demonstrably don't, NaN when it can't be seen (stop, SC, end of race).
                    "Fall off" = on two consecutive laps j, j+1 with j in L+1..L+3 the car
                    is >= 1.0 s slower than its reference (median of its last 3 clean laps
                    up to L on this set), after removing the field's common change (median
                    over cars that stayed on the same set from L-2 to j), and it started
                    lap j in free air (>= 1 s behind the car ahead), so a car held up in
                    traffic or a one-lap mistake doesn't count. 0 needs laps L+1..L+4 clean
                    and on the same set.
* ``y_fresh_lap``   pit-entry rows: time of the first flying lap on the new set (in-lap
                    + 2) if it is clean and the set is new; ``y_fresh_compound`` is the
                    compound fitted (read at the end of that stint, when it is confirmed).
"""

from __future__ import annotations

from statistics import median

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..config import DRY_COMPOUNDS
from ..state import LapRecord, RaceState
from .tasks import Task, lap_end_rows

CLIFF_S = 1.0  # sudden loss that counts as falling off, s/lap
CLIFF_LAPS = 3  # horizon of the cliff label
FREE_AIR_S = 1.0  # a lap started closer than this to the car ahead may be held up
MIN_FIELD = 5  # cars needed to measure the field's common change
RACING = 1.25  # a green-flag lap slower than this times the race's median clean lap is neutralised, not racing


def _clean(x: LapRecord | None) -> bool:
    return (x is not None and x.lap_time is not None and x.lap > 1 and not x.is_in_lap and not x.is_out_lap
            and x.track_status == "1")


class _Race:
    """Final laps and stops of one race, indexed for labelling."""

    def __init__(self, final: RaceState) -> None:
        self.laps: dict[str, dict[int, LapRecord]] = {}
        for x in final.laps:
            self.laps.setdefault(x.driver, {})[x.lap] = x
        self.stops: dict[str, list[int]] = {}
        for pe in final.pit_events:  # every pit entry, red flag included: tyres may change
            self.stops.setdefault(pe.driver, []).append(pe.in_lap)
        self._common: dict[tuple[int, int], float | None] = {}
        times = [x.lap_time for x in final.laps if _clean(x)]
        self.limit = RACING * median(times) if times else float("inf")

    def clean(self, x: LapRecord | None) -> bool:
        """A clean lap that is also racing pace (some neutralised laps are published under green)."""
        return _clean(x) and x.lap_time <= self.limit

    def same_set(self, drv: str, a: int, b: int) -> bool:
        """Laps a..b (a <= b) were driven on one set: no stop ended a lap in [a, b-1]."""
        return not any(a <= s < b for s in self.stops.get(drv, ()))

    def ref(self, drv: str, L: int, n: int = 3, back: int = 5) -> float | None:
        """Median of the car's last ``n`` clean laps up to L on the set it has at L."""
        laps = self.laps.get(drv, {})
        vals = []
        for k in range(L, max(1, L - back), -1):
            if not self.same_set(drv, k, L):
                break
            if self.clean(laps.get(k)):
                vals.append(laps[k].lap_time)
            if len(vals) == n:
                break
        return median(vals) if len(vals) >= 2 else None

    def common(self, L: int, j: int) -> float | None:
        """The field's change from its reference at L to lap j, over cars that stayed out."""
        key = (L, j)
        if key not in self._common:
            deltas = []
            for drv, laps in self.laps.items():
                x = laps.get(j)
                if not self.clean(x) or not self.same_set(drv, L - 2, j):
                    continue
                r = self.ref(drv, L, back=3)
                if r is not None:
                    deltas.append(x.lap_time - r)
            self._common[key] = median(deltas) if len(deltas) >= MIN_FIELD else None
        return self._common[key]

    def free_air(self, drv: str, j: int) -> bool:
        prev = self.laps.get(drv, {}).get(j - 1)
        return prev is None or prev.position == 1 or prev.interval is None or prev.interval >= FREE_AIR_S

    def cliff(self, drv: str, L: int) -> float:
        laps = self.laps.get(drv, {})
        r = self.ref(drv, L)
        if r is None:
            return float("nan")
        flags = []  # per lap L+1..L+CLIFF_LAPS+1: fell off? (None = can't tell)
        for j in range(L + 1, L + CLIFF_LAPS + 2):
            x = laps.get(j)
            c = self.common(L, j)
            if not self.clean(x) or not self.same_set(drv, L, j) or c is None:
                break
            flags.append(x.lap_time - r - c >= CLIFF_S and self.free_air(drv, j))
        for k in range(min(CLIFF_LAPS, len(flags) - 1)):
            if flags[k] and flags[k + 1]:
                return 1.0
        return 0.0 if len(flags) == CLIFF_LAPS + 1 else float("nan")

    def ahead(self, drv: str, L: int, h: int) -> float:
        """Time of lap L+h if laps L+1..L+h are all clean and on the set the car has at L."""
        laps = self.laps.get(drv, {})
        if not self.same_set(drv, L, L + h):
            return float("nan")
        if not all(self.clean(laps.get(L + k)) for k in range(1, h + 1)):
            return float("nan")
        return float(laps[L + h].lap_time)

    def fresh(self, pe) -> tuple[float, str]:
        laps = self.laps.get(pe.driver, {})
        nxt = [s for s in self.stops.get(pe.driver, ()) if s > pe.in_lap]
        end = min(nxt) if nxt else max(laps, default=pe.in_lap)
        seg = [laps[k] for k in range(pe.in_lap + 1, end + 1) if k in laps]
        if not seg or pe.out_lap != pe.in_lap + 1:
            return float("nan"), ""
        last = seg[-1]
        compound = last.compound or ""
        new_set = last.tyre_age is not None and last.tyre_age == last.lap - pe.in_lap
        x = laps.get(pe.in_lap + 2)
        if not new_set or compound not in DRY_COMPOUNDS or not self.clean(x) or pe.in_lap + 2 > end:
            return float("nan"), compound
        return float(x.lap_time), compound


def tyre_labels(rows: list[dict], final: RaceState) -> None:
    race = _Race(final)
    for row in rows:
        drv, L = row["driver"], row["lap"]
        if row.get("kind", "lap_end") == "pit_entry":
            y, comp = race.fresh(final.pit_events[row["pit_event"]])
            row["y_fresh_lap"], row["y_fresh_compound"] = y, comp
            continue
        row["y_next_lap"] = race.ahead(drv, L, 1)
        row["y_lap_5"] = race.ahead(drv, L, 5)
        row["y_cliff_3"] = race.cliff(drv, L)


# ----------------------------------------------------------------------------- selections
def _dry_lap_end(df: pd.DataFrame) -> pd.DataFrame:
    d = lap_end_rows(df)
    return d[d["compound"].isin(DRY_COMPOUNDS) & d["last_lap"].notna()]


def next_lap_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = _dry_lap_end(df)
    return d[d["y_next_lap"].notna()].copy()


def lap5_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = _dry_lap_end(df)
    return d[d["y_lap_5"].notna()].copy()


def cliff_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = _dry_lap_end(df)
    return d[d["y_cliff_3"].notna()].copy()


def fresh_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["kind"] == "pit_entry") & df["y_fresh_lap"].notna()]
    return d[d["last_lap"].notna()].copy()


# ----------------------------------------------------------------------------- models
def _col(df: pd.DataFrame, *cols: str) -> np.ndarray:
    """First non-missing value among ``cols`` (engineer values may be None early on)."""
    out = np.full(len(df), np.nan)
    for c in cols:
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            out = np.where(np.isnan(out), v, out)
    return out


class _Fixed:
    """A prediction read from the row (no training)."""

    name = ""
    cols: tuple[str, ...] = ()

    def fit(self, df: pd.DataFrame, target: str):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _col(df, *self.cols)


class LastLap(_Fixed):
    """The lap just completed (whatever it was)."""

    name = "last_lap"
    cols = ("last_lap",)


class LastCleanLap(_Fixed):
    """The car's last clean lap on its current set."""

    name = "last_clean_lap"
    cols = ("tyre__last_clean_s", "last_lap")


class StintMedian(_Fixed):
    """Median of the clean laps on the current set (the v0.1 stub's pace)."""

    name = "stint_median"
    cols = ("tyre__stint_pace", "last_lap")


class TyrePace(_Fixed):
    """The engineer's ``pace_s``: next clean lap, fuel and wear included."""

    name = "tyre_pace"
    cols = ("tyre__pace_s", "tyre__last_clean_s", "last_lap")


class TyreExtrap5:
    """Five laps ahead from the engineer: pace_s + 4 * (deg_s_per_lap - fuel_s_per_lap)."""

    name = "tyre_extrap"

    def fit(self, df, target):
        return self

    def predict(self, df):
        pace = _col(df, "tyre__pace_s", "tyre__last_clean_s", "last_lap")
        slope = _col(df, "tyre__deg_s_per_lap") - _col(df, "tyre__fuel_s_per_lap")
        return pace + 4 * np.nan_to_num(slope)


class BaseRate:
    name = "base_rate"

    def fit(self, df, target):
        self.p = float(df[target].mean())
        return self

    def predict(self, df):
        return np.full(len(df), self.p)


class TyreCliffRisk:
    """The engineer's ``cliff_risk`` (base rate of the training races where it is missing)."""

    name = "tyre_cliff_risk"

    def fit(self, df, target):
        self.p = float(df[target].mean())
        return self

    def predict(self, df):
        v = _col(df, "tyre__cliff_risk")
        return np.clip(np.where(np.isnan(v), self.p, v), 0.0, 1.0)


# learned models: residual on top of a reference, so one model serves every circuit
LAP_FEATURES = [
    "tyre_age", "laps_in_stint", "n_clean_stint", "race_frac", "laps_remaining", "position", "interval_ahead",
    "gap_behind", "status_age_s", "track_temp", "cmp_soft", "cmp_medium", "cmp_hard", "stint", "pace_drop",
    "pace_slope", "tyre__pace_sd_s", "tyre__deg_s_per_lap", "tyre__deg_sd", "tyre__fuel_s_per_lap",
    "tyre__traffic", "tyre__resid_last_s", "tyre__resid_trend_s", "tyre__stint_clean_laps",
]


def _rel_features(df: pd.DataFrame, ref: np.ndarray) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for c in LAP_FEATURES:
        x[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in ("last_lap", "stint_best", "tyre__stint_pace", "tyre__last_clean_s"):
        x[f"{c}_rel"] = (pd.to_numeric(df[c], errors="coerce") if c in df else np.nan) - ref
    return x.astype(float)


class _ResidualGBM:
    name = ""
    ref_cols: tuple[str, ...] = ("tyre__pace_s", "tyre__last_clean_s", "last_lap")
    clip = 5.0

    def _ref(self, df):
        return _col(df, *self.ref_cols)

    def fit(self, df, target):
        ref = self._ref(df)
        y = np.clip(df[target].to_numpy(float) - ref, -self.clip, self.clip)
        ok = ~np.isnan(y)
        self.model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
            loss="absolute_error", random_state=0,
        ).fit(_rel_features(df, ref)[ok], y[ok])
        return self

    def predict(self, df):
        ref = self._ref(df)
        return ref + self.model.predict(_rel_features(df, ref))


class NextLapGBM(_ResidualGBM):
    """Learned correction to ``pace_s`` (traffic, recent form), trained on earlier races."""

    name = "next_lap_gbm"


class Lap5GBM(_ResidualGBM):
    name = "lap5_gbm"

    def _ref(self, df):
        return TyreExtrap5().predict(df)


class CliffGBM:
    """Learned cliff probability from tyre and pace features, trained on earlier races."""

    name = "cliff_gbm"

    def fit(self, df, target):
        ref = _col(df, "tyre__pace_s", "last_lap")
        self.model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=10.0,
            random_state=0,
        ).fit(_rel_features(df, ref), df[target].astype(int))
        return self

    def predict(self, df):
        ref = _col(df, "tyre__pace_s", "last_lap")
        return self.model.predict_proba(_rel_features(df, ref))[:, 1]


def _fresh_pred(df: pd.DataFrame) -> np.ndarray:
    """``fresh_<compound>_s`` for the compound actually fitted.

    The fitted compound (``y_fresh_compound``) is part of the question - "if you fit X
    now, how fast is the first flying lap?" - not something being predicted.
    """
    out = np.full(len(df), np.nan)
    comp = df["y_fresh_compound"].to_numpy()
    for c in DRY_COMPOUNDS:
        col = f"tyre__fresh_{c.lower()}_s"
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            out = np.where(comp == c, v, out)
    return out


class StintBest(_Fixed):
    """Fresh-tyre guess: the car's best clean lap on the set it is taking off."""

    name = "stint_best"
    cols = ("stint_best", "last_lap")


class TyreFresh:
    name = "tyre_fresh"

    def fit(self, df, target):
        return self

    def predict(self, df):
        v = _fresh_pred(df)
        return np.where(np.isnan(v), _col(df, "stint_best", "last_lap"), v)


class FreshGBM(_ResidualGBM):
    name = "fresh_gbm"

    def _ref(self, df):
        return TyreFresh().predict(df)


def tyre_tasks(**_) -> list[Task]:
    return [
        Task("next_lap_time", "regression", "y_next_lap", next_lap_rows,
             (LastLap, LastCleanLap, StintMedian, TyrePace, NextLapGBM), keep=("compound", "tyre_age")),
        Task("lap_time_5", "regression", "y_lap_5", lap5_rows,
             (LastLap, LastCleanLap, StintMedian, TyrePace, TyreExtrap5, Lap5GBM), keep=("compound", "tyre_age")),
        Task("tyre_cliff_3", "binary", "y_cliff_3", cliff_rows, (BaseRate, TyreCliffRisk, CliffGBM),
             keep=("compound", "tyre_age")),
        Task("fresh_tyre_pace", "regression", "y_fresh_lap", fresh_rows, (StintBest, TyreFresh, FreshGBM),
             keep=("y_fresh_compound", "sc", "vsc")),
    ]
