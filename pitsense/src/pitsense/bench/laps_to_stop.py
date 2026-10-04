"""Laps until each car's next stop: a full distribution (owner: stop timing).

Label (reads the finished race, scoring only), on every lap-end row of car d after lap L:

* ``y_laps_to_stop``  in-lap of the next non-red-flag stop minus L (>= 1); NaN if the car never stops again.
* ``y_stop_event``    1 if there is a next stop.
* ``y_stop_obs``      the observed time: the laps to the stop, or, for a car that stops no more, the laps it
                      still completed. ``y_stop_retired`` says whether it retired instead of taking the flag.
                      A car that takes the flag without stopping again has "never" as its time (known), a
                      retirement is censored at its last lap: the only real censoring.

Model (``SurvivalGBM``): discrete-time survival. A stacked person-period table has one record per row and horizon
h = 1..15 (a lap is at risk while the car is observed; a stop on lap L+h ends it: label 1 at that h, 0 at the
earlier ones; a row without a stop contributes 0s up to its last observed lap, so a retirement or a stop beyond
15 laps does not bias the hazards). One boosted classifier learns the hazard h(x, h) from all bench features
plus the horizon; the CDF is ``1 - prod(1 - h)``, forced to 0 beyond the laps left in the race.
Outputs per row (``summarize``): ``p_stop_le_k`` for k = 1, 2, 3, 5, 8, ``laps_to_stop_exp`` (restricted mean,
E[min(T, 16)]) and ``laps_to_stop_med`` (16 = "more than 15 laps").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..state import RaceState
from .features import feature_columns
from .tasks import Task, lap_end_rows

H = 15  # horizons 1..H
CAP = H + 1  # "more than H laps"
KS = (1, 2, 3, 5, 8)
HORIZONS = np.arange(1, H + 1)


# ----------------------------------------------------------------------------- label
def stop_labels(rows: list[dict], final: RaceState) -> None:
    stops: dict[str, list[int]] = {}
    for pe in final.pit_events:
        if not pe.under_red:
            stops.setdefault(pe.driver, []).append(pe.in_lap)
    last: dict[str, int] = {}
    for lap in final.laps:
        last[lap.driver] = max(last.get(lap.driver, 0), lap.lap)
    retired = {n for n, d in final.drivers.items() if not d.running}
    for row in rows:
        if row.get("kind", "lap_end") != "lap_end":
            continue
        drv, L = row["driver"], row["lap"]
        nxt = next((s for s in sorted(stops.get(drv, ())) if s > L), None)
        row["y_laps_to_stop"] = float(nxt - L) if nxt is not None else float("nan")
        row["y_stop_event"] = int(nxt is not None)
        row["y_stop_obs"] = float(nxt - L) if nxt is not None else float(max(last.get(drv, 0) - L, 0))
        row["y_stop_retired"] = int(nxt is None and drv in retired)


# ----------------------------------------------------------------------------- cdf helpers
def _left(df: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(pd.to_numeric(df["laps_remaining"], errors="coerce").to_numpy(float), nan=99.0)


def cdf_from_hazard(hz: np.ndarray, laps_left: np.ndarray | None = None) -> np.ndarray:
    """P(T <= k) for k = 1..H from hazards [n, H]; hazards beyond the laps left in the race are zero."""
    hz = np.clip(hz, 0.0, 0.999)
    if laps_left is not None:
        hz = np.where(HORIZONS[None, :] <= laps_left[:, None], hz, 0.0)
    return 1.0 - np.cumprod(1.0 - hz, axis=1)


def summarize(cdf: np.ndarray) -> dict[str, np.ndarray]:
    """The engineer-facing outputs of a CDF matrix [n, H]."""
    out = {f"p_stop_le_{k}": cdf[:, k - 1] for k in KS}
    out["laps_to_stop_exp"] = 1.0 + (1.0 - cdf).sum(axis=1)  # E[min(T, H+1)] = sum_{k=0..H} S(k), S(0) = 1
    out["laps_to_stop_med"] = np.where(cdf[:, -1] >= 0.5, (cdf < 0.5).sum(axis=1) + 1, CAP).astype(float)
    return out


# ----------------------------------------------------------------------------- stacked table
def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan for c in cols},
                        index=df.index).astype(np.float32)


def _stack_x(x: np.ndarray, cols: list[str], h: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Feature matrix of the person-period records: row ``rows[i]`` at horizon ``h[i]``, plus timing features."""
    X = x[rows]
    ci = {c: i for i, c in enumerate(cols)}
    hh = h.astype(np.float32)
    return np.column_stack([X, hh, X[:, ci["laps_remaining"]] - hh, X[:, ci["tyre_age"]] + hh])


class SurvivalGBM:
    """Discrete-time survival: boosted hazard over horizons 1..15 on the stacked person-period table."""

    name = "survival_gbm"
    params = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=10.0,
                  random_state=0)
    thin = 2  # train on every ``thin``-th lap (a car's consecutive rows are strongly correlated)
    half_life_years = 3.0  # weight a training race by 0.5 ** (age / half_life) relative to the latest training race
    all_engineer_keys = False  # True: every numeric "<engineer>__<key>" column (slower, and drifts: over-predicts)
    # undeclared engineer values that carry timing information (rivals' stop probabilities, pit-stop margins)
    extra = ("rivals__pit_prob_1", "rivals__pit_prob_3", "rivals__pit_prob_5", "rivals__ratio", "rivals__undercut_threat",
             "rivals__undercut_chance", "rivals__typical_stint_laps", "rivals__in_pit_window", "rivals__stuck",
             "pitstop__loss_now", "pitstop__passers_expected", "pitstop__margin_s", "pitstop__rejoin_delta")

    def _columns(self, df: pd.DataFrame) -> list[str]:
        cols = feature_columns() + [c for c in self.extra if c in df.columns]
        if self.all_engineer_keys:
            cols = cols + sorted(c for c in df.columns if "__" in c and c not in cols and not c.startswith("weather__")
                                 and (pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])))
        return cols

    def fit(self, df: pd.DataFrame, target: str = "y_stop_obs") -> "SurvivalGBM":
        self.cols = self._columns(df)
        d = df[df["lap"] % self.thin == 0] if self.thin > 1 else df
        obs = d["y_stop_obs"].to_numpy(float)
        ev = d["y_stop_event"].to_numpy(int)
        n_at = np.minimum(obs, H).astype(int)  # records per row: the laps it was observed, at most H
        rows = np.repeat(np.arange(len(d)), n_at)
        h = np.concatenate([np.arange(1, m + 1) for m in n_at]) if n_at.sum() else np.array([], int)
        w = None
        if self.half_life_years:
            age = (d["start_utc"].max() - d["start_utc"]).dt.days.to_numpy(float) / 365.25
            w = (0.5 ** (age / self.half_life_years))[rows]
        y = ((h == obs[rows]) & (ev[rows] == 1)).astype(np.int8)
        X = _stack_x(_num(d, self.cols).to_numpy(), self.cols, h, rows)
        X[:, np.isnan(X).all(axis=0)] = 0.0  # a column that is missing everywhere (tiny training sets) breaks the binning
        self.model = HistGradientBoostingClassifier(**self.params).fit(X, y, sample_weight=w)
        return self

    def hazards(self, df: pd.DataFrame) -> np.ndarray:
        x = _num(df, self.cols).to_numpy()
        n = len(df)
        p = self.model.predict_proba(_stack_x(x, self.cols, np.tile(HORIZONS, n), np.repeat(np.arange(n), H)))[:, 1]
        return p.reshape(n, H)

    def predict_cdf(self, df: pd.DataFrame) -> np.ndarray:
        return cdf_from_hazard(self.hazards(df), _left(df))

    predict = predict_cdf  # what the model bundle calls: the [n, 15] CDF


class SurvivalPerHorizon(SurvivalGBM):
    """Discrete-time survival with one boosted hazard model per horizon h = 1..15, each fit on the cars still at
    risk at h (observed for h laps without a stop earlier) with the label "stops exactly at h". Same hazards as
    the stacked table, but each horizon keeps its own base rate (the pooled model over-predicts short horizons)."""

    name = "survival_hz"
    params = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=300, l2_regularization=10.0,
                  random_state=0)
    thin = 1
    extra = ()  # the declared bench features only: the rivals' / pit-stop extras did not help on 2025 and made it over-predict

    def fit(self, df: pd.DataFrame, target: str = "y_stop_obs") -> "SurvivalPerHorizon":
        self.cols = self._columns(df)
        d = df[df["lap"] % self.thin == 0] if self.thin > 1 else df
        X = _num(d, self.cols).to_numpy()
        X[:, np.isnan(X).all(axis=0)] = 0.0
        obs, ev = d["y_stop_obs"].to_numpy(float), d["y_stop_event"].to_numpy(int)
        w = None
        if self.half_life_years:
            w = 0.5 ** ((d["start_utc"].max() - d["start_utc"]).dt.days.to_numpy(float) / 365.25 / self.half_life_years)
        self.models = []
        for h in HORIZONS:
            at = obs >= h
            y = ((obs == h) & (ev == 1))[at].astype(np.int8)
            self.models.append(HistGradientBoostingClassifier(**self.params).fit(
                X[at], y, sample_weight=None if w is None else w[at]))
        return self

    def hazards(self, df: pd.DataFrame) -> np.ndarray:
        X = _num(df, self.cols).to_numpy()
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])


class BaseHazard:
    """Climatology: one hazard per horizon, the same for every row."""

    name = "base_hazard"

    def fit(self, df, target="y_stop_obs"):
        obs, ev = df["y_stop_obs"].to_numpy(float), df["y_stop_event"].to_numpy(int)
        self.h = np.array([((obs == k) & (ev == 1)).sum() / max((obs >= k).sum(), 1) for k in HORIZONS])
        return self

    def predict_cdf(self, df):
        return cdf_from_hazard(np.tile(self.h, (len(df), 1)), _left(df))

    predict = predict_cdf


class GeoGBMHazard:
    """The core ``gbm_hazard`` extended geometrically: a constant per-lap hazard from its P(stop within 3 laps)."""

    name = "geo_gbm_hazard"
    target = "y_pit_3"
    k = 3

    def fit(self, df, target="y_stop_obs"):
        from .models import GBMHazard

        self.m = GBMHazard().fit(df, self.target)
        return self

    def predict_cdf(self, df):
        p = np.clip(self.m.predict(df), 1e-6, 0.999)
        hz = 1.0 - (1.0 - p) ** (1.0 / self.k)
        return cdf_from_hazard(np.tile(hz[:, None], (1, H)), _left(df))

    predict = predict_cdf


class GeoGBMHazard1(GeoGBMHazard):
    """The same from its P(stop within 1 lap)."""

    name = "geo_gbm_hazard_1"
    target = "y_pit_1"
    k = 1


class RivalsHazard:
    """The rivals engineer's ``pit_prob_1/3/5`` as a cumulative hazard, interpolated linearly in between and
    extended beyond 5 laps at the 3-to-5 rate (no training; climatology where the engineer has no value)."""

    name = "rivals_hazard"

    def fit(self, df, target="y_stop_obs"):
        self.base = BaseHazard().fit(df)
        return self

    def predict_cdf(self, df):
        P = _num(df, [f"rivals__pit_prob_{k}" for k in (1, 3, 5)]).to_numpy(float)
        bad = np.isnan(P).any(axis=1)
        P = np.maximum.accumulate(np.clip(np.nan_to_num(P, nan=0.0), 1e-6, 0.995), axis=1)
        Hc = -np.log1p(-P)  # cumulative hazard at 1, 3, 5
        k = HORIZONS[None, :].astype(float)
        c = np.where(k <= 1, Hc[:, [0]] * k,
                     np.where(k <= 3, Hc[:, [0]] + (Hc[:, [1]] - Hc[:, [0]]) * (k - 1) / 2,
                              Hc[:, [1]] + (Hc[:, [2]] - Hc[:, [1]]) * (k - 3) / 2))
        step = np.diff(np.maximum.accumulate(c, axis=1), axis=1, prepend=0.0)
        cdf = cdf_from_hazard(1.0 - np.exp(-step), _left(df))
        cdf[bad] = self.base.predict_cdf(df)[bad]
        return cdf

    predict = predict_cdf


MODELS = (BaseHazard, GeoGBMHazard1, GeoGBMHazard, RivalsHazard, SurvivalPerHorizon)


# ----------------------------------------------------------------------------- scores
def _km_censor(obs: np.ndarray, cens: np.ndarray):
    """Kaplan-Meier survival G(t) of the censoring time (censored = retired without a later stop)."""
    G, g = [], 1.0
    for t in np.unique(obs[cens == 1]):
        g *= 1.0 - ((obs == t) & (cens == 1)).sum() / max((obs >= t).sum(), 1)
        G.append((t, g))

    def f(t):
        out = np.ones(np.shape(t), float)
        for s, v in G:
            out = np.where(np.asarray(t) >= s, v, out)
        return out

    return f


def concordance(obs, ev, score, retired, horizon: int = H, chunk: int = 4_000_000) -> float:
    """Harrell's C over pairs (i, j): i stops at T_i <= horizon and j is known to be later (stops later, never
    stops again, or was seen beyond T_i before retiring). ``score`` is larger when the stop is expected later;
    a pair is concordant when i's score is below j's, ties count half."""
    obs, ev, score, retired = (np.asarray(a) for a in (obs, ev, score, retired))
    ti = np.where((ev == 1) & (obs <= horizon))[0]
    tj = np.where((ev == 1) | (retired == 1), obs, np.inf)
    step = max(1, chunk // max(len(obs), 1))
    conc = tied = tot = 0.0
    for a in range(0, len(ti), step):
        i = ti[a:a + step]
        m = tj[None, :] > obs[i][:, None]
        s_i, s_j = score[i][:, None], score[None, :]
        conc += ((s_i < s_j) & m).sum()
        tied += ((s_i == s_j) & m).sum()
        tot += m.sum()
    return float((conc + 0.5 * tied) / tot) if tot else float("nan")


def _ece(y: np.ndarray, p: np.ndarray, bins=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5)) -> float:
    b = np.digitize(p, bins)
    return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).sum() for i in np.unique(b)) / len(y))


def _outcome(df: pd.DataFrame, k: int):
    """(stopped within k, outcome known): a retiree observed fewer than k laps without a stop is unknown."""
    obs, ev, ret = (df[c].to_numpy() for c in ("y_stop_obs", "y_stop_event", "y_stop_retired"))
    hit = (ev == 1) & (obs <= k)
    return hit, hit | (ret == 0) | (obs >= k)


def survival_scores(df: pd.DataFrame, cdf: np.ndarray) -> dict:
    """c-index, integrated Brier score (IPCW for retirements), calibration at k = 1/3/5, window accuracy."""
    obs = df["y_stop_obs"].to_numpy(float)
    ev = df["y_stop_event"].to_numpy(int)
    ret = df["y_stop_retired"].to_numpy(int)
    s = summarize(cdf)
    out = {"n": len(df), "events": int(ev.sum()), "c_index": concordance(obs, ev, s["laps_to_stop_exp"], ret)}
    G = _km_censor(obs, ret)
    ibs = []
    for k in range(1, H + 1):
        hit, known = _outcome(df, k)
        # inverse-probability-of-censoring weights: a known outcome counts 1 / G(min(T, k))
        w = np.where(known, 1.0 / np.clip(G(np.minimum(np.where(hit, obs, k), k) - 1e-9), 0.05, None), 0.0)
        ibs.append(float(np.sum(w * (cdf[:, k - 1] - hit) ** 2) / np.sum(w)))
        if k in (1, 3, 5):
            out[f"brier_{k}"] = ibs[-1]
            out[f"cal_err_{k}"] = _ece(hit[known].astype(float), cdf[known, k - 1])
            out[f"cal_ratio_{k}"] = float(cdf[known, k - 1].mean() / max(hit[known].mean(), 1e-9))  # mean predicted / observed
    out["ibs"] = float(np.mean(ibs))
    e = ev == 1
    med, real = s["laps_to_stop_med"][e], obs[e]
    out["window_acc"] = float(np.mean(np.abs(med - real) <= 2))
    near = real <= H
    out["window_acc_le15"] = float(np.mean(np.abs(med[near] - real[near]) <= 2))
    return out


def calibration_k(df: pd.DataFrame, cdf: np.ndarray, k: int) -> pd.DataFrame:
    from .metrics import calibration_table

    hit, known = _outcome(df, k)
    return calibration_table(hit[known].astype(float), cdf[known, k - 1])


# ----------------------------------------------------------------------------- task
def rows_for_task(df: pd.DataFrame) -> pd.DataFrame:
    d = lap_end_rows(df)
    return d[d["y_stop_obs"].notna()].copy()


def evaluate(df: pd.DataFrame, *, test_year: int, min_train_races: int):
    from .evaluate import EvalResult, _splits, fit_predict_cdf

    scored = []
    for race_id, train, test in _splits(rows_for_task(df), test_year, min_train_races):
        scored.append((race_id, test, {cls.name: fit_predict_cdf(cls, train, test) for cls in MODELS}))
    if not scored:
        raise ValueError(f"laps_to_stop: no {test_year} race has {min_train_races}+ earlier races")
    T = pd.concat([t for _, t, _ in scored], ignore_index=True)
    names = [m.name for m in MODELS]
    C = {n: np.vstack([c[n] for _, _, c in scored]) for n in names}
    board = pd.DataFrame([{"model": n, **survival_scores(T, C[n])} for n in names]).sort_values("ibs").reset_index(drop=True)
    per = pd.DataFrame([{"race_id": rid, **{n: survival_scores(t, c[n])["ibs"] for n in names}} for rid, t, c in scored])
    out = T[["race_id", "driver", "tla", "lap", "t", "y_laps_to_stop", "y_stop_obs", "y_stop_event", "y_stop_retired"]].copy()
    for n in names:
        for key, v in summarize(C[n]).items():
            out[f"{n}__{key}"] = v
    cal = {n: pd.concat([calibration_k(T, C[n], k).assign(k=k) for k in (1, 3, 5)]).assign(bin=lambda x: x["bin"].astype(str))
           for n in ("survival_hz", "rivals_hazard")}
    return EvalResult("laps_to_stop", board, per, out, cal)


def laps_to_stop_tasks(**_) -> list[Task]:
    return [Task("laps_to_stop", "survival", "y_stop_obs", rows_for_task, MODELS, evaluate=evaluate)]
