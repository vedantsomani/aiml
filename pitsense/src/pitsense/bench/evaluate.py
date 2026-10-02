"""Expanding-window evaluation: for each test race, train on every earlier race only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..asof import assert_trained_before
from .metrics import binary_scores, calibration_table, position_scores, regression_scores
from .tasks import Task, core_tasks

# a Grand Prix lasts at most ~4h (incl. red flags); used as a conservative end time
RACE_SPAN = timedelta(hours=4)


@dataclass
class EvalResult:
    task: str
    leaderboard: pd.DataFrame
    per_race: pd.DataFrame
    predictions: pd.DataFrame
    calibration: dict[str, pd.DataFrame]


def _races(df: pd.DataFrame) -> pd.DataFrame:
    return df[["race_id", "start_utc", "year"]].drop_duplicates().sort_values("start_utc")


def _splits(df: pd.DataFrame, test_year: int, min_train_races: int):
    races = _races(df)
    for _, race in races[races.year == test_year].iterrows():
        train = df[df.start_utc + RACE_SPAN < race.start_utc]
        if train.race_id.nunique() < min_train_races:
            continue
        yield race.race_id, train, df[df.race_id == race.race_id]


def fit_predict(cls, train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Fit a fresh model on ``train`` and predict ``test``.

    Every model in every task goes through here, so this is where the training
    cutoff is enforced: no training race may still be running when the test race starts.
    """
    assert_trained_before([train.start_utc.max() + RACE_SPAN], test.start_utc.min())
    return cls().fit(train, target).predict(test)


def evaluate_task(task: Task, df: pd.DataFrame, test_year: int = 2026, min_train_races: int = 10) -> EvalResult:
    if task.evaluate is not None:
        return task.evaluate(df, test_year=test_year, min_train_races=min_train_races)
    data = task.select(df)
    preds = []
    for _race_id, train, test in _splits(data, test_year, min_train_races):
        out = test[["race_id", "driver", "tla", "lap", "t", *task.keep, task.target]].copy()
        for cls in task.models:
            out[cls.name] = fit_predict(cls, train, test, task.target)
        preds.append(out)
    if not preds:
        raise ValueError(f"{task.name}: no {test_year} race has {min_train_races}+ earlier races to train on")
    return _SCORERS[task.kind](task, pd.concat(preds, ignore_index=True))


def _score_binary(task: Task, P: pd.DataFrame) -> EvalResult:
    names = [c.name for c in task.models]
    y = P[task.target].to_numpy(float)
    board = pd.DataFrame([{"model": n, **binary_scores(y, P[n].to_numpy(float))} for n in names])
    board["brier_skill"] = 1 - board["brier"] / board.loc[0, "brier"]  # vs the reference baseline
    board = board.sort_values("log_loss").reset_index(drop=True)
    per_race = (
        P.groupby("race_id", sort=False)
        .apply(lambda g: pd.Series({n: binary_scores(g[task.target].to_numpy(float), g[n].to_numpy(float))["log_loss"] for n in names}), include_groups=False)
        .reset_index()
    )
    cal = {n: calibration_table(y, P[n].to_numpy(float)) for n in names}
    return EvalResult(task.name, board, per_race, P, cal)


def _score_position(task: Task, P: pd.DataFrame) -> EvalResult:
    names = [c.name for c in task.models]
    y = P[task.target].to_numpy(float)
    board = pd.DataFrame([{"model": n, **position_scores(y, P[n].to_numpy(float))} for n in names])
    board = board.sort_values(["mae", "exact"], ascending=[True, False]).reset_index(drop=True)
    per_race = (
        P.groupby("race_id", sort=False)
        .apply(lambda g: pd.Series({n: position_scores(g[task.target], g[n])["exact"] for n in names}), include_groups=False)
        .reset_index()
    )
    return EvalResult(task.name, board, per_race, P, {})


def _score_regression(task: Task, P: pd.DataFrame) -> EvalResult:
    names = [c.name for c in task.models]
    y = P[task.target].to_numpy(float)
    board = pd.DataFrame([{"model": n, **regression_scores(y, P[n].to_numpy(float))} for n in names])
    board = board.sort_values("mae").reset_index(drop=True)
    per_race = (
        P.groupby("race_id", sort=False)
        .apply(lambda g: pd.Series({n: regression_scores(g[task.target], g[n])["mae"] for n in names}), include_groups=False)
        .reset_index()
    )
    return EvalResult(task.name, board, per_race, P, {})


_SCORERS = {"binary": _score_binary, "position": _score_position, "regression": _score_regression}


def evaluate_pit(df: pd.DataFrame, horizon: int = 3, test_year: int = 2026, min_train_races: int = 10) -> EvalResult:
    (task,) = [t for t in core_tasks(horizons=(horizon,)) if t.kind == "binary"]
    return evaluate_task(task, df, test_year, min_train_races)


def evaluate_rejoin(df: pd.DataFrame, test_year: int = 2026, min_train_races: int = 10) -> EvalResult:
    (task,) = [t for t in core_tasks(horizons=()) if t.name == "position_after_stop"]
    return evaluate_task(task, df, test_year, min_train_races)


def write_report(results: list[EvalResult], out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# PitSense-Bench leaderboard", ""]
    lines.append(
        f"Test races: {meta['test_races']} ({meta['test_year']}), each scored with models trained only on "
        f"races that finished before it started. Decision points: {meta['decision_points']:,} "
        f"from {meta['races']} races. Feature version {meta['feature_version']}."
    )
    lines.append("")
    for r in results:
        lines.append(f"## {r.task}")
        lines.append("")
        lines.append(_md_table(r.leaderboard))
        lines.append("")
        if r.calibration:
            best = r.leaderboard.iloc[0]["model"]
            lines.append(f"Calibration of `{best}` (predicted vs observed rate):")
            lines.append("")
            lines.append(_md_table(r.calibration[best].assign(bin=lambda x: x["bin"].astype(str))))
            lines.append("")
        r.predictions.to_parquet(out_dir / f"predictions_{r.task}.parquet", index=False)
        r.per_race.to_csv(out_dir / f"per_race_{r.task}.csv", index=False)
    path = out_dir / "leaderboard.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "leaderboard.json").write_text(
        json.dumps({"meta": meta, **{r.task: r.leaderboard.to_dict(orient="records") for r in results}}, indent=1, default=str),
        encoding="utf-8",
    )
    return path


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(out)
