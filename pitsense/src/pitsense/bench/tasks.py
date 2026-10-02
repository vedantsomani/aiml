"""Benchmark tasks: which decision rows, what to predict, which models, how to score.

Add a task by writing a factory ``f(**options) -> list[Task]`` in your own
module and listing it in ``pitsense.registry.TASKS``. A task that needs more
than the rows (e.g. running simulations) supplies its own ``evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

KINDS = ("binary", "position", "regression")


@dataclass(frozen=True)
class Task:
    name: str  # also the report file names: predictions_<name>.parquet, per_race_<name>.csv
    kind: str  # binary | position | regression (picks the scorer)
    target: str  # a y_* column
    select: Callable[[pd.DataFrame], pd.DataFrame]  # the decision rows this task is scored on
    models: tuple[type, ...] = ()  # the first one is the reference baseline
    keep: tuple[str, ...] = ()  # extra columns copied into the predictions file
    # custom evaluation: f(df, *, test_year, min_train_races) -> EvalResult
    evaluate: Callable[..., Any] | None = None


def lap_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Lap-end decisions, without cars that retire within 3 laps (a retirement isn't a strategy call)."""
    return df[(df["kind"] == "lap_end") & (df["y_retire_3"] == 0)].copy()


def pit_entry_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pit entries whose rejoin position is known."""
    return df[(df["kind"] == "pit_entry") & df.y_pos_after_stop.notna()].copy()


def core_tasks(horizons: tuple[int, ...] = (1, 3), **_) -> list[Task]:
    from .. import registry
    from .models import PIT_MODELS, REJOIN_MODELS

    pit_models = PIT_MODELS + registry.task_models("pit_within")
    rejoin_models = REJOIN_MODELS + registry.task_models("position_after_stop")
    tasks = [Task(f"pit_within_{h}", "binary", f"y_pit_{h}", lap_end_rows, pit_models) for h in horizons]
    tasks.append(
        Task("position_after_stop", "position", "y_pos_after_stop", pit_entry_rows, rejoin_models,
             keep=("position", "cars_within_pitloss", "sc", "vsc"))
    )
    return tasks
