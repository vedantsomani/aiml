"""What runs on the pit wall and in the benchmark: one line per component.

Entries are "module:attribute" strings, imported on first use, so an optional
dependency (e.g. torch for the voice) is only needed by what actually runs.
To add a component, write it in your own module and add one line here.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Engineer subclasses (pitwall/engineer.py). Order only matters for display; dependencies are resolved.
ENGINEERS: list[str] = [
    "pitsense.pitwall.engineers.tyre:TyreEngineer",
    "pitsense.pitwall.engineers.pitstop:PitStopEngineer",
    "pitsense.pitwall.engineers.rivals:RivalsEngineer",
    "pitsense.pitwall.engineers.rules:RulesEngineer",
    "pitsense.pitwall.engineers.weather:WeatherEngineer",
    "pitsense.pitwall.engineers.models:ModelsEngineer",
    "pitsense.pitwall.engineers.strategy:StrategyEngineer",
    "pitsense.pitwall.engineers.head:HeadOfStrategy",
]

# Benchmark task factories: f(**options) -> list[Task]  (bench/tasks.py). Options: horizons.
TASKS: list[str] = [
    "pitsense.bench.tasks:core_tasks",
]

# Extra models for the core tasks (bench/tasks.py), scored next to the built-in ones.
TASK_MODELS: dict[str, list[str]] = {
    "pit_within": [],  # every pit_within_k task
    "position_after_stop": [],
}

# Labelers: f(rows, final_state) -> None, adding y_* columns in place (bench/labels.py).
LABELERS: list[str] = [
    "pitsense.bench.labels:pit_labels",
]

# CLI extensions: f(subparsers) -> None, adding `pitsense <command>` parsers (cli.py).
COMMANDS: list[str] = []


def load(spec: str) -> Any:
    module, _, attr = spec.partition(":")
    return getattr(import_module(module), attr)


def engineer_classes() -> list[type]:
    return [load(s) for s in ENGINEERS]


def tasks(**options) -> list:
    return [t for s in TASKS for t in load(s)(**options)]


def task_models(family: str) -> tuple[type, ...]:
    return tuple(load(s) for s in TASK_MODELS.get(family, []))


def labelers() -> list:
    return [load(s) for s in LABELERS]


def command_adders() -> list:
    return [load(s) for s in COMMANDS]
