"""Train the benchmark's models once, save them as a versioned bundle, load them for a race.

The benchmark refits every model for every test race on the races before it. The
live pit wall cannot refit mid-race, so it loads a bundle trained once on every
race that ended before a *cutoff*. A bundle records when its training data ends
(``train_end_utc``); loading it for a race refuses the bundle if that is not
before the race starts (``asof.assert_trained_before``).

Training window: a race counts as finished at ``start_utc + RACE_SPAN`` (the same
conservative rule as ``bench.evaluate``), so a bundle trained with
``cutoff = race start`` is identical to what the benchmark scores that race with.

Model choice: one model per task, the best on the 2025 benchmark by each task's
primary metric (``BEST_MODEL``; chosen with ``--test-year 2025``, so 2026 stays a
clean test). Tasks not listed use their last (most elaborate) model. Tasks that list
no models are skipped.

Metrics stored in a bundle are *holdout* scores: the chosen model refit without the
last ``holdout`` training races and scored on them. The saved model is fit on all.

Pickle files are code: only load bundles you trained yourself.
"""

from __future__ import annotations

import pickle
import subprocess
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .asof import assert_trained_before
from .bench.evaluate import RACE_SPAN
from .bench.features import FEATURE_VERSION
from .bench.metrics import binary_scores, position_scores, regression_scores
from .bench.tasks import Task
from .config import data_dir

# Best model per task on the 2025 benchmark (primary metric: log loss / MAE).
BEST_MODEL: dict[str, str] = {
    "pit_within_1": "gbm_hazard",
    "pit_within_3": "gbm_hazard",
    "position_after_stop": "pitstop_gbm",
    "next_lap_time": "next_lap_gbm",
    "lap_time_5": "lap5_gbm",
    "tyre_cliff_3": "cliff_gbm",
    "fresh_tyre_pace": "fresh_gbm",
    "pit_loss": "pitstop_loss",
    "laps_to_stop": "survival_hz",
}

_SCORERS = {"binary": binary_scores, "position": position_scores, "regression": regression_scores}


def models_dir() -> Path:
    return data_dir() / "models"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


@dataclass
class TrainedBundle:
    train_end_utc: datetime  # latest end of any training race; must be before a race for it to use the bundle
    cutoff_utc: datetime
    trained_on: tuple[str, ...]  # race ids
    feature_version: str
    git_commit: str | None
    models: dict[str, object] = field(default_factory=dict)  # task name -> fitted model
    metrics: dict[str, dict] = field(default_factory=dict)  # task name -> {model, kind, holdout_races, scores}
    targets: dict[str, str] = field(default_factory=dict)  # task name -> y_* column

    def check_for(self, race_start_utc: datetime) -> "TrainedBundle":
        assert_trained_before([self.train_end_utc], race_start_utc)
        return self

    def predict(self, task: str, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.models[task].predict(df), dtype=float)

    def describe(self) -> str:
        lines = [f"bundle {self.feature_version} @ {self.git_commit}: {len(self.trained_on)} races, "
                 f"training ends {self.train_end_utc:%Y-%m-%d %H:%M}Z, cutoff {self.cutoff_utc:%Y-%m-%d %H:%M}Z"]
        for task, m in self.metrics.items():
            key = {"binary": "log_loss", "survival": "ibs"}.get(m.get("kind"), "mae")
            lines.append(f"  {task:<20} {m['model']:<14} holdout {key} {m['scores'].get(key, float('nan')):.4f}"
                         f" (n={m['scores'].get('n', 0)})")
        return "\n".join(lines)


def _as_utc(x) -> datetime:
    ts = pd.Timestamp(x)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _pick(task: Task, choices: dict[str, str]):
    name = choices.get(task.name)
    if name is None:
        return task.models[-1]
    for cls in task.models:
        if cls.name == name:
            return cls
    raise KeyError(f"{task.name}: no model {name!r} (have {[c.name for c in task.models]})")


def train_bundle(df: pd.DataFrame, cutoff_utc, *, tasks: list[Task] | None = None,
                 choices: dict[str, str] | None = None, holdout: int = 3) -> TrainedBundle:
    """Fit the chosen model of every row task on the races that ended before ``cutoff_utc``."""
    from . import registry

    cutoff = _as_utc(cutoff_utc)
    choices = {**BEST_MODEL, **(choices or {})}
    train = df[df["start_utc"] + RACE_SPAN < pd.Timestamp(cutoff)]
    if train.empty:
        raise ValueError(f"no race ended before {cutoff}")
    train_end = (train["start_utc"].max() + RACE_SPAN).to_pydatetime()
    assert_trained_before([train_end], cutoff)
    races = train[["race_id", "start_utc"]].drop_duplicates().sort_values("start_utc")["race_id"].tolist()
    bundle = TrainedBundle(train_end, cutoff, tuple(races), FEATURE_VERSION, _git_commit())
    for task in tasks if tasks is not None else registry.tasks(horizons=(1, 3)):
        if not task.models:
            continue
        cls = _pick(task, choices)
        data = task.select(train)
        bundle.models[task.name] = cls().fit(data, task.target)
        bundle.targets[task.name] = task.target
        bundle.metrics[task.name] = _holdout(task, cls, data, races, holdout)
    return bundle


def _holdout(task: Task, cls, data: pd.DataFrame, races: list[str], holdout: int) -> dict:
    out = {"model": cls.name, "kind": task.kind, "holdout_races": [], "scores": {}}
    if holdout and len(races) > holdout + 1:
        hold = races[-holdout:]
        fit, test = data[~data.race_id.isin(hold)], data[data.race_id.isin(hold)]
        if len(test) and len(fit):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = cls().fit(fit, task.target).predict(test)
            out["holdout_races"] = hold
            if task.kind == "survival":  # the model predicts a CDF [n, 15]; scored against the rows' observed times
                from .bench.laps_to_stop import survival_scores

                out["scores"] = survival_scores(test, np.asarray(pred, float))
            else:
                out["scores"] = _SCORERS[task.kind](test[task.target].to_numpy(float), np.asarray(pred, float))
    return out


# ----------------------------------------------------------------------------- files
def bundle_path(bundle: TrainedBundle, folder: Path | None = None) -> Path:
    stamp = bundle.cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    return (folder or models_dir()) / f"{bundle.feature_version}_{stamp}.pkl"


def save_bundle(bundle: TrainedBundle, path: Path | None = None) -> Path:
    path = Path(path) if path else bundle_path(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def load_bundle(path: Path, race_start_utc: datetime | None = None) -> TrainedBundle:
    """Load a bundle; with ``race_start_utc`` refuse it unless it was trained before that race."""
    bundle = pickle.loads(Path(path).read_bytes())
    if not isinstance(bundle, TrainedBundle):
        raise TypeError(f"{path} is not a TrainedBundle")
    if race_start_utc is not None:
        bundle.check_for(race_start_utc)
    return bundle


def latest_bundle_for(race_start_utc: datetime, folder: Path | None = None) -> TrainedBundle | None:
    """The most recently trained saved bundle that is allowed for this race, or None."""
    best = None
    for p in sorted((folder or models_dir()).glob("*.pkl")):
        try:
            b = load_bundle(p, race_start_utc)
        except Exception:
            continue
        if b.feature_version == FEATURE_VERSION and (best is None or b.train_end_utc > best.train_end_utc):
            best = b
    return best


# ----------------------------------------------------------------------------- CLI
def _cutoff(words: list[str] | None, df: pd.DataFrame) -> datetime:
    if not words:  # everything: after the last race in the benchmark has ended
        return (df["start_utc"].max() + RACE_SPAN + pd.Timedelta(seconds=1)).to_pydatetime()
    if len(words) == 2 and words[0].isdigit():
        from . import archive

        return archive.find_session(int(words[0]), words[1]).start_utc
    if len(words) == 1:
        return _as_utc(words[0])
    raise SystemExit("--before takes '<year> <race>' (e.g. 2026 hungary) or one UTC timestamp")


def cmd_train(a) -> None:
    from .bench.dataset import load_bench

    df = load_bench()
    bundle = train_bundle(df, _cutoff(a.before, df), holdout=a.holdout)
    path = save_bundle(bundle, Path(a.out) if a.out else None)
    print(bundle.describe())
    print(f"saved {path}")


def add_commands(sub) -> None:
    s = sub.add_parser("train", help="train the benchmark models once on races before a cutoff and save a bundle")
    s.add_argument("--before", nargs="+", metavar="YEAR RACE | UTC",
                   help="cutoff: a race start ('2026 hungary') or a UTC time; default: after every benchmark race")
    s.add_argument("--out", help="bundle file (default: data/models/<version>_<cutoff>.pkl)")
    s.add_argument("--holdout", type=int, default=3, help="last N training races used only for the stored metrics")
    s.set_defaults(fn=cmd_train)
