"""Safety-car benchmark (owner: safetycar): "SC or VSC starts within the next 2 laps".

Labels read the finished race (the future) and are only used to score.

* ``y_sc_within_2``  lap-end rows: 1 if a safety car or VSC starts after the row and by the time the leader
                     completes two more laps, else 0. NaN while a neutralisation or red flag is already on.
* ``y_sc2`` / ``y_vsc2``  the same for SC only / VSC only.
* ``y_next_onset_t`` race time of the next SC / VSC start after the row (NaN: none), for lead-time scoring.

Rows of one lap are near-identical (the engineer's values are race-level), so the task keeps one row per
race lap: the first car across the line. ``python -m pitsense sc-report`` adds the warning table: how many
deployments were warned about before they started, lead time, and false alarms per race.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..pitwall.engineers.safetycar import ALERT_P, sample_labels, status_starts
from ..state import RaceState
from .metrics import CAL_BINS, binary_scores, calibration_table
from .models import BaseRate
from .tasks import Task

NEUTRAL = ("4", "5", "6", "7")


# ----------------------------------------------------------------------------- labels
def safetycar_labels(rows: list[dict], final: RaceState) -> None:
    first: dict[int, float] = {}
    for rec in final.laps:
        if rec.lap not in first or rec.t_end < first[rec.lap]:
            first[rec.lap] = rec.t_end
    laps = sorted(first)
    lap_t = [first[x] for x in laps]
    idx = {lap: i for i, lap in enumerate(laps)}
    starts = status_starts(list(final.status_log))
    allstarts = sorted(starts["sc"] + starts["vsc"])
    for row in rows:
        nan = float("nan")
        row["y_sc_within_2"] = row["y_sc2"] = row["y_vsc2"] = nan
        nxt = [s for s in allstarts if s > row["t"]]
        row["y_next_onset_t"] = nxt[0] if nxt else nan
        i = idx.get(row["lap"])
        if row.get("kind", "lap_end") != "lap_end" or i is None or row["track_status"] in NEUTRAL:
            continue
        sc, vsc = sample_labels(starts, lap_t, i)
        row["y_sc2"], row["y_vsc2"], row["y_sc_within_2"] = float(sc), float(vsc), float(sc or vsc)


# ----------------------------------------------------------------------------- models
def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(float) if name in df else np.full(len(df), default)


class CircuitBase:
    """Reference: the circuit's share of laps followed by an SC / VSC in earlier races, shrunk to all races."""

    name = "circuit_base_rate"
    PSEUDO = 40.0  # laps

    def fit(self, df, target):
        y = df[target].to_numpy(float)
        self.g = float(y.mean())
        by = df.assign(_y=y).groupby("circuit_key")["_y"].agg(["sum", "count"])
        self.rate = ((by["sum"] + self.PSEUDO * self.g) / (by["count"] + self.PSEUDO)).to_dict()
        return self

    def predict(self, df):
        return np.array([self.rate.get(c, self.g) for c in df["circuit_key"]], dtype=float)


class Engineer:
    """The safetycar engineer's own probability (SC or VSC)."""

    name = "engineer"

    def fit(self, df, target):
        return self

    def predict(self, df):
        return _col(df, "safetycar__neutral_prob_2laps", 0.02)


SIGNALS = ("safetycar__dy_sectors", "safetycar__y_sectors", "safetycar__yellow_age_s", "safetycar__stopped_msgs",
           "safetycar__incident_msgs", "safetycar__vehicle_msgs", "safetycar__loss_max_s", "safetycar__lap1_chaos",
           "safetycar__circuit_sc_rate", "safetycar__circuit_vsc_rate", "race_frac", "yellow", "lap")


class SignalGBM:
    """Gradient boosting on the same signals, trained on benchmark rows of earlier races (a check on the engineer's logistic)."""

    name = "gbm_signals"

    def fit(self, df, target):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.m = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.05, min_samples_leaf=40,
                                                l2_regularization=5.0, random_state=0)
        self.m.fit(self._x(df), df[target].astype(int))
        return self

    @staticmethod
    def _x(df):
        return np.column_stack([_col(df, c, 0.0) for c in SIGNALS])

    def predict(self, df):
        return self.m.predict_proba(self._x(df))[:, 1]


MODELS = (CircuitBase, BaseRate, Engineer, SignalGBM)


# ----------------------------------------------------------------------------- task
def sc_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per race lap (the first car across the line), neutralised laps left out."""
    d = df[(df["kind"] == "lap_end") & df["y_sc_within_2"].notna()]
    d = d.sort_values(["race_id", "lap", "t"]).drop_duplicates(["race_id", "lap"], keep="first")
    return d.copy()


def sc_tasks(**_) -> list[Task]:
    return [Task("sc_within_2", "binary", "y_sc_within_2", sc_rows, MODELS,
                 keep=("circuit_key", "y_sc2", "y_vsc2", "y_next_onset_t", "safetycar__sc_prob_2laps",
                       "safetycar__vsc_prob_2laps", "safetycar__neutral_prob_2laps", "safetycar__sc_reason",
                       "yellow", "total_laps"))]


# ----------------------------------------------------------------------------- warnings
def warning_table(P: pd.DataFrame, col: str, theta: float, warn=None) -> dict:
    """Lead time and false alarms for a warning that fires when ``P[col] >= theta`` (or ``warn(P)`` is true).

    A deployment is *warned* if the warning was up at some lap end before it started and within 2 laps of it.
    Lead time = deployment time minus the first such warning. A false alarm is a run of consecutive lap-end
    warnings none of which is followed by a deployment within 2 laps.
    """
    ev_n = ev_hit = 0
    leads, laps_ahead = [], []
    fa = 0
    races = P.race_id.nunique()
    for _, g in P.sort_values(["race_id", "t"]).groupby("race_id", sort=False):
        on = (g[col].to_numpy(float) >= theta) if warn is None else warn(g).to_numpy(bool)
        y = g["y_sc_within_2"].to_numpy(float) > 0
        t = g["t"].to_numpy(float)
        onset = g["y_next_onset_t"].to_numpy(float)
        # deployments seen from the rows: unique next-onset times of positive rows
        for s in sorted(set(onset[y])):
            ev_n += 1
            w = on & y & (onset == s)
            if w.any():
                ev_hit += 1
                first = int(np.argmax(w))
                leads.append(s - t[first])
                laps_ahead.append(int(w.sum()))
        # false-alarm episodes
        run_has_true = None
        for k in range(len(on)):
            if on[k]:
                run_has_true = y[k] if run_has_true is None else (run_has_true or y[k])
            elif run_has_true is not None:
                fa += not run_has_true
                run_has_true = None
        if run_has_true is not None:
            fa += not run_has_true
    return {"deployments": ev_n, "warned": ev_hit, "warned_share": ev_hit / ev_n if ev_n else float("nan"),
            "lead_median_s": float(np.median(leads)) if leads else float("nan"),
            "lead_min_s": float(np.min(leads)) if leads else float("nan"),
            "false_alarms": fa, "false_alarms_per_race": fa / races if races else float("nan"),
            "warning_rows_share": float((P[col].to_numpy(float) >= theta).mean()) if warn is None else float(warn(P).mean())}


def choose_theta(P: pd.DataFrame, col: str, max_fa_per_race: float = 1.0, max_rows_share: float = 0.05) -> float:
    """Smallest threshold with at most ``max_fa_per_race`` false alarms per race and a warning up in at most
    ``max_rows_share`` of the samples (chosen on the selection year)."""
    for th in np.round(np.arange(0.02, 0.9, 0.01), 2):
        w = warning_table(P, col, float(th))
        if w["false_alarms_per_race"] <= max_fa_per_race and w["warning_rows_share"] <= max_rows_share:
            return float(th)
    return 0.9


# ----------------------------------------------------------------------------- continuous (live-like) replay
def replay_race(slug: str, step: float = 10.0, feeds: bool = False) -> pd.DataFrame:
    """The engineer's values every ``step`` s of race time (the pit wall asks every update, not only at lap ends).

    Labels use the same rule as the lap-end task: an SC / VSC starts after t and before the leader has
    completed two more laps. Rows while a neutralisation or red flag is on are left out.
    """
    from bisect import bisect_right

    from .. import registry
    from ..archive import downloaded_sessions
    from ..config import bench_dir
    from ..events import load_archive_session
    from ..pitwall.engineer import Context
    from ..pitwall.engineers.safetycar import SafetyCarEngineer
    from ..pitwall.memory import RaceMemory
    from ..pitwall.wall import PitWall
    from ..state import replay
    from . import history as H

    ref = next(r for r in downloaded_sessions() if r.slug == slug and r.session_name == "Race")
    hist = H.load(bench_dir() / "history.json")
    prior = H.prior_for(hist, ref.circuit_key, ref.start_utc)
    log = load_archive_session(ref.local_dir, feeds=feeds)
    final = replay(log)
    first: dict[int, float] = {}
    for rec in final.laps:
        if rec.lap not in first or rec.t_end < first[rec.lap]:
            first[rec.lap] = rec.t_end
    lap_t = [first[k] for k in sorted(first)]
    starts = status_starts(list(final.status_log))
    onsets = sorted(starts["sc"] + starts["vsc"])
    ctx = Context.for_race(prior, log.meta, history=hist, race_start_utc=ref.start_utc)
    wall = PitWall(ctx, RaceMemory(), [SafetyCarEngineer])
    state = RaceState(log.meta)
    out, nxt = [], None
    events = log.events
    for k, e in enumerate(events):
        state.apply(e)
        wall.observe(state)
        if k + 1 < len(events) and events[k + 1].t <= e.t:
            continue
        if not state.started or state.finished_t is not None or state.t < (nxt if nxt is not None else -1):
            continue
        nxt = state.t + step
        if state.track_status in NEUTRAL:
            continue
        v = wall.race_values(state)
        i = bisect_right(lap_t, state.t)  # leader laps completed
        t1 = lap_t[i + 1] if i + 1 < len(lap_t) else float("inf")
        later = [s for s in onsets if state.t < s <= t1]
        nxt_on = [s for s in onsets if s > state.t]
        out.append({"race_id": slug, "t": state.t, "lap": i + 1, "y_sc_within_2": float(bool(later)),
                    "y_next_onset_t": nxt_on[0] if nxt_on else float("nan"),
                    "sc": v["safetycar__sc_prob_2laps"], "vsc": v["safetycar__vsc_prob_2laps"],
                    "engineer": v["safetycar__neutral_prob_2laps"], "warn": float(v["safetycar__sc_warn"]), "reason": v["safetycar__sc_reason"],
                    "yellow": float(state.track_status == "2"), "feed_stopped": v["safetycar__feed_stopped"]})
    return pd.DataFrame(out)


def _replay_job(args):
    return replay_race(*args)


def continuous_year(year: int, step: float = 10.0, feeds: bool = False, jobs: int = 3) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    from ..archive import downloaded_sessions

    slugs = [r.slug for r in downloaded_sessions() if r.year == year and r.session_name == "Race"]
    with ProcessPoolExecutor(jobs) as ex:
        frames = list(ex.map(_replay_job, [(s, step, feeds) for s in slugs]))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- report
def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) or v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def evaluate_year(df: pd.DataFrame, year: int, min_train: int = 10):
    from .evaluate import evaluate_task

    return evaluate_task(sc_tasks()[0], df, test_year=year, min_train_races=min_train)


def cmd_sc_report(a) -> None:
    from .dataset import load_bench

    warnings.filterwarnings("ignore")
    df = load_bench()
    sel = evaluate_year(df, a.select_year, a.min_train)
    sel_c = continuous_year(a.select_year, a.step, False, a.jobs) if a.continuous else None
    if a.theta is not None:
        theta, how = a.theta, "given"
    elif sel_c is not None:
        theta, how = choose_theta(sel_c, "engineer", a.max_fa), f"chosen on the {a.select_year} live-like replay: smallest with <= {a.max_fa} false alarms per race and a warning up in <= 5 % of samples"
    else:
        theta, how = ALERT_P, "the engineer's alert default"
    text = [f"Warning threshold on engineer P(SC or VSC within 2 laps): {theta:.2f} ({how}; alert default {ALERT_P:.2f}).\n"]
    for year in [a.select_year] + [y for y in a.test_years if y != a.select_year]:
        res = sel if year == a.select_year else evaluate_year(df, year, a.min_train)
        P = res.predictions
        text.append(f"## sc_within_2, {year}: {P.race_id.nunique()} races, {len(P)} laps, {int(P.y_sc_within_2.sum())} positive "
                    f"({P.y_sc_within_2.mean():.3f})\n")
        b = res.leaderboard
        text.append(_md(b[["model", "n", "positives", "log_loss", "brier", "brier_skill", "auc", "avg_precision"]].round(4)) + "\n")
        for who in ("engineer",):
            text.append(f"Calibration of {who}:\n\n" + _md(res.calibration[who].assign(bin=lambda x: x["bin"].astype(str)).round(4)) + "\n")
        rows = []
        for nm, col, th, w in (("engineer", "engineer", theta, None),
                               ("engineer @ alert default", "engineer", ALERT_P, None),
                               ("rule: status 2 (yellow)", "yellow", 0.5, None)):
            rows.append({"warning": nm, "theta": th, **warning_table(P, col, th, w)})
        text.append(_md(pd.DataFrame(rows).round(3)) + "\n")
        # per kind
        for kind in ("y_sc2", "y_vsc2"):
            from sklearn.metrics import roc_auc_score

            yk = P[kind].to_numpy(float)
            pk = P["safetycar__sc_prob_2laps" if kind == "y_sc2" else "safetycar__vsc_prob_2laps"].to_numpy(float)
            ok = 0 < yk.sum() < len(yk)
            text.append(f"{kind}: positives {int(yk.sum())}, AUC of its own probability {roc_auc_score(yk, pk) if ok else float('nan'):.3f}, "
                        f"mean predicted {pk.mean():.4f} vs observed {yk.mean():.4f}\n")
    if a.continuous:
        text.append(continuous_section(a, a.select_year, a.test_years, sel_c, theta))
    out = "\n".join(text)
    print(out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(out, encoding="utf-8")


def continuous_section(a, select_year: int, test_years: list[int], sel: pd.DataFrame, th: float) -> str:
    """Live-like scoring: the engineer asked every ``--step`` s. Threshold chosen on the selection year."""
    from .metrics import binary_scores

    text = [f"## Live-like replay (engineer asked every {a.step:.0f} s of race time)\n",
            f"Warning threshold on P(SC or VSC within 2 laps): {th:.2f}.\n"]
    for year in [select_year] + [y for y in test_years if y != select_year]:
        for feeds in ((False, True) if (a.feeds and year >= 2025) else (False,)):
            P = sel if (year == select_year and not feeds) else continuous_year(year, a.step, feeds, a.jobs)
            y = P.y_sc_within_2.to_numpy(float)
            sc = binary_scores(y, P.engineer.to_numpy(float))
            rows = [{"warning": "engineer alert (sc_warn)", "theta": ALERT_P, **warning_table(P, "warn", 0.5)},
                    {"warning": "engineer, probability only", "theta": th, **warning_table(P, "engineer", th)},
                    {"warning": "rule: status 2 (yellow)", "theta": 0.5, **warning_table(P, "yellow", 0.5)}]
            label = " with telemetry / position feeds" if feeds else ""
            text.append(f"### {year}{label}: {P.race_id.nunique()} races, {len(P)} samples, {int(y.sum())} positive; "
                        f"log loss {sc['log_loss']:.4f}, Brier {sc['brier']:.4f}, AUC {sc['auc']:.3f}\n")
            text.append(_md(pd.DataFrame(rows).round(3)) + "\n")
            if feeds:
                st = P[P.feed_stopped > 0]
                text.append(f"Rows with a car stopped on track (telemetry): {len(st)}, "
                            f"positive share {st.y_sc_within_2.mean() if len(st) else float('nan'):.3f}\n")
    return "\n".join(text)


def add_commands(sub) -> None:
    s = sub.add_parser("sc-report", help="safety-car engineer: sc_within_2 scores, lead time and false alarms")
    s.add_argument("--select-year", type=int, default=2025)
    s.add_argument("--test-years", type=int, nargs="+", default=[2026])
    s.add_argument("--theta", type=float, help="warning threshold (default: chosen on the selection year)")
    s.add_argument("--max-fa", type=float, default=1.0, help="false alarms per race allowed when choosing the threshold")
    s.add_argument("--min-train", type=int, default=10)
    s.add_argument("--continuous", action="store_true", help="also score the engineer asked every --step seconds (live-like)")
    s.add_argument("--step", type=float, default=10.0)
    s.add_argument("--feeds", action="store_true", help="also replay 2025-26 with telemetry / position feeds loaded")
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--out", help="markdown file to write")
    s.set_defaults(fn=cmd_sc_report)
