import ast
import json
from pathlib import Path

import pytest

import pitsense.pitwall
from pitsense.bench.dataset import decision_rows
from pitsense.bench.features import FeatureBuilder
from pitsense.pitloss import PitLossPrior
from pitsense.pitwall import Context, Engineer, PitWall, TeamConfig
from pitsense.pitwall.types import ACTIONS
from pitsense.pitwall.wall import ordered
from pitsense.state import RaceState


def _replay(log, until=None, **ctx):
    wall = PitWall(Context(prior=PitLossPrior(), **ctx))  # every registered engineer
    state = RaceState(log.meta)
    for e in log.events:
        if until is not None and e.t > until:
            break
        state.apply(e)
        wall.observe(state)
    return state, wall


def test_every_engineer_runs_and_snapshots_are_plain_json(race_log):
    for until in (race_log.events[len(race_log) // 2].t, None):
        state, wall = _replay(race_log, until)
        assert [e.name for e in wall.engineers][-1] == "head"  # dependencies first
        snap = wall.snapshot(state)
        assert snap.tower and snap.cars and snap.calls
        assert all(c.action in ACTIONS for c in snap.calls)
        json.dumps(snap.to_dict(), allow_nan=False)  # strict JSON: no NaN, no live objects


def test_team_focus(race_log):
    state, wall = _replay(race_log, team=TeamConfig(team="team 22"))
    assert wall.snapshot(state).focus == ["22"]
    assert {c.car for c in wall.calls(state)} == {"22"}
    assert TeamConfig(team="team").match({"Team 11", "Team 22"}) is None  # ambiguous
    assert TeamConfig(team="TEAM 11").match({"Team 11", "Team 111"}) == "Team 11"  # exact name wins


def test_bench_rows_carry_engineer_values(race_log):
    rows, _ = decision_rows(race_log, PitLossPrior())
    row = next(r for r in rows if r["kind"] == "lap_end")
    assert {"pitstop__loss_now", "tyre__stint_pace", "rules__must_stop", "weather__rainfall"} <= row.keys()
    assert not any(k.startswith(("strategy__", "head__")) for k in row)  # slow engineers stay off the rows


class _Aliasing(Engineer):
    name = "aliasing"

    def car(self, state, number, view):
        return {"laps": state.laps}  # a live list that keeps growing after the decision


def test_engineers_may_only_return_scalars(race_log):
    def builder(prior, meta, ctx=None):
        return FeatureBuilder(prior, meta, ctx=ctx, engineers=[_Aliasing])

    with pytest.raises(ValueError, match="scalars only"):
        decision_rows(race_log, PitLossPrior(), builder=builder)


def test_dependency_order_and_errors():
    class A(Engineer):
        name = "a"

    class B(Engineer):
        name = "b"
        requires = ("a",)

    class Loop(Engineer):
        name = "loop"
        requires = ("loop",)

    class Orphan(Engineer):
        name = "orphan"
        requires = ("nobody",)

    assert [c.name for c in ordered([B, A])] == ["a", "b"]
    with pytest.raises(ValueError, match="cycle"):
        ordered([Loop])
    with pytest.raises(ValueError, match="isn't on this pit wall"):
        ordered([Orphan])


FUTURE = {"labels", "evaluate", "leakcheck", "dataset"}  # modules that read or score the future


# the runtime, its sources and its command line feed events to the wall (they read the log); nothing else may.
FEEDERS = {"runtime.py", "sources.py", "commands.py"}


def test_pitwall_code_never_imports_the_future():
    for path in Path(pitsense.pitwall.__file__).parent.rglob("*.py"):
        if path.name in FEEDERS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                assert name.split(".")[-1] not in FUTURE, f"{path.name} imports {name}"
                assert name not in ("load_archive_session", "EventLog"), f"{path.name} reads the event log"
