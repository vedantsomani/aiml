"""Fair call scoring, decision value (simulator counterfactual) and the versioned report, on synthetic races."""

import numpy as np

from pitsense.bench import callscore, decision
from pitsense.pitwall.engineers.strategy import analysis
from pitsense.pitwall.engineers.strategy.run import simulate_field
from pitsense.pitwall.engineers.strategy.sim import NOSTOP, Draws
from pitsense.pitwall.runtime import shadow_score
from pitsense.state import replay

from .conftest import make_log
from .test_strategy import _field

FACTS = {"stops": {"1": [10], "2": [20]}, "neutral": {"1": [], "2": [18, 19]}, "total": 30, "sc_race": True, "wet": False}


def _c(car, action, lap):
    return {"kind": "call", "car": car, "action": action, "car_lap": lap}


def test_each_action_is_scored_on_its_own_claim():
    s = lambda car, a, l: callscore.score_call(_c(car, a, l), FACTS)  # noqa: E731
    assert s("1", "BOX", 10)["right"] and s("1", "BOX", 8)["right"]
    late = s("1", "BOX", 12)  # called after the stop: legacy +-2 says right, the fair rule says wrong
    assert late["legacy_right"] and not late["right"]
    assert s("1", "PREPARE_BOX", 7)["right"] and not s("1", "PREPARE_BOX", 6)["right"]
    assert s("1", "STAY_OUT", 11)["right"] and not s("1", "STAY_OUT", 9)["right"]
    # BOX_IF_SC: no SC/VSC for car 1 in the window -> not scored, but "held" (it stayed out)
    u = s("1", "BOX_IF_SC", 3)
    assert not u["scored"] and u["right"] is None and u["triggered"] is False and u["held"]
    # car 2: SC on lap 18, stopped lap 20 (within 18..20) -> triggered and right; from lap 13 the SC is outside the window
    t = s("2", "BOX_IF_SC", 15)
    assert t["scored"] and t["triggered"] and t["right"]
    assert not s("2", "BOX_IF_SC", 13)["triggered"]


def test_summary_counts_box_if_sc_separately_and_recall_by_condition():
    calls = [_c("1", "BOX", 10), _c("1", "BOX_IF_SC", 3), _c("2", "BOX_IF_SC", 15), _c("2", "STAY_OUT", 5)]
    sc = callscore.score_race(calls, FACTS)
    per = [{"race": "r1", "facts": FACTS, **sc}, {"race": "r2", "facts": dict(FACTS, sc_race=False), **sc}]
    out = callscore.summarise(per)
    assert out["by_action"]["BOX"]["rate"] == 1.0
    assert out["by_action"]["BOX_IF_SC"]["n"] == 2  # triggered calls only (one per race)
    assert out["box_if_sc"] == {"calls": 4, "triggered": 2, "untriggered": 2, "held_when_untriggered": 1.0}
    assert out["recall"]["green"]["stops"] == 4 and out["recall"]["sc_vsc"]["stops"] == 0
    sp = callscore.splits(per)
    assert set(sp["safety_car"]) == {"SC/VSC race", "no SC/VSC"}
    assert set(sp["phase"]) <= {"early", "mid", "late"}


def test_bootstrap_is_deterministic_and_brackets_the_point():
    num, den = np.array([3, 5, 1, 4]), np.array([6, 7, 4, 5])
    a, b = callscore.boot_ratio(num, den), callscore.boot_ratio(num, den)
    assert a == b and a[1] <= a[0] <= a[2]
    assert callscore.boot_ratio(np.array([1]), np.array([2]))[1:] == (None, None)


def test_race_facts_on_the_synthetic_race():
    f = callscore.race_facts(replay(make_log()))
    assert f["stops"] == {"22": [3]} and f["sc_race"] and not f["wet"]
    assert any(f["neutral"].values())


def test_shadow_score_no_longer_counts_box_if_sc_as_a_box():
    final = replay(make_log())  # car 22 stops at the end of lap 3, SC around laps 5-6
    calls = [_c("22", "BOX", 3), _c("11", "BOX_IF_SC", 2)]
    r = shadow_score(calls, final, k=2)
    assert r["box_calls"] == 1 and r["box_precision"] == 1.0
    assert r["box_if_sc"]["calls"] == 1 and r["box_if_sc"]["triggered"] == 1 and r["box_if_sc"]["right"] == 0


def test_team_plan_and_sc_policy():
    assert decision.team_plan([(5, 1), (12, 2), (40, 0)], 6, 30) == ((12, 2),)
    assert decision.team_plan([(12, None)], 6, 30) is None  # a wet tyre: no dry counterfactual
    status = np.zeros((3, 10), dtype=np.int8)
    status[0, 2] = 2  # SC two laps after the next one
    status[2, 7] = 1  # VSC outside the 5-lap window
    stop, comp = decision.sc_policy(((25, 2),), status, 10, 40, np.array([18.0, 28.0, 38.0]), 1)
    assert stop[0, 0] == 13 and comp[0, 0] == 1
    assert stop[1, 0] == 25 and stop[2, 0] == 25 and stop[1, 1] == NOSTOP


def _moment(seed=0):
    F = _field(C=6, total=40, A=12)
    info = {"stops_done": {n: 0 for n in F.cars}, "stint_left": {n: None for n in F.cars}}
    D = Draws(F, 64, 1)
    run = simulate_field(F, D)
    a = analysis.analyse_focus(F, info, run, D, "2", None, 32, 8)
    D2 = Draws(F, 64, 99 + seed)
    return decision.evaluate_moment(F, D2, simulate_field(F, D2), 2, a, ((20, 2),))


def test_decision_value_on_a_synthetic_field_is_deterministic_and_sane():
    dv, dv2 = _moment(), _moment()
    assert {k: v["exp_pos"] for k, v in dv.items() if k != "_cur"} == {k: v["exp_pos"] for k, v in dv2.items() if k != "_cur"}
    assert {"now", "stay", "team", "sc"} <= set(dv)
    for k, v in dv.items():
        if k != "_cur":
            assert 1 <= v["exp_pos"] <= 6 and 0 <= v["p_lose2"] <= 1 and v["exp_pts"] >= 0
    for action in decision.OURS:
        m = decision.moment_metrics({"race": "r", "action": action, "car_lap": 13, "total": 40, "dv": dv, "team_next": 20})
        assert m["regret_pos"] >= 0 and m["regret_pts"] >= -1e-9
        assert 0 <= m["p_ours_ahead"] + m["p_team_ahead"] <= 1
    res = [{"moments": [{"race": "r", "action": "BOX", "car_lap": 13, "total": 40, "dv": dv, "finish": 3, "team_next": 20}]}]
    s = decision.summarise(res)
    assert s["with_estimate"] == 1 and s["calibration"]["n"] == 1


def test_holdout_audit_passes_and_report_renders(tmp_path):
    from pitsense.bench import report_all

    audit = report_all.holdout_audit(test_year=2026)
    assert audit and all(r["test_year_used_for_tuning"] is False for r in audit)
    rep = report_all.assemble({}, test_year=2026)
    md = report_all.render_md(rep)
    assert "Provenance" in md and "Hold-out audit" in md


def test_race_cache_is_atomic_and_survives_a_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSENSE_DATA", str(tmp_path))
    monkeypatch.setattr(decision, "run_race", lambda args: {"slug": args[0], "moments": [1]})
    assert decision._load_cached("x", "h") is None
    p = decision._cache_path("x", "h")
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\x80truncated")  # a run killed mid-write (old non-atomic writer)
    assert decision._load_cached("x", "h") is None and not p.exists()
    assert decision._cached_run(("x", 2026, "h", False))["moments"] == [1]
    assert decision._load_cached("x", "h")["slug"] == "x"
    assert not list(p.parent.glob("*.tmp*"))
