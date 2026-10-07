"""``pitsense results``: one versioned report with every headline number -> ``reports/results.json`` + ``results.md``.

Each component is a result file written by its own run and stamped by ``provenance.stamp`` (data version =
race count + manifest hash, config = strategy SETTINGS / sim PARAMS with hash, training cutoff, git commit):

=============  =================================================  ==========================================
component      file                                               produced by
=============  =================================================  ==========================================
benchmark      reports/leaderboard.json                           ``pitsense bench run`` (core tasks)
strategy       reports/strategy_bench/leaderboard.json            ``pitsense results --run strategy``
decision       reports/decision.json                              ``pitsense results --run decision``
voice          reports/voice_eval.json                            ``pitsense results --run voice``
quali          reports/quali/quali_bench.json                     ``pitsense results --run quali``
=============  =================================================  ==========================================

Every headline number in results.json carries the stamp of the file it came from; ``results.md`` prints it
next to each table. The README's results section is regenerated from results.md (``readme_sync``), so the
README never quotes a number this report does not hold. The hold-out audit checks that the test year is not
used to tune any component.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import provenance

OUT = Path("reports")
FILES = {"benchmark": "leaderboard.json", "strategy": "strategy_bench/leaderboard.json", "decision": "decision.json",
         "voice": "voice_eval.json", "quali": "quali/quali_bench.json"}

# headline rows: (label, component, task, model, metrics, baseline model, baseline metrics)
HEADLINES = [
    ("Car pits next lap", "benchmark", "pit_within_1", None, ("log_loss", "auc"), "base_rate", ("log_loss",)),
    ("Car pits within 3 laps", "benchmark", "pit_within_3", None, ("log_loss", "auc"), "base_rate", ("log_loss",)),
    ("Rejoin position after a stop", "benchmark", "position_after_stop", None, ("exact", "within_1"), "no_change", ("exact", "within_1")),
    ("Time a stop costs, s", "benchmark", "pit_loss", None, ("mae",), None, ("mae",)),
    ("Next lap time, s", "benchmark", "next_lap_time", None, ("mae",), None, ("mae",)),
    ("Lap time 5 laps ahead, s", "benchmark", "lap_time_5", None, ("mae",), None, ("mae",)),
    ("Tyre cliff within 3 laps", "benchmark", "tyre_cliff_3", None, ("auc", "log_loss"), None, ("auc",)),
    ("Pace on fresh tyres, s", "benchmark", "fresh_tyre_pace", None, ("mae",), None, ("mae",)),
    ("Undercut by the car behind within 5 laps", "benchmark", "undercut_5", None, ("auc",), None, ("auc",)),
    ("Rain within 10 minutes", "benchmark", "rain_10min", None, ("log_loss",), None, ("log_loss",)),
    ("Safety car within 2 laps", "benchmark", "sc_within_2", None, ("log_loss",), None, ("log_loss",)),
    ("Laps until next stop", "benchmark", "laps_to_stop", None, ("c_index",), None, ("c_index",)),
    ("Finishing position at 25/50/75 %", "strategy", "strategy_finish", "sim", ("rps", "mae"), "current_position", ("rps", "mae")),
    ("Next stop, laps", "strategy", "strategy_nextstop", "sim", ("mae_laps", "within_2"), "rivals_hazard", ("mae_laps",)),
    ("Box calls, top-5 finishers (legacy +-2 rule)", "strategy", "strategy_calls", "head", ("precision", "recall", "stay_out_acc"), None, ()),
]


# --------------------------------------------------------------------------- hold-out audit
def holdout_audit(test_year: int = 2026) -> list[dict]:
    """Every tuned component and what it was tuned on; automatic checks where the code allows them."""
    rows = []

    def add(component, tuned_on, check, ok, how):
        rows.append({"component": component, "tuned_on": tuned_on, "check": check, "passed": bool(ok),
                     "test_year_used_for_tuning": not ok, "how": how})

    # 1. benchmark models: expanding window, the split refuses training races that overlap the test race
    import pandas as pd

    from .evaluate import RACE_SPAN, _splits

    t0 = pd.Timestamp("2025-03-01", tz="UTC")
    df = pd.DataFrame({"race_id": [f"r{i}" for i in range(6)], "year": [2025] * 3 + [test_year] * 3,
                       "start_utc": [t0 + pd.Timedelta(14 * i, unit="D") for i in range(6)]})
    ok = all(tr.start_utc.max() + RACE_SPAN < te.start_utc.min() and (tr.year < test_year).sum() >= 3
             for _, tr, te in _splits(df, test_year, 1))
    add("benchmark models (all tasks)", "races that ended before each test race", "automatic", ok,
        "evaluate._splits on a synthetic calendar; fit_predict asserts the cutoff on every fit")
    # 2. the models bundle the simulator / head use in the test year
    from ..config import data_dir

    p = data_dir() / "models" / f"strategy_bench_{test_year}.pkl"
    if p.exists():
        from .. import modelstore
        from ..archive import downloaded_sessions

        starts = [r.start_utc for r in downloaded_sessions() if r.year == test_year and r.session_name == "Race"]
        b = modelstore.load_bundle(p)
        ok = bool(starts) and b.train_end_utc < min(starts)
        add("models bundle for the pit wall replays", f"races ended before {b.train_end_utc:%Y-%m-%d}", "automatic", ok,
            f"train_end_utc < first {test_year} race start")
    # 3. voice: the corpus split holds out every test-year race
    from ..voice.synth import split_races

    sit = {f"{y}-{i:02d}": (y,) for y in (2024, 2025, test_year) for i in range(4)}
    tr, va, te = split_races(sit)
    ok = all(sit[r][0] < test_year for r in tr + va) and all(sit[r][0] == test_year for r in te) and len(te) == 4
    add("voice (scratch model, LoRA fine-tune)", f"races before {test_year}", "automatic", ok, "voice.synth.split_races")
    # 4. hand-tuned settings: declared, with the doc that records the tuning
    for comp, years, doc in (
        ("strategy simulator + head thresholds (SETTINGS, sim.PARAMS)", "2025", "docs/engineers/strategy.md"),
        ("wet engine thresholds", "2018-2024 wet races", "docs/engineers/strategy.md"),
        ("safety-car engineer", "2025 (--select-year)", "docs/engineers/safetycar.md"),
        ("qualifying model", "2025 sessions", "bench/quali.py"),
        ("benchmark model settings", "2025", "README / docs/models.md"),
    ):
        add(comp, years, "declared", str(test_year) not in years, doc)
    add("decision value / fair call scoring", "nothing (windows and options fixed before scoring)", "declared", True,
        "bench/decision.py, bench/callscore.py")
    return rows


# --------------------------------------------------------------------------- components
def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_strategy(out: Path, test_year: int, jobs: int) -> None:
    from .evaluate import write_report
    from .strategy import collect, strategy_tasks

    collect(test_year, jobs=jobs)
    res = [t.evaluate(None, test_year=test_year, min_train_races=1) for t in strategy_tasks()]
    meta = {"test_year": test_year, "provenance": provenance.stamp(
        f"models bundle trained on races that ended before the first {test_year} race; simulator settings tuned on 2025")}
    write_report(res, out / "strategy_bench", meta)


def run_decision(out: Path, test_year: int, jobs: int) -> None:
    from . import decision

    res = decision.collect(test_year, jobs=jobs)
    rep = {"provenance": provenance.stamp(f"models bundle trained before the first {test_year} race; history as-of each race",
                                          protocol="grid top 4 per race, full simulator mode, decisions every 2nd lap, 288 fresh futures"),
           "decision": decision.summarise(res), "calls": decision.fair_scores(res),
           "races": [r["slug"] for r in res]}
    (out / "decision.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")


def run_voice(out: Path, test_year: int, jobs: int, src: Path | None = None) -> None:
    rep = _read(src) if src else None
    if rep is None:
        from ..voice.evaluate import evaluate

        rep = evaluate(None, 150, None, log=lambda *a: None)
    rep["provenance"] = provenance.stamp(f"voice models trained on races before {test_year}; every {test_year} race held out")
    (out / "voice_eval.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")


def run_quali(out: Path, test_year: int, jobs: int) -> None:
    from .quali import run

    res = run(None, write_model=False)
    res["provenance"] = provenance.stamp("qualifying model tuned and fitted on 2025 sessions; 2026 scored once")
    (out / "quali").mkdir(parents=True, exist_ok=True)
    (out / "quali" / "quali_bench.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


RUNNERS = {"strategy": run_strategy, "decision": run_decision, "voice": run_voice, "quali": run_quali}


# --------------------------------------------------------------------------- assembling
def _num(v):
    return round(float(v), 4) if isinstance(v, (int, float)) and v == v else None


def _board_rows(src: dict, task: str, model: str | None, metrics, base: str | None, base_metrics) -> dict | None:
    board = src.get(task)
    if not board:
        return None
    best = next((r for r in board if r.get("model") == model), None) if model else board[0]
    ref = None
    if base_metrics:
        ref = next((r for r in board if r.get("model") == base), None) if base else board[-1] if len(board) > 1 else None
        if ref is None and len(board) > 1:
            ref = board[-1]
    if best is None:
        return None
    return {"model": best.get("model"), "score": {m: _num(best.get(m)) for m in metrics if m in best},
            "baseline": ({"model": ref.get("model"), **{m: _num(ref.get(m)) for m in base_metrics if m in ref}} if ref and ref is not best else None),
            "n": best.get("n") or best.get("races")}


def assemble(srcs: dict, test_year: int = 2026) -> dict:
    """``srcs``: component -> loaded result file (with its ``provenance`` / ``meta.provenance``)."""
    def prov(c):
        s = srcs.get(c) or {}
        return s.get("provenance") or (s.get("meta") or {}).get("provenance")

    head = []
    for label, comp, task, model, mets, base, bmets in HEADLINES:
        s = srcs.get(comp)
        row = _board_rows(s, task, model, mets, base, bmets) if s else None
        if row:
            head.append({"label": label, "task": task, **row, "provenance": prov(comp)})
    q = srcs.get("quali")
    if q and "ko_logloss" in q:
        ko = q["ko_logloss"]
        head.append({"label": "Qualifying knock-out", "task": "quali_ko", "model": "logit", "score": {"log_loss": ko.get("logit")},
                     "baseline": {"model": "rank", "log_loss": ko.get("rank")}, "n": (q.get("rows_2026") or {}).get("ko"),
                     "provenance": prov("quali")})
    v = srcs.get("voice")
    voice = None
    if v:
        voice = {"provenance": prov("voice"), "test_races": len(v.get("test_races", []))}
        for kind in ("template", "scratch", "llm"):
            r = v.get(kind)
            if r:
                voice[kind] = {"fallback_rate": r.get("overall_fallback_rate"),
                               "tasks": {t: {k: x.get(k) for k in ("n", "fact_error_rate", "fact_errors", "guard_pass", "bleu", "exact")
                                             if k in x} for t, x in (r.get("tasks") or {}).items()}}
    d = srcs.get("decision")
    sources = {c: prov(c) for c in FILES if srcs.get(c)}
    hashes = {(p or {}).get("data", {}).get("manifest_sha256") for p in sources.values() if p}
    cfgs = {(p or {}).get("config", {}).get("hash") for p in sources.values() if p}
    return {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "test_year": test_year,
            "current": provenance.stamp("n/a (the stamp of this report itself)"),
            "consistent": {"data": len(hashes) <= 1, "config": len(cfgs) <= 1, "data_hashes": sorted(h for h in hashes if h),
                           "config_hashes": sorted(c for c in cfgs if c)},
            "sources": sources, "headlines": head, "decision": (d or {}).get("decision"), "calls": (d or {}).get("calls"),
            "decision_provenance": prov("decision"), "voice": voice, "holdout_audit": holdout_audit(test_year)}


def load_sources(out: Path) -> dict:
    return {c: _read(out / f) for c, f in FILES.items() if (out / f).exists()}


# --------------------------------------------------------------------------- markdown
def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(c, nd=3):
    if not c or c.get("mean") is None:
        return "-"
    lo, hi = c.get("ci90") or [None, None]
    return f"{_f(c['mean'], nd)} [{_f(lo, nd)}, {_f(hi, nd)}]" if lo is not None else _f(c["mean"], nd)


def _rate(c):
    if not c or c.get("rate") is None:
        return f"- (n {c.get('n', 0) if c else 0})"
    lo, hi = c.get("ci90") or [None, None]
    ci = f" [{lo:.2f}, {hi:.2f}]" if lo is not None else ""
    return f"{c['right']}/{c['n']} = {c['rate']:.2f}{ci}"


def render_md(rep: dict) -> str:
    L = ["# PitSense results", "",
         f"Generated by `pitsense results` ({rep['generated_utc']}). Test year: **{rep['test_year']}**; every model and "
         "setting was trained or tuned without it (hold-out audit below). Each table names the run it came from.", ""]
    cons = rep.get("consistent") or {}
    L += [f"Data versions in this report: {', '.join(cons.get('data_hashes') or ['-'])}; configs: "
          f"{', '.join(cons.get('config_hashes') or ['-'])}" + ("" if cons.get("data") and cons.get("config") else
                                                                 "  **WARNING: components come from different data or configs.**"), ""]
    # benchmark
    L += ["<!-- results:benchmark -->", "## Benchmark tasks", "",
          "| Task | Model | Score | Baseline | n |", "|---|---|---|---|---|"]
    for h in rep.get("headlines", []):
        sc = ", ".join(f"{k} {_f(v)}" for k, v in h["score"].items())
        b = h.get("baseline")
        bs = (f"{b['model']}: " + ", ".join(f"{k} {_f(v)}" for k, v in b.items() if k != "model")) if b else "-"
        L.append(f"| {h['label']} | `{h['model']}` | {sc} | {bs} | {_f(h.get('n'))} |")
    L += ["", "Runs: " + "; ".join(f"{c}: {provenance.short(p)}" for c, p in (rep.get("sources") or {}).items()
                                    if p and c in ("benchmark", "strategy", "quali")), ""]
    # calls
    calls = rep.get("calls") or {}
    s = calls.get("summary")
    if s:
        L += ["<!-- results:calls -->", "## Call scores (fair rules)", "",
              f"Grid top 4 of each {rep['test_year']} race ({s['races']} races), calls every 2nd lap, logged on change. "
              "BOX: stop in L..L+2; PREPARE_BOX: stop in L..L+3; STAY_OUT: no stop in L..L+2; BOX_IF_SC: scored only when an "
              "SC/VSC came within L..L+4, then right with a stop within 2 laps of it. Rates with per-race bootstrap 90 % CIs.", "",
              "| Action | Right / scored |", "|---|---|"]
        for a in ("BOX", "PREPARE_BOX", "STAY_OUT", "BOX_IF_SC"):
            L.append(f"| {a} | {_rate(s['by_action'][a])} |")
        bi = s["box_if_sc"]
        L += [f"| BOX + PREPARE_BOX | {_rate(s['box_or_prepare'])} |",
              f"| legacy rule (+-2, BOX_IF_SC as a box) | {_rate(s['legacy_box_precision'])} |", "",
              f"BOX_IF_SC: {bi['calls']} calls, {bi['triggered']} triggered, {bi['untriggered']} untriggered "
              f"(car stayed out after {_f(bi['held_when_untriggered'], 2)} of those). Stop recall: all {_rate_r(s['recall']['all'])}, "
              f"green-flag {_rate_r(s['recall']['green'])}, under SC/VSC {_rate_r(s['recall']['sc_vsc'])}.", ""]
        sp = calls.get("splits") or {}
        for key, title in (("weather", "weather"), ("safety_car", "safety-car presence"), ("phase", "race phase"),
                           ("call_under_sc", "track status at the call"), ("circuit", "circuit")):
            t = sp.get(key)
            if not t:
                continue
            L += [f"By {title}:", "", "| | BOX | PREPARE_BOX | STAY_OUT | BOX_IF_SC |", "|---|---|---|---|---|"]
            for k, row in t.items():
                name = k.split("-grand-prix")[0] if key == "circuit" else k
                L.append(f"| {name} | " + " | ".join(_rate(row[a]) for a in ("BOX", "PREPARE_BOX", "STAY_OUT", "BOX_IF_SC")) + " |")
            L.append("")
    # decision value
    d = rep.get("decision")
    if d and d.get("all", {}).get("n"):
        L += ["<!-- results:decision -->", "## Decision value (simulator estimate)", "",
              "**These are model estimates, not observed outcomes.** At each decision moment the strategy simulator, on the "
              "as-of state, scores our call, its alternative, and what the team really did next, on the same 288 fresh "
              "futures. Regret: our option's expected finish minus the best option's (now / soon / later / box-if-SC / team). "
              "Negative 'vs team' = our call finishes ahead. Means with per-race bootstrap 90 % CIs.", "",
              "| Action | n | E[finish] | regret, places | regret, points | vs alternative, places | vs team, places | vs team, points | P(lose 2+) ours / team |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a, b in [("all", d["all"])] + list(d["by_action"].items()):
            if not b.get("n"):
                continue
            L.append(f"| {a} | {b['n']} | {_ci(b.get('exp_pos'), 2)} | {_ci(b.get('regret_pos'))} | {_ci(b.get('regret_pts'))} | "
                     f"{_ci(b.get('vs_alt_pos'))} | {_ci(b.get('vs_team_pos'))} | {_ci(b.get('vs_team_pts'))} | "
                     f"{_f((b.get('p_lose2') or {}).get('mean'), 3)} / {_f((b.get('team_p_lose2') or {}).get('mean'), 3)} |")
        c = d.get("calibration") or {}
        if c.get("n"):
            L += ["", "**How far to trust the simulator.** Its forecast for the plan the team really drove, against where the car "
                  f"really finished ({c['n']} moments, {c['races']} races):", "",
                  "| Check | Simulator | Current position |", "|---|---|---|",
                  f"| RPS (lower better) | {_ci(c['rps'], 4)} | {_ci(c['rps_current_position'], 4)} |",
                  f"| MAE of expected finish, places | {_ci(c['mae_pos'], 2)} | {_ci(c['mae_current_position'], 2)} |",
                  f"| bias (expected - real), places | {_ci(c['bias_pos'], 2)} | |",
                  f"| 80 % interval coverage | {_ci(c['coverage80'], 2)} | |",
                  f"| points bias (expected - real) | {_ci(c['points_bias'], 2)} | |", "",
                  f"PIT histogram (10 bins, flat = calibrated): {c['pit_histogram']}. P(lose 2+ places) reliability: "
                  + "; ".join(f"{r['bin']}: predicted {_f(r['predicted'], 2)}, observed {_f(r['observed'], 2)} (n {r['n']})"
                              for r in c["p_lose2_reliability"]) + ".", ""]
        L += ["Run: " + provenance.short(rep.get("decision_provenance") or {}), ""]
    # voice
    v = rep.get("voice")
    if v:
        L += ["<!-- results:voice -->", "## Voice", "", f"Held-out {rep['test_year']} situations ({v['test_races']} races).", "",
              "| Model | Task | n | fact errors | guard fallback |", "|---|---|---|---|---|"]
        for kind in ("template", "scratch", "llm"):
            r = v.get(kind)
            if not r:
                continue
            for t, x in r["tasks"].items():
                fe = x.get("fact_error_rate", x.get("fact_errors"))
                L.append(f"| {kind} | {t} | {_f(x.get('n'))} | {_f(fe)} | {_f(r.get('fallback_rate'))} |")
        L += ["", "Run: " + provenance.short(v.get("provenance") or {}), ""]
    # audit + provenance
    L += ["<!-- results:audit -->", "## Hold-out audit", "", "| Component | Tuned / trained on | Check | Test year untouched |", "|---|---|---|---|"]
    for r in rep.get("holdout_audit", []):
        L.append(f"| {r['component']} | {r['tuned_on']} | {r['check']} ({r['how']}) | {'yes' if r['passed'] else '**NO**'} |")
    L += ["", "## Provenance", ""]
    for c, p in (rep.get("sources") or {}).items():
        if p:
            L.append(f"* **{c}**: {provenance.short(p)} (run {p.get('run_utc')})")
    cur = rep.get("current") or {}
    L += [f"* this report: {provenance.short(cur)}", "",
          "Config of the current checkout (strategy SETTINGS): `" + json.dumps((cur.get("config") or {}).get("SETTINGS"), default=str) + "`", ""]
    return "\n".join(L)


def _rate_r(c):
    return f"{c['covered']}/{c['stops']}" + (f" = {c['rate']:.2f}" if c.get("rate") is not None else "")


def write(out: Path = OUT, test_year: int = 2026) -> tuple[Path, Path]:
    rep = assemble(load_sources(out), test_year)
    j, m = out / "results.json", out / "results.md"
    j.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    m.write_text(render_md(rep), encoding="utf-8")
    return j, m


def cmd_results(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in a.run or ():
        if c == "voice" and a.voice_json:
            run_voice(out, a.test_year, a.jobs, Path(a.voice_json))
        else:
            RUNNERS[c](out, a.test_year, a.jobs)
    j, m = write(out, a.test_year)
    if a.readme:
        from .readme_sync import sync

        sync(m, Path(a.readme))
    print(f"wrote {j} and {m}")


def add_commands(sub) -> None:
    s = sub.add_parser("results", help="one versioned report of every headline number -> reports/results.json + results.md")
    s.add_argument("--out", default="reports")
    s.add_argument("--test-year", type=int, default=2026)
    s.add_argument("--run", nargs="*", choices=sorted(RUNNERS), help="(re)compute these components first")
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--voice-json", help="with --run voice: stamp an existing `pitsense voice eval --out` file instead of re-running")
    s.add_argument("--readme", nargs="?", const="README.md", help="also regenerate the README results section from results.md")
    s.set_defaults(fn=cmd_results)
