"""The post-race report: every pit-wall call graded against what happened, plus the team's race in numbers.

Inputs are a finished race log and either a pit-wall calls log (``pitsense pitwall`` writes one) or nothing, in which
case the race is replayed through the same runtime (as fast as possible) to produce the calls and alerts. Nothing
here feeds back into the engineers: the grading reads the finished race.
"""

from __future__ import annotations

import bisect
import re
import statistics as stats
import tempfile
from html import escape as esc
from pathlib import Path

from ..events import EventLog
from ..state import RaceState, replay
from . import svg
from .prerace import prerace, session_start, team_config
from .svg import BAD, OK, TEAM_COLS, WARN, chip, section, table

BOX_ACTIONS = ("BOX", "PREPARE_BOX", "BOX_IF_SC")
RADIO_WORDS = {  # what makes a radio message worth showing (weights)
    "box": 3, "pit": 2, "stop": 2, "undercut": 3, "overcut": 3, "safety car": 3, "vsc": 3, "tyre": 2, "tire": 2, "tyres": 2, "tires": 2,
    "degradation": 2, "graining": 3, "blistering": 3, "puncture": 4, "damage": 4, "engine": 3, "power": 2, "brake": 3, "brakes": 3,
    "gearbox": 3, "problem": 3, "issue": 3, "stay out": 3, "push": 1, "lift": 2, "save": 2, "fuel": 2, "gap": 1, "plan": 2, "rain": 3,
    "wet": 2, "slow": 1, "flag": 1, "penalty": 3, "leave": 1, "fast": 1, "strategy": 3, "soft": 1, "medium": 1, "hard": 1,
}


# ----------------------------------------------------------------------------- the race
def lap_clock(final: RaceState):
    """Leader lap end times, so a session time can be turned into the lap it fell in."""
    ends = sorted(x.t_end for x in final.laps if x.position == 1)

    def lap_at(t: float) -> int:
        return bisect.bisect_left(ends, t) + 1

    return lap_at


def replay_calls(log: EventLog, team_text: str, ref=None, *, timeout: float = 1800.0):
    """Run the race through the live runtime at full speed: (call and alert records, final state)."""
    from ..pitwall.runtime import PitWallRuntime, ReplaySource, load_call_log

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        rt = PitWallRuntime(ReplaySource(log, 0, ref=ref), team=team_config(team_text), log_dir=Path(tmp), models=True)
        rt.start()
        if not rt.wait(timeout):
            rt.stop()
            raise SystemExit("the replay did not finish in time")
        rt.stop()
        if rt.status == "error":
            raise SystemExit(f"replay failed: {rt.error}")
        recs = load_call_log(rt.log_path)
        meta = {"observe_errors": rt.observe_errors, "snapshot_errors": rt.snapshot_errors, "models": rt.model_info}
        return recs, rt.state, meta


def _stops(final: RaceState, car: str) -> list[dict]:
    laps = {x.lap: x for x in final.laps if x.driver == car}
    from ..pitloss import LapIndex, measure_stops

    idx = getattr(final, "_report_idx", None)
    if idx is None:
        idx = final._report_idx = LapIndex(final.laps)
        final._report_loss = {(s.driver, s.in_lap): s for s in measure_stops(idx, final.pit_events)}
    out = []
    for p in final.pit_events:
        if p.driver != car:
            continue
        nxt = laps.get(p.out_lap if p.out_lap is not None else p.in_lap + 1)
        rec = next((r for r in final.pit_stops if r.driver == car and r.lap in (p.in_lap, p.in_lap + 1) and (r.stop_time or r.lane_time)), None)
        s = final._report_loss.get((car, p.in_lap))
        out.append({"in_lap": p.in_lap, "out_lap": p.out_lap, "compound": nxt.compound if nxt else None, "red_flag": p.under_red,
                    "status": p.status_at_entry, "lane_time_s": None if p.lane_time is None else round(p.lane_time, 2),
                    "stationary_s": None if rec is None or rec.stop_time is None else round(rec.stop_time, 2),
                    "loss_s": None if s is None else s.loss, "condition": None if s is None else s.condition})
    return out


def _stints(final: RaceState, car: str, stops: list[dict], total: int) -> list[dict]:
    laps = {x.lap: x for x in final.laps if x.driver == car}
    last = max(laps, default=0)
    bounds = [0] + [s["in_lap"] for s in stops if not s["red_flag"] or True] + [last]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        rec = laps.get(a + 1) or laps.get(b)
        if rec is not None and b > a:
            out.append({"from": a, "to": b, "compound": rec.compound, "laps": b - a})
    return out


def _positions(final: RaceState, car: str) -> list[list]:
    d = final.drivers[car]
    pts = [[0, d.grid or None]]
    pts += [[x.lap, x.position] for x in sorted((x for x in final.laps if x.driver == car), key=lambda x: x.lap) if x.position]
    return pts


def car_story(final: RaceState, car: str, total: int) -> dict:
    d = final.drivers[car]
    stops = _stops(final, car)
    pos = _positions(final, car)
    return {"car": car, "tla": d.tla, "team": d.team, "grid": d.grid, "finish": d.position, "running": d.running,
            "laps_done": d.laps, "stops": stops, "stints": _stints(final, car, stops, total), "positions": pos}


# ----------------------------------------------------------------------------- grading
def grade_calls(calls: list[dict], final: RaceState, k: int = 2, cars: set[str] | None = None) -> dict:
    """Score each logged call as ``runtime.shadow_score`` does, but per call.

    A box call (BOX, PREPARE_BOX, BOX_IF_SC) from a car's lap L is right when that car's real stop (its in-lap,
    red-flag stops excluded) is within L-k..L+k; STAY_OUT is right when it has no stop in L..L+k.
    """
    from ..pitwall.runtime import shadow_score

    stops: dict[str, list[int]] = {}
    comp_at: dict[tuple[str, int], str | None] = {}
    laps = {(x.driver, x.lap): x for x in final.laps}
    for p in final.pit_events:
        if not p.under_red:
            stops.setdefault(p.driver, []).append(p.in_lap)
            o = laps.get((p.driver, p.out_lap if p.out_lap is not None else p.in_lap + 1))
            comp_at[(p.driver, p.in_lap)] = o.compound if o else None
    rows = []
    for c in calls:
        if c.get("kind") != "call" or c.get("action") not in (*BOX_ACTIONS, "STAY_OUT"):
            continue
        if cars and c["car"] not in cars:
            continue
        L = int(c.get("car_lap") or c.get("lap") or 0)
        mine = sorted(stops.get(c["car"], []))
        if c["action"] == "STAY_OUT":
            ok = not any(0 <= s - L <= k for s in mine)
            nearest = next((s for s in mine if s >= L), None)
        else:
            near = [s for s in mine if abs(s - L) <= k]
            ok = bool(near)
            nearest = min(near, key=lambda s: (abs(s - L), s)) if near else next((s for s in mine if s >= L), None)
        tla = final.drivers[c["car"]].tla if c["car"] in final.drivers else c["car"]
        rows.append({"t": c["t"], "lap": L, "car": c["car"], "tla": tla, "action": c["action"], "compound": c.get("compound"),
                     "confidence": c.get("confidence"), "right": ok, "next_stop_lap": nearest,
                     "stop_compound": comp_at.get((c["car"], nearest)) if nearest else None,
                     "tyre_match": bool(ok and c["action"] != "STAY_OUT" and c.get("compound") and c.get("compound") == comp_at.get((c["car"], nearest))),
                     "reason": (c.get("reasons") or [{}])[0].get("text") if c.get("reasons") else None})
    by: dict[str, dict] = {}
    for r in rows:
        a = by.setdefault(r["action"], {"calls": 0, "right": 0})
        a["calls"] += 1
        a["right"] += r["right"]
    for a in by.values():
        a["rate"] = round(a["right"] / a["calls"], 3) if a["calls"] else None
    box = [r for r in rows if r["action"] in BOX_ACTIONS]
    summary = shadow_score(calls, final, k, cars)
    summary["by_action"] = {a: by[a] for a in (*BOX_ACTIONS, "STAY_OUT") if a in by}
    summary["box_tyre_match"] = round(sum(r["tyre_match"] for r in box if r["right"]) / max(1, sum(r["right"] for r in box)), 3) if box else None
    return {"k": k, "calls": rows, "summary": summary}


def plan_vs_actual(pre_plans: dict | None, calls: list[dict], story: dict) -> dict:
    """The team's real stops against the pre-race Plan A and the last Plan A the wall had before the first stop."""
    real = [(s["in_lap"], s["compound"]) for s in story["stops"] if not s["red_flag"]]
    out: dict = {"actual": [[l, c] for l, c in real], "pre_race_plan_a": None, "wall_plan_a": None}
    plan_a = next((p for p in (pre_plans or {}).get("plans", []) if p["name"] == "A"), None)
    if plan_a:
        out["pre_race_plan_a"] = {"stops": plan_a["stops"], "exp_pos": plan_a["exp_pos"]}
        out["pre_race_delta"] = [[a[0] - b[0] if b else None, a[1] == b[1] if b else None] for a, b in zip(plan_a["stops"], real + [None] * len(plan_a["stops"]))]
        out["stops_actual_vs_plan"] = [len(real), len(plan_a["stops"])]
    first = real[0][0] if real else 10**6
    pa = [c for c in calls if c.get("kind") == "call" and c.get("car") == story["car"] and c.get("plan_a") and int(c.get("car_lap") or 0) <= first]
    if pa:
        p = pa[-1]["plan_a"]
        out["wall_plan_a"] = {"lap": pa[-1].get("car_lap"), "stops": [[s["lap"], s["compound"]] for s in p.get("stops", [])], "exp_pos": p.get("expected_position")}
    return out


# ----------------------------------------------------------------------------- alerts and radio
def mechanic_alerts(recs: list[dict], final: RaceState, team: set[str], lap_at) -> list[dict]:
    out = []
    for r in recs:
        if r.get("kind") != "alert" or not str(r.get("engineer", "")).startswith("mechanic"):
            continue
        car = r.get("car")
        d = final.drivers.get(car) if car else None
        since = r.get("since")
        out.append({"car": car, "tla": d.tla if d else car, "team": bool(car in team), "engineer": r["engineer"], "code": r.get("code"),
                    "severity": r.get("severity"), "message": r.get("message"), "since_t": since, "since_lap": lap_at(since) if since is not None else None,
                    "fired_t": r.get("t"), "fired_lap": r.get("lap"), "delay_s": round(r["t"] - since, 1) if since is not None and r.get("t") is not None else None,
                    "car_out": bool(d and not d.running), "car_laps": d.laps if d else None})
    out.sort(key=lambda a: (not a["team"], a["fired_t"] or 0, str(a["car"])))
    return out


def radio_highlights(final: RaceState, team: set[str], lap_at, limit: int = 12) -> dict:
    store = final.feeds.radio
    msgs = store._items  # (car, t, utc, path): every capture published
    rows = []
    for car, t, _utc, path in msgs:
        text = (store.transcripts.get(path) or "").strip()
        d = final.drivers.get(car)
        words = text.lower()
        score = sum(w for k, w in RADIO_WORDS.items() if re.search(r"\b" + re.escape(k) + r"\b", words)) + min(len(text.split()), 40) / 40
        rows.append({"car": car, "tla": d.tla if d else car, "t": round(t, 1), "lap": lap_at(t), "text": text, "score": round(score, 2), "team": car in team})
    mine = [r for r in rows if r["team"]]
    with_text = [r for r in mine if r["text"]]
    pick = sorted(sorted(with_text, key=lambda r: (-r["score"], r["t"]))[:limit], key=lambda r: r["t"])
    return {"total": len(rows), "team_total": len(mine), "team_with_transcript": len(with_text), "highlights": pick}


# ----------------------------------------------------------------------------- the report
def build_report(log: EventLog, team_text: str, *, calls: list[dict] | None = None, k: int = 2, ref=None, pre: dict | None = None,
                 sims: int = 288, voice=None) -> dict:
    meta_run = {}
    if calls is None:
        calls, final, meta_run = replay_calls(log, team_text, ref)
    else:
        final = replay(log)
    team = team_config(team_text)
    cars = team.focus(final)
    if not cars:
        raise SystemExit(f"no car matches team {team_text!r}")
    total = final.total_laps or max((x.lap for x in final.laps), default=0)
    pre = pre or prerace(log, team_text, sims=sims, voice=voice)
    lap_at = lap_clock(final)
    stories = {c: car_story(final, c, total) for c in cars}
    graded = grade_calls(calls, final, k, set(cars))
    field = grade_calls(calls, final, k, None)["summary"]
    leader = sorted((x for x in final.laps if x.position == 1), key=lambda x: x.lap)
    sc_laps = sorted({x.lap for x in leader if {"4", "5"} & set(x.track_status.split(","))})
    vsc_laps = sorted({x.lap for x in leader if {"6", "7"} & set(x.track_status.split(","))} - set(sc_laps))
    others = [{"car": n, "tla": d.tla, "positions": _positions(final, n)} for n, d in sorted(final.drivers.items()) if n not in cars]
    t_start = session_start(log)
    return {
        "race": pre["race"], "team": team_text, "team_name": pre["team_name"], "cars": cars, "total_laps": total,
        "source": "calls log" if not meta_run else "replay", "k": k, "replay": meta_run,
        "as_of_briefing": pre["as_of"], "session_start_t": t_start,
        "neutralised": {"sc_laps": sc_laps, "vsc_laps": vsc_laps},
        "cars_story": stories,
        "plan_vs_actual": {c: plan_vs_actual(((pre["plans"].get("cars") or {}).get(c)), calls, stories[c]) for c in cars},
        "pit_loss_prior": pre["circuit"]["pit_loss"],
        "field_stop_loss_median": _median_loss(final),
        "grading": graded, "grading_field": field,
        "mechanic_alerts": mechanic_alerts(calls, final, set(cars), lap_at),
        "radio": radio_highlights(final, set(cars), lap_at),
        "others_positions": others,
    }


def _median_loss(final: RaceState) -> dict:
    final_idx = [s for s in getattr(final, "_report_loss", {}).values()]
    out = {}
    for cond in ("green", "sc", "vsc"):
        xs = [s.loss for s in final_idx if s.condition == cond]
        out[cond] = round(stats.median(xs), 1) if xs else None
    return out


# ----------------------------------------------------------------------------- HTML
def _verdict(r: dict) -> str:
    return f'<span class="{"ok" if r["right"] else "bad"}">{"right" if r["right"] else "wrong"}</span>'


def _charts(rep: dict) -> str:
    total, cars = rep["total_laps"], rep["cars"]
    ymax = max([len(rep["others_positions"]) + len(cars), 20])
    series = [{"name": o["tla"], "colour": "#5b6b7e", "opacity": .35, "width": 1, "pts": [(l, p) for l, p in o["positions"] if p]} for o in rep["others_positions"]]
    markers = []
    for i, c in enumerate(cars):
        s = rep["cars_story"][c]
        col = TEAM_COLS[i % 4]
        series.append({"name": s["tla"], "colour": col, "width": 3, "pts": [(l, p) for l, p in s["positions"] if p], "label": s["tla"]})
        by_lap = {l: p for l, p in s["positions"] if p}
        for st in s["stops"]:
            if st["in_lap"] in by_lap:
                markers.append({"x": st["in_lap"], "y": by_lap[st["in_lap"]], "colour": col, "title": f"{s['tla']} pit stop, lap {st['in_lap']} ({st['compound'] or '?'})", "r": 5})
    bands = [(l - 1, l, WARN, f"safety car, lap {l}") for l in rep["neutralised"]["sc_laps"]] + [(l - 1, l, "#a78bfa", f"VSC, lap {l}") for l in rep["neutralised"]["vsc_laps"]]
    pos = svg.line_chart(series, x0=0, x1=total, y0=1, y1=ymax, y_inv=True, bands=bands, markers=markers, xlabel="lap (0 = grid)", ylabel="position",
                         label="position over the race", yticks=[1, 5, 10, 15, 20] if ymax >= 20 else None)
    leg = svg.legend([(rep["cars_story"][c]["tla"], TEAM_COLS[i % 4]) for i, c in enumerate(cars)] + [("other cars", "#5b6b7e"), ("safety car", WARN), ("VSC", "#a78bfa")])
    # actual stints with the calls over them
    rows, marks = [], []
    calls_by_car: dict[str, list] = {}
    for r in rep["grading"]["calls"]:
        calls_by_car.setdefault(r["car"], []).append(r)
    for i, c in enumerate(cars):
        s = rep["cars_story"][c]
        rows.append({"label": f"{s['tla']} #{c} actual", "stints": [(x["from"], x["to"], x["compound"], f"{x['compound']}: laps {x['from'] + 1}-{x['to']} ({x['laps']})") for x in s["stints"]]})
        for r in calls_by_car.get(c, []):
            colour = OK if r["right"] else BAD
            if r["action"] == "STAY_OUT" and r["right"]:
                continue  # keeps the lane readable: only the wrong stay-outs and every box call are marked
            marks.append({"row": i, "lap": r["lap"], "colour": colour, "title": f"lap {r['lap']} {r['action']} {r['compound'] or ''}: {'right' if r['right'] else 'wrong'}"
                          + (f" (real stop lap {r['next_stop_lap']})" if r["next_stop_lap"] else " (no stop followed)")})
    stints = svg.stint_chart(rows, total, marks=marks, label="stints and calls")
    # actual vs plan A
    prow = []
    for i, c in enumerate(cars):
        s = rep["cars_story"][c]
        pv = rep["plan_vs_actual"][c]
        prow.append({"label": f"{s['tla']} actual", "stints": [(x["from"], x["to"], x["compound"], f"{x['compound']} laps {x['from'] + 1}-{x['to']}") for x in s["stints"]]})
        pa = pv.get("pre_race_plan_a")
        if pa:
            first = s["stints"][0]["compound"] if s["stints"] else None
            bounds = [0] + [p[0] for p in pa["stops"]] + [total]
            comps = [first] + [p[1] for p in pa["stops"]]
            prow.append({"label": f"{s['tla']} Plan A (pre-race)", "stints": [(bounds[j], bounds[j + 1], comps[j], f"{comps[j]} laps {bounds[j] + 1}-{bounds[j + 1]}") for j in range(len(comps))]})
    plan = svg.stint_chart(prow, total, label="actual stints against pre-race Plan A")
    # pit loss per stop
    items = []
    for i, c in enumerate(cars):
        for st in rep["cars_story"][c]["stops"]:
            v = st["loss_s"] if st["loss_s"] is not None else st["lane_time_s"]
            if v is None:
                continue
            kind = st["condition"] or ("red flag" if st["red_flag"] else {"4": "safety car", "6": "VSC", "7": "VSC"}.get(st["status"], "green") + ", lane time")
            items.append((f"{rep['cars_story'][c]['tla']} L{st['in_lap']}", v, TEAM_COLS[i % 4], f"{v:.1f} s ({kind})"))
    pl = rep["pit_loss_prior"]
    items.append(("expected (green, prior)", pl["green"], "#5b6b7e", f"{pl['green']:.1f} s"))
    if rep["field_stop_loss_median"].get("green"):
        items.append(("field median (green)", rep["field_stop_loss_median"]["green"], "#5b6b7e", f"{rep['field_stop_loss_median']['green']:.1f} s"))
    loss = svg.bar_chart(items, w=700, label="pit loss per stop") if items else ""
    # shadow score
    sm = rep["grading"]["summary"]
    bars = [(f"{a} calls", v["rate"] or 0, OK if (v["rate"] or 0) >= .5 else WARN, f"{v['right']}/{v['calls']} right") for a, v in sm["by_action"].items()]
    if sm.get("stop_recall") is not None:
        bars.append(("stops with a box call", sm["stop_recall"], "#4c8dff", f"{sm['stop_recall']:.0%} of {sm['real_stops']}"))
    shadow = svg.bar_chart(bars, w=700, vmax=1.0, label="shadow score by action") if bars else '<div class="mut">no graded calls</div>'
    return dict(pos=pos, leg=leg, stints=stints, plan=plan, loss=loss, shadow=shadow)


def render(rep: dict) -> str:
    r = rep["race"]
    title = f"Post-race report: {r.get('year')} {r.get('meeting_name')} - {rep['team_name']}"
    ch = _charts(rep)
    total = rep["total_laps"]
    sm, gf = rep["grading"]["summary"], rep["grading_field"]
    head = "".join(f'<div class="kpi"><b style="color:{TEAM_COLS[i % 4]}">{esc(s["tla"])}: P{s["grid"]} to {("P%s" % s["finish"]) if s["running"] else "out"}</b><span>{len(s["stops"])} stop(s), {s["laps_done"]} laps</span></div>' for i, s in enumerate(rep["cars_story"].values()))
    p = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    head += (f'<div class="kpi"><b>{p(sm["box_precision"])}</b><span>box calls right (+-{rep["k"]} laps), {sm["box_calls"]} calls</span></div>'
             f'<div class="kpi"><b>{p(sm["stay_out_accuracy"])}</b><span>stay-out calls right, {sm["stay_out_calls"]} calls</span></div>'
             f'<div class="kpi"><b>{p(sm["stop_recall"])}</b><span>real stops covered by a box call ({sm["real_stops"]})</span></div>')
    summary = (f'<div class="grid">{head}</div><div class="foot">{len(rep["grading"]["calls"])} graded calls for the team (whole field: box precision {p(gf["box_precision"])}, '
               f'stay-out {p(gf["stay_out_accuracy"])}). Source: {esc(rep["source"])}. Red-flag stops are not scored; NO_CALL is not scored.</div>')
    race = section("The race", summary + "<h3>Position over the race</h3>" + ch["leg"] + ch["pos"])
    shadow = section("Shadow score: every call against what happened", ch["shadow"]
                     + '<div class="foot">A box call (BOX, PREPARE_BOX, BOX_IF_SC) from lap L is right if the car\'s real stop is within '
                       f'{rep["k"]} laps of L. STAY_OUT is right if there is no stop in the next {rep["k"]} laps. BOX_IF_SC is conditional but scored the same way.</div>'
                     + "<h3>Per-call timeline (diamonds: green right, red wrong; right stay-outs are not drawn)</h3>" + ch["stints"]
                     + _call_table(rep))
    plan = section("Actual stops against our Plan A", ch["plan"] + _plan_table(rep))
    stops = section("Pit loss per stop", ch["loss"] + _stop_table(rep))
    mech = section("Mechanic alerts", _mech(rep))
    radio = section("Driver radio highlights (real transcripts)", _radio(rep))
    return svg.page(title, f"PitSense pit wall. {total} laps. Graded against the finished race.", race + shadow + plan + stops + mech + radio)


def _call_table(rep: dict) -> str:
    rows = []
    for c in rep["grading"]["calls"]:
        if c["action"] == "STAY_OUT" and c["right"]:
            continue
        conf = "-" if c["confidence"] is None else f"{c['confidence']:.2f}"
        tail = f"real stop lap {c['next_stop_lap']}" + (f" ({esc(c['stop_compound'])})" if c["stop_compound"] else "") if c["next_stop_lap"] else "no stop followed"
        rows.append([str(c["lap"]), esc(c["tla"]), f"<b>{esc(c['action'])}</b>" + (" " + chip(c["compound"]) if c["compound"] else ""), conf, _verdict(c), esc(tail), esc(c["reason"] or "")])
    if not rows:
        return ""
    return ('<h3>Calls that were not a right stay-out</h3>' + table(["Lap", "Car", "Call", "Conf.", "Verdict", "What happened", "Main reason"], rows, num=(0, 3)))


def _plan_table(rep: dict) -> str:
    rows = []
    for c, pv in rep["plan_vs_actual"].items():
        tla = rep["cars_story"][c]["tla"]
        t = lambda xs: ", ".join(f"L{l} {k}" for l, k in xs) if xs else "no stop"  # noqa: E731
        pa, wp = pv.get("pre_race_plan_a"), pv.get("wall_plan_a")
        rows.append([esc(tla), esc(t(pv["actual"])), esc(t(pa["stops"])) + f" (P{pa['exp_pos']:.1f})" if pa else "-",
                     esc(t(wp["stops"])) + f" (lap {wp['lap']})" if wp else "-"])
    return table(["Car", "Actual stops", "Pre-race Plan A (expected finish)", "Wall's Plan A at the first stop"], rows)


def _stop_table(rep: dict) -> str:
    rows = []
    for c, s in rep["cars_story"].items():
        for st in s["stops"]:
            rows.append([esc(s["tla"]), str(st["in_lap"]), chip(st["compound"]), "-" if st["lane_time_s"] is None else f"{st['lane_time_s']:.1f}",
                         "-" if st["stationary_s"] is None else f"{st['stationary_s']:.1f}", "-" if st["loss_s"] is None else f"{st['loss_s']:.1f}",
                         esc(st["condition"] or ("red flag" if st["red_flag"] else {"4": "safety car (loss not measured)", "6": "VSC (loss not measured)", "7": "VSC (loss not measured)"}.get(st["status"], "green (loss not measured)")))])
    if not rows:
        return '<div class="mut">no stops</div>'
    return table(["Car", "In-lap", "Fitted", "Pit lane (s)", "Stationary (s)", "Loss vs field (s)", "Track state"], rows, num=(1, 3, 4, 5))


def _mech(rep: dict) -> str:
    al = rep["mechanic_alerts"]
    mine = [a for a in al if a["team"]]
    rest = [a for a in al if not a["team"]]
    def rows(xs):
        return [[esc(a["tla"] or "-"), esc(a["engineer"].replace("mechanic_", "")), f'<span class="{"bad" if a["severity"] == "critical" else "warn"}">{esc(a["severity"] or "")}</span>',
                 "-" if a["since_lap"] is None else f"lap {a['since_lap']}", "-" if a["fired_lap"] is None else f"lap {a['fired_lap']}",
                 "-" if a["delay_s"] is None else f"{a['delay_s']:.0f}", esc(a["message"] or ""), "out" if a["car_out"] else "running"] for a in xs]
    head = ["Car", "Engineer", "Severity", "Evidence from", "Fired", "Delay (s)", "Message", "Car at the flag"]
    out = ""
    out += ("<h3>Our cars</h3>" + table(head, rows(mine), num=(5,))) if mine else '<div class="mut">No mechanic alert fired for our cars.</div>'
    if rest:
        out += f'<h3>Rest of the field ({len(rest)})</h3>' + table(head, rows(rest[:12]), num=(5,))
    return out + '<div class="foot">"Evidence from" is when the problem first showed in the data; "Fired" is when the pit wall raised the alert. The replay reads timing, radio and race control; telemetry only if CarData was in the log.</div>'


def _radio(rep: dict) -> str:
    rd = rep["radio"]
    if not rd["highlights"]:
        return f'<div class="mut">{rd["team_total"]} radio clips for our cars, none with a transcript on disk.</div>'
    out = [f'<div class="foot">{rd["team_with_transcript"]} of {rd["team_total"]} clips from our cars have a transcript ({rd["total"]} clips in the race). Most relevant shown, in time order.</div>']
    for h in rd["highlights"]:
        out.append(f'<div class="quote"><span class="mut">lap {h["lap"]} - {esc(h["tla"])}</span><br>{esc(h["text"])}</div>')
    return "".join(out)
