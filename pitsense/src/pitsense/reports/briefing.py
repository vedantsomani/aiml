"""The pre-race briefing page: one self-contained, printable HTML file from ``prerace.prerace``."""

from __future__ import annotations

from html import escape as esc

from . import svg
from .svg import COMPOUND_COL, TEAM_COLS, chip, section, table


def _n(x, nd=1, unit="") -> str:
    return "n/a" if x is None else f"{x:.{nd}f}{unit}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _plan_text(stops) -> str:
    return ", ".join(f"L{l} {c}" for l, c in stops) if stops else "no stop"


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><b>{esc(value)}</b><span>{esc(label)}</span></div>'


def _grid(d: dict) -> str:
    items = []
    mine = set(d["cars"])
    for g in d["grid"]:
        gap = g.get("gap_to_best_s")
        col = TEAM_COLS[0] if g["car"] in mine else "#5b6b7e"
        items.append((f"P{g['pos']} {g['tla']}", gap if gap is not None else 0.0, col, "pole time" if gap == 0 else (_n(gap, 3, " s") if gap is not None else "no time")))
    chart = svg.bar_chart(items, w=620, label="gap to the best qualifying lap") if items else ""
    rows = []
    for g in d["grid"]:
        star = " style=\"font-weight:700\"" if g["car"] in mine else ""
        tyre = (chip(g["tyre"]) + (" new" if g["tyre_new"] else " used" if g["tyre_new"] is False else "")) if g["tyre"] else '<span class="mut">unknown</span>'
        rows.append([f"<span{star}>P{g['pos']}</span>", f"<span{star}>{esc(g['tla'])} #{esc(g['car'])}</span>", esc(g["team"]), tyre,
                     _n(g["quali_s"], 3) if g["quali_s"] else "-", _n(g.get("gap_to_best_s"), 3) if g.get("gap_to_best_s") is not None else "-"])
    t = table(["Grid", "Car", "Team", "Starting tyre", "Best quali (s)", "Gap (s)"], rows, num=(4, 5))
    return f'<div class="grid" style="grid-template-columns:minmax(300px,1fr) minmax(300px,1fr);align-items:start"><div>{t}</div><div>{chart}</div></div>'


def _tyres(d: dict) -> str:
    rows = []
    for c in d["cars"]:
        t = d["tyres"].get(c) or {}
        who = next((g for g in d["grid"] if g["car"] == c), {})
        if t.get("new_soft_left") is None:
            rows.append([f"{esc(who.get('tla', c))} #{esc(c)}", "-", "-", "-", '<span class="mut">no earlier session on disk</span>'])
        else:
            rows.append([f"{esc(who.get('tla', c))} #{esc(c)}", str(t["new_soft_left"]), str(t["new_medium_left"]), str(t["new_hard_left"]), esc(t.get("used_sets") or "-")])
    note = ('<div class="foot">New sets left assume 13 dry sets (8 soft, 3 medium, 2 hard; 12 at a sprint weekend) and no sets handed back: '
            "the feed does not say, so these are upper bounds. Used sets are shown as compound letter plus laps run.</div>")
    return table(["Car", "New soft", "New medium", "New hard", "Used sets free (laps)"], rows, num=(1, 2, 3)) + note


def _circuit(d: dict) -> str:
    c = d["circuit"]
    sc, ot, pl = c["safety_car"], c["overtaking"], c["pit_loss"]
    src = pl["source"].replace("circuit:", "last visit ").replace("global", "all circuits")
    k = "".join([
        _kpi(_n(pl["green"], 1, " s"), f"pit loss, green ({src})"), _kpi(_n(pl["sc"], 1, " s"), "pit loss under the safety car"),
        _kpi(_n(pl["vsc"], 1, " s"), "pit loss under VSC"),
        _kpi(_pct(sc["p_sc_in_race"]), f"chance of a safety car ({sc['races_with_sc']} of {c['past_races_here']} races here)"),
        _kpi(_pct(sc["p_vsc_in_race"]), f"chance of a VSC ({sc['races_with_vsc']} of {c['past_races_here']})"),
        _kpi(f"{ot['label']} ({ot['ratio']:.1f}x)", f"overtaking: {_pct(ot['pass_chance_per_lap'])} a lap vs {_pct(ot['all_circuits'])} on average"),
    ])
    rows = [{"label": comp.title(), "colour": COMPOUND_COL[comp], "q": tuple(s["q"]), "n": s["n"]} for comp, s in c["stints"].items()]
    hi = max([r["q"][4] for r in rows] + [20])
    chart = svg.range_chart(rows, lo=0, hi=hi, label="typical stint lengths here") if rows else ""
    scope = ", ".join(f"{k.title()}: {v}" for k, v in c["stint_scope"].items())
    note = (f'<div class="foot">Stints are past stints at this circuit ({c["past_races_here"]} races: {", ".join(map(str, c["years_here"])) or "none"}); '
            f"with fewer than 6 stints of a compound all circuits are used ({esc(scope)}). Bar: middle half of stints, line: shortest to longest, tick: median. "
            f"Overtaking: {esc(ot['note'])}.</div>")
    return f'<div class="grid">{k}</div><h3>Typical stint length by tyre</h3>{chart}{note}'


def _weather(d: dict) -> str:
    w = d["weather"]
    now = w["now"]
    k = [_kpi(_n(now.get("AirTemp"), 1, " C"), "air"), _kpi(_n(now.get("TrackTemp"), 1, " C"), "track"),
         _kpi(_n(now.get("Humidity"), 0, " %"), "humidity"), _kpi("raining" if now.get("Rainfall") else "dry", "track now"),
         _kpi(_pct(w.get("rain_prob_10min")), "rain within 10 min (nowcast)"),
         _kpi(f"{d['circuit']['rain_history']['rained']} of {d['circuit']['rain_history']['races']}", "past races here with rain")]
    ser = w["series"]
    charts = ""
    if len(ser) >= 2:
        t0 = ser[0][0]
        for key, name, col in (("TrackTemp", "Track temperature (C)", "#f5a623"), ("AirTemp", "Air temperature (C)", "#4c8dff")):
            pts = [((t - t0) / 60.0, v.get(key)) for t, v in ser if v.get(key) is not None]
            if len(pts) >= 2:
                lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
                charts += (f"<h3>{esc(name)} before the start</h3>" + svg.line_chart(
                    [{"name": name, "colour": col, "pts": pts}], x0=0, x1=max(pts[-1][0], 1), y0=lo - 1, y1=hi + 1, w=520, h=150,
                    xlabel="minutes of weather data before lights out", label=name))
    return f'<div class="grid">{"".join(k)}</div>{charts}'


def _rain_note(d: dict) -> str:
    w = d["weather"]
    p = w.get("rain_prob_10min")
    if p is None:
        return ""
    return ""


def _strategy(d: dict) -> str:
    out = []
    plans = d["plans"]
    if not plans.get("ok"):
        return f'<div class="mut">No simulation: {esc(str(plans.get("why")))}</div>'
    total = d["total_laps"]
    for ci, car in enumerate(d["cars"]):
        e = plans["cars"].get(car)
        who = next((g for g in d["grid"] if g["car"] == car), {})
        out.append(f'<h3 style="color:{TEAM_COLS[ci % 4]}">{esc(who.get("tla", car))} #{esc(car)}: starts P{who.get("pos")} on {esc(who.get("tyre") or "unknown tyres")}</h3>')
        b = d["brief"].get(car) or {}
        if b.get("text"):
            out.append(f'<div class="brief">{esc(b["text"])}<div class="mut" style="font-size:11px">written by the voice ({esc(b.get("source", ""))}, checked against the facts)</div></div>')
        if not e or not e["plans"]:
            out.append(f'<div class="mut">No plan: {esc((e or {}).get("why") or "no data")}</div>')
            continue
        start = who.get("tyre") or "MEDIUM"
        rows, tab = [], []
        for p in e["plans"]:
            bounds = [0] + [s[0] for s in p["stops"]] + [total]
            comps = [start] + [s[1] for s in p["stops"]]
            rows.append({"label": f"Plan {p['name']}  P{p['exp_pos']:.1f}", "stints": [(bounds[i], bounds[i + 1], comps[i], f"{comps[i]} laps {bounds[i] + 1}-{bounds[i + 1]}") for i in range(len(comps))]})
            tab.append([f"<b>{p['name']}</b>", esc(_plan_text(p["stops"])), _n(p["exp_pos"], 2), _n(p["exp_pts"], 1), _pct(p["p_podium"]), _pct(p["p_points"]),
                        "-" if p["name"] == "A" else f"+{p['loss_vs_a']:.2f}"])
        out.append(svg.stint_chart(rows, total, label=f"plans for car {car}"))
        out.append(table(["Plan", "Stops (in-lap, tyre fitted)", "Exp. finish", "Exp. points", "Podium", "Points", "vs A (places)"], tab, num=(2, 3, 4, 5, 6)))
        sc = e.get("sc_play")
        if sc:
            gain = sc["places_gained"]
            verdict = f"gains {gain:.2f} places" if gain > 0.05 else "gains nothing worth a call" if gain > -0.05 else f"costs {-gain:.2f} places"
            out.append(f'<div class="foot">Safety car in the first laps: boxing for {esc(sc["stop"][1])} that lap {esc(verdict)} against staying on Plan A '
                       f"(expected finish P{sc['exp_pos']:.1f}).</div>")
    out.append('<div class="foot">Plans are ranked on 288 seeded simulated races (same engine as the live strategy engineer), '
               f"with pace from {esc(d['pace']['session'] or 'the grid order')} lap times x {d['model']['pace_k']}, wear from past stints here, and stop timing from this circuit's history. "
               "Treat expected positions as ranking aids, not forecasts.</div>")
    return "".join(out)


def _rivals(d: dict) -> str:
    out = []
    for car in d["cars"]:
        r = d["rivals"].get(car) or {}
        who = next((g for g in d["grid"] if g["car"] == car), {})
        out.append(f'<h3>{esc(who.get("tla", car))} #{esc(car)}</h3>')

        def rows(lst):
            return [[f"{esc(x['tla'])} #{esc(x['car'])}", esc(x["team"]), f"P{x['grid']}", chip(x["start_tyre"]) if x["start_tyre"] else "-",
                     "-" if x["pace_delta_s"] is None else f"{x['pace_delta_s']:+.3f}", _pct(x["p_ahead_at_flag"])] for x in lst]
        head = ["Rival", "Team", "Grid", "Tyre", "Quali pace vs us (s)", "Ahead of us at the flag"]
        if r.get("threats"):
            out.append("<div class=\"mut\">Threats from behind</div>" + table(head, rows(r["threats"]), num=(4, 5)))
        if r.get("targets"):
            out.append("<div class=\"mut\" style=\"margin-top:6px\">Targets ahead (lowest chance of finishing ahead of us first)</div>" + table(head, rows(r["targets"]), num=(4, 5)))
    out.append('<div class="foot">"Ahead of us at the flag" is the share of simulated races in which that car finishes in front while we run Plan A. '
               "Quali pace is the difference of best qualifying laps (positive: they were slower).</div>")
    return "".join(out)


def render(d: dict) -> str:
    r = d["race"]
    title = f"Pre-race briefing: {r.get('year')} {r.get('meeting_name')} - {d['team_name']}"
    a = d["as_of"]
    sub = ("PitSense pit wall. Only what was known at lights out"
           + (f" (session clock {a['session_start_t']:.0f} s, {a['events_used']} feed messages read)." if a["cut"] else " (no start marker in the feed: whole feed used)."))
    cars = ", ".join(f"#{c}" for c in d["cars"])
    body = "".join([
        section("Strategy: Plan A, B and C", _strategy(d)),
        section("Rival threats", _rivals(d)),
        section("Grid and starting tyres", _grid(d)),
        section(f"Tyre sets left ({cars})", _tyres(d)),
        section(f"Circuit: {r.get('circuit') or r.get('meeting_name')}, {d['total_laps']} laps", _circuit(d)),
        section("Weather outlook", _weather(d)),
    ])
    return svg.page(title, sub, body)
