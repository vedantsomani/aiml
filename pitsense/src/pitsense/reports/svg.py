"""Inline SVG charts and the page shell for the pit-wall reports (no scripts, no external files).

Every chart is a string of SVG. Colours come from CSS classes defined in ``PAGE_CSS`` so the page is dark on
screen and light on paper. Hover text is a native ``<title>`` element.
"""

from __future__ import annotations

from html import escape as esc

COMPOUND_COL = {"SOFT": "#e5484d", "MEDIUM": "#f5c542", "HARD": "#dcdcdc", "INTERMEDIATE": "#3fb950", "WET": "#4c8dff"}
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
TEAM_COLS = ("#2dd4bf", "#f472b6", "#a78bfa", "#fb923c")  # our cars, in fixed order
OK, BAD, WARN = "#3fb950", "#e5484d", "#f5a623"

PAGE_CSS = """
:root{--bg:#0b0f14;--panel:#121923;--panel2:#18212d;--fg:#e6edf3;--mut:#8b98a8;--grid:#243040;--acc:#2dd4bf;--line:#2b3a4d}
@media print{:root{--bg:#fff;--panel:#fff;--panel2:#f3f5f7;--fg:#111;--mut:#555;--grid:#d5dae0;--acc:#0e7c70;--line:#c4ccd6}
 .sec{break-inside:avoid}body{font-size:11px}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif;padding:20px 16px}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 2px;letter-spacing:.5px}h2{font-size:15px;margin:0 0 10px;text-transform:uppercase;letter-spacing:1.2px;color:var(--acc)}
h3{font-size:13px;margin:12px 0 6px;color:var(--fg)}
.sub{color:var(--mut);margin-bottom:14px}
.sec{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.kpi{background:var(--panel2);border-radius:6px;padding:8px 10px}.kpi b{display:block;font-size:20px}.kpi span{color:var(--mut);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:4px 8px;text-align:left;border-bottom:1px solid var(--grid)}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.8px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-block;min-width:18px;text-align:center;border-radius:9px;padding:0 6px;font-weight:700;font-size:11px;color:#111}
.tag{display:inline-block;border-radius:4px;padding:0 6px;font-size:11px;font-weight:700;border:1px solid var(--line)}
.ok{color:#3fb950}.bad{color:#e5484d}.warn{color:#f5a623}.mut{color:var(--mut)}
.brief{border-left:3px solid var(--acc);padding:4px 12px;margin:6px 0;background:var(--panel2);border-radius:0 6px 6px 0}
.quote{border-left:3px solid var(--line);padding:2px 10px;margin:5px 0}
svg{max-width:100%;height:auto;display:block}svg text{fill:var(--mut);font:11px "Segoe UI",system-ui,sans-serif}
svg .t{fill:var(--fg)}svg .grid{stroke:var(--grid);stroke-width:1}svg .axis{stroke:var(--mut);stroke-width:1}
svg .ring{stroke:var(--panel)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--mut);margin:4px 0 8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.foot{color:var(--mut);font-size:12px;margin-top:10px}
"""


def page(title: str, subtitle: str, body: str) -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(title)}</title><style>{PAGE_CSS}</style></head><body><div class=\"wrap\">"
            f'<h1>{esc(title)}</h1><div class="sub">{esc(subtitle)}</div>{body}</div></body></html>\n')


def section(title: str, inner: str) -> str:
    return f'<div class="sec"><h2>{esc(title)}</h2>{inner}</div>'


def chip(compound: str | None) -> str:
    c = (compound or "").upper()
    return f'<span class="chip" style="background:{COMPOUND_COL.get(c, "#888")}">{COMPOUND_LETTER.get(c, "?")}</span>'


def legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def table(head: list[str], rows: list[list], num: tuple[int, ...] = ()) -> str:
    """Rows hold HTML-safe strings (callers escape what they built from data)."""
    th = "".join(f'<th class="{"n" if i in num else ""}">{h}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"n" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{tr}</table>"


def _f(x: float) -> str:
    return f"{x:.1f}"


def _ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    step = next((s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500) if s >= raw), 1000)
    t = (int(lo // step) + (0 if lo % step == 0 else 1)) * step
    out = []
    while t <= hi + 1e-9:
        out.append(float(t))
        t += step
    return out


class Plot:
    """A plotting area with linear scales; ``y_inv`` puts small values at the top (positions)."""

    def __init__(self, w: int, h: int, x0: float, x1: float, y0: float, y1: float, *, y_inv: bool = False,
                 left: int = 44, right: int = 12, top: int = 10, bottom: int = 28) -> None:
        self.w, self.h, self.l, self.r, self.t, self.b = w, h, left, right, top, bottom
        self.x0, self.x1, self.y0, self.y1, self.inv = x0, x1, y0, y1, y_inv
        self.parts: list[str] = []

    def X(self, x: float) -> float:
        return self.l + (x - self.x0) / max(self.x1 - self.x0, 1e-9) * (self.w - self.l - self.r)

    def Y(self, y: float) -> float:
        f = (y - self.y0) / max(self.y1 - self.y0, 1e-9)
        if self.inv:
            f = 1 - f
        return self.t + (1 - f) * (self.h - self.t - self.b)

    def add(self, s: str) -> None:
        self.parts.append(s)

    def axes(self, xlabel: str = "", ylabel: str = "", xticks: list[float] | None = None, yticks: list[float] | None = None) -> None:
        xs = xticks if xticks is not None else _ticks(self.x0, self.x1)
        ys = yticks if yticks is not None else _ticks(min(self.y0, self.y1), max(self.y0, self.y1), 5)
        for v in ys:
            y = self.Y(v)
            self.add(f'<line class="grid" x1="{self.l}" x2="{self.w - self.r}" y1="{_f(y)}" y2="{_f(y)}"/><text x="{self.l - 6}" y="{_f(y + 4)}" text-anchor="end">{v:g}</text>')
        for v in xs:
            x = self.X(v)
            self.add(f'<line class="grid" x1="{_f(x)}" x2="{_f(x)}" y1="{self.t}" y2="{self.h - self.b}"/><text x="{_f(x)}" y="{self.h - self.b + 15}" text-anchor="middle">{v:g}</text>')
        self.add(f'<line class="axis" x1="{self.l}" x2="{self.w - self.r}" y1="{self.h - self.b}" y2="{self.h - self.b}"/>')
        if xlabel:
            self.add(f'<text x="{(self.l + self.w - self.r) / 2:.0f}" y="{self.h - 2}" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            cy = (self.t + self.h - self.b) / 2
            self.add(f'<text x="10" y="{cy:.0f}" text-anchor="middle" transform="rotate(-90 10 {cy:.0f})">{esc(ylabel)}</text>')

    def svg(self, label: str) -> str:
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">'
                + "".join(self.parts) + "</svg>")


def band(p: Plot, a: float, b: float, colour: str, title: str) -> None:
    x0, x1 = p.X(a), p.X(b)
    p.add(f'<rect x="{_f(x0)}" y="{p.t}" width="{_f(max(x1 - x0, 1))}" height="{p.h - p.t - p.b}" fill="{colour}" opacity=".16"><title>{esc(title)}</title></rect>')


def line_chart(series: list[dict], *, x0: float, x1: float, y0: float, y1: float, y_inv: bool = False, w: int = 1000, h: int = 330,
               xlabel: str = "", ylabel: str = "", bands: list[tuple] = (), markers: list[dict] = (), label: str = "chart",
               yticks: list[float] | None = None) -> str:
    """series: {name, colour, pts [(x, y)], width, opacity, label}; markers: {x, y, colour, title, r}; bands: (a, b, colour, title)."""
    p = Plot(w, h, x0, x1, y0, y1, y_inv=y_inv, right=40)
    p.axes(xlabel, ylabel, yticks=yticks)
    for a, b, col, title in bands:
        band(p, a, b, col, title)
    for s in series:
        pts = [pt for pt in s["pts"] if pt[1] is not None]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{_f(p.X(x))},{_f(p.Y(y))}" for i, (x, y) in enumerate(pts))
        p.add(f'<path d="{d}" fill="none" stroke="{s["colour"]}" stroke-width="{s.get("width", 2)}" opacity="{s.get("opacity", 1)}" stroke-linejoin="round"><title>{esc(s["name"])}</title></path>')
        if s.get("label"):
            x, y = pts[-1]
            p.add(f'<text class="t" x="{_f(p.X(x) + 4)}" y="{_f(p.Y(y) + 4)}">{esc(s["label"])}</text>')
    for m in markers:
        p.add(f'<circle class="ring" cx="{_f(p.X(m["x"]))}" cy="{_f(p.Y(m["y"]))}" r="{m.get("r", 4)}" fill="{m["colour"]}" stroke-width="2"><title>{esc(m.get("title", ""))}</title></circle>')
    return p.svg(label)


def stint_chart(rows: list[dict], total: int, *, w: int = 1000, label: str = "tyre stints", marks: list[dict] = ()) -> str:
    """rows: {label, stints [(start_lap, end_lap, compound, text)]}; laps run 0..total.
    marks: {row, lap, colour, title}: diamonds on a row (calls, real stops)."""
    rh, left = 26, 170
    h = 14 + rh * len(rows) + 26
    p = Plot(w, h, 0, total, 0, 1, left=left, right=14, top=10, bottom=26)
    p.axes("lap", yticks=[])
    for i, r in enumerate(rows):
        y = p.t + i * rh + 3
        p.add(f'<text class="t" x="{left - 8}" y="{y + 14}" text-anchor="end">{esc(r["label"])}</text>')
        for a, b, comp, text in r["stints"]:
            x0, x1 = p.X(a), p.X(b)
            col = COMPOUND_COL.get((comp or "").upper(), "#7b8794")
            p.add(f'<rect x="{_f(x0)}" y="{y}" width="{_f(max(x1 - x0 - 1, 1))}" height="{rh - 8}" rx="3" fill="{col}"><title>{esc(text)}</title></rect>')
            if x1 - x0 > 26:
                p.add(f'<text x="{_f((x0 + x1) / 2)}" y="{y + 13}" text-anchor="middle" style="fill:#111;font-weight:700">{COMPOUND_LETTER.get((comp or "").upper(), "?")}</text>')
    for m in marks:
        y = p.t + m["row"] * rh + 3 + (rh - 8) / 2
        x = p.X(m["lap"])
        p.add(f'<path d="M{_f(x)},{_f(y - 6)}L{_f(x + 5)},{_f(y)}L{_f(x)},{_f(y + 6)}L{_f(x - 5)},{_f(y)}Z" fill="{m["colour"]}" class="ring" stroke-width="1.5"><title>{esc(m.get("title", ""))}</title></path>')
    return p.svg(label)


def bar_chart(items: list[tuple], *, w: int = 520, label: str = "bars", vmax: float | None = None) -> str:
    """items: (label, value, colour, text) as horizontal bars."""
    rh, left = 22, 150
    h = 8 + rh * len(items)
    vmax = vmax or max([v for _, v, _, _ in items] + [1e-9])
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">']
    for i, (name, v, col, text) in enumerate(items):
        y = 4 + i * rh
        bw = max(0.0, (w - left - 90) * v / vmax)
        out.append(f'<text class="t" x="{left - 8}" y="{y + 13}" text-anchor="end">{esc(name)}</text><rect x="{left}" y="{y + 2}" width="{_f(bw)}" height="{rh - 8}" rx="3" fill="{col}"><title>{esc(text)}</title></rect>'
                   f'<text x="{_f(left + bw + 6)}" y="{y + 13}">{esc(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def range_chart(rows: list[dict], *, lo: float, hi: float, w: int = 520, label: str = "ranges") -> str:
    """rows: {label, colour, q: (min, p25, median, p75, max), n}: stint-length spread per compound."""
    rh, left = 28, 130
    h = 12 + rh * len(rows) + 24
    p = Plot(w, h, lo, hi, 0, 1, left=left, right=14, top=6, bottom=24)
    p.axes("laps", yticks=[])
    for i, r in enumerate(rows):
        y = p.t + i * rh + 8
        mn, q1, md, q3, mx = r["q"]
        p.add(f'<text class="t" x="{left - 8}" y="{y + 9}" text-anchor="end">{esc(r["label"])} (n={r["n"]})</text>')
        p.add(f'<line x1="{_f(p.X(mn))}" x2="{_f(p.X(mx))}" y1="{y + 5}" y2="{y + 5}" stroke="{r["colour"]}" stroke-width="2"><title>{mn:g} to {mx:g} laps</title></line>')
        p.add(f'<rect x="{_f(p.X(q1))}" y="{y - 2}" width="{_f(max(p.X(q3) - p.X(q1), 2))}" height="14" rx="3" fill="{r["colour"]}" opacity=".85"><title>middle half {q1:g} to {q3:g} laps</title></rect>')
        p.add(f'<line x1="{_f(p.X(md))}" x2="{_f(p.X(md))}" y1="{y - 3}" y2="{y + 13}" stroke="#111" stroke-width="2"><title>median {md:g} laps</title></line>')
    return p.svg(label)
