"""Qualifying engineer: cut line, drop zone, predicted cut time and send-now guidance.

Active only in Qualifying and Sprint Qualifying (the feed carries ``SessionPart``); in any other
session every method returns nothing. The logic lives in ``pitsense/quali.py``.

Race values: part, time_left_s, cut_pos, cut_time_s, pred_cut_s, bubble_s (gap between the cut car
and the first car out), n_on_track, traffic_ahead, running.
Car values: best_s, rank, gap_to_cut_s, in_zone, p_ko, send, laps_needed, slack_s, eligible.
"""

from __future__ import annotations

import math

from ... import quali as Q
from ..engineer import Engineer
from ..types import Alert

ALERT_WINDOW_S = 180.0  # drop-zone alerts start with this much time left


def fmt_clock(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


class QualiEngineer(Engineer):
    name = "quali"
    in_bench = False

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self.tracker = Q.QualiTracker()
        self.sprint = str(ctx.meta.get("session_name", "")) == "Sprint Qualifying"
        self._cache: tuple | None = None
        self.outline = ctx.meta.get("track_outline")  # {"x": [...], "y": [...]} from an earlier race

    # ------------------------------------------------------------------ plumbing
    def observe(self, state) -> None:
        self.tracker.observe(state)
        if state.topics.get("SessionInfo", {}).get("Name") == "Sprint Qualifying":
            self.sprint = True

    def _all(self, state):
        key = (id(state), self.tracker.n)
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        pic = Q.read(state, self.tracker)
        out = None
        if pic is not None:
            tr = self.tracker
            tr.sample_cut(pic.part, state.t, pic.cut_time)
            pred = Q.predict_cut(pic, self.sprint) if pic.cut_pos is not None else None
            out_s = Q.lap_estimates(state, pic) if pic.cut_pos is not None or pic.cars else None
            on_track = sum(1 for c in pic.cars.values() if not c.in_pit)
            waiting = sum(1 for c in pic.cars.values() if c.in_pit and (c.best is None or c.in_zone))
            traffic = self._traffic(state, on_track, {n for n, c in pic.cars.items() if not c.in_pit})
            guide = {n: Q.guidance(state, pic, c, pred, on_track, out_s, max(waiting - 1, 0)) for n, c in pic.cars.items()}
            for n, c in pic.cars.items():
                tr.mark(("elig", pic.part, n), state.t)
            out = (pic, pred, on_track, traffic, guide, out_s)
        self._cache = (key, out)
        return out

    def _traffic(self, state, on_track: int, out: set) -> dict:
        """Cars on track by the timing feed; with car positions and a circuit outline, also how many are in
        the first 40 % of the lap after the line (the cars in front of one leaving the pits now)."""
        res = {"n_on_track": on_track, "traffic_ahead": None}
        try:
            pos = state.feeds.telemetry.latest_position()
        except Exception:
            return res
        pts = [p for n, p in pos.items() if n in out and p.get("on_track") and p["x"] == p["x"] and p["y"] == p["y"]]
        if self.outline and pts:
            xs, ys = self.outline["x"], self.outline["y"]
            n = len(xs)
            cnt = 0
            for p in pts:
                i = min(range(n), key=lambda k: (xs[k] - p["x"]) ** 2 + (ys[k] - p["y"]) ** 2)
                if i / n < 0.4:
                    cnt += 1
            res["traffic_ahead"] = cnt
        return res

    # ------------------------------------------------------------------ values
    def race(self, state, view):
        a = self._all(state)
        if a is None:
            return {}
        pic, pred, on_track, traffic, _, _ = a
        bubble = (pic.first_out_time - pic.cut_time) if (pic.first_out_time is not None and pic.cut_time is not None) else None
        return {
            "part": pic.part, "time_left_s": None if pic.time_left is None else round(pic.time_left, 1),
            "cut_pos": pic.cut_pos, "cut_time_s": pic.cut_time,
            "pred_cut_s": pred, "bubble_s": None if bubble is None else round(bubble, 3),
            "n_on_track": traffic["n_on_track"], "traffic_ahead": traffic["traffic_ahead"],
            "running": pic.running, "sprint": self.sprint,
        }

    def car(self, state, number, view):
        a = self._all(state)
        if a is None:
            return {}
        pic, pred, _, _, guide, _ = a
        c = pic.cars.get(number)
        if c is None:
            return {"eligible": False}
        g = guide[number]
        return {
            "eligible": True, "best_s": c.best, "rank": c.rank, "gap_to_cut_s": c.gap_to_cut,
            "in_zone": c.in_zone, "p_ko": Q.p_knocked_out(c, pic, pred),
            "send": g["status"], "laps_needed": g["laps_needed"], "slack_s": g["slack_s"],
        }

    def alerts(self, state, view):
        a = self._all(state)
        if a is None:
            return []
        pic, pred, on_track, _, guide, _ = a
        tl = pic.time_left
        if not pic.running or tl is None or tl > ALERT_WINDOW_S or pic.cut_pos is None:
            return []
        out = []
        for n in pic.order:
            c, g = pic.cars[n], guide[n]
            if not g["needs_run"] or not (c.in_zone or c.best is None):
                continue
            tla = state.drivers[n].tla if n in state.drivers else n
            since = round(self.tracker.mark(("window", pic.part), state.t), 1)
            where = "no time set" if c.best is None else f"{c.gap_to_cut:+.3f}s to the cut"
            if g["status"] == "ON_TRACK":
                run = "on track, lap in progress" if g["laps_needed"] == 0 else "on his out-lap"
            elif g["status"] == "TOO_LATE":
                run = "too late for another lap"
            else:
                run = "needs one run (out-lap + push)"
            sev = "critical" if g["status"] in ("SEND_NOW", "TOO_LATE") else "warn"
            zone = "drop zone" if c.in_zone else "unsafe"
            out.append(Alert(state.t, self.name, "drop_zone" if g["status"] != "SEND_NOW" else "send_now", sev,
                             f"car {n} ({tla}) in the {zone} (P{c.rank}, {where}), {fmt_clock(tl)} left, {run}"
                             + (", SEND NOW" if g["status"] == "SEND_NOW" else ""),
                             car=n, since=since, data={"rank": c.rank, "slack_s": g["slack_s"]}))
        return out
