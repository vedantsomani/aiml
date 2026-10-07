"""Rival strategist: who is around each car, who is about to stop, who threatens an undercut.

Per car:
* neighbours: ``ahead`` / ``behind`` / ``gap_ahead`` / ``gap_behind``;
* ``pit_prob_1`` / ``pit_prob_3`` (and ``pit_prob_5``): chance the car stops within 1 / 3 / 5 laps;
* ``undercut_threat``: chance the car behind gets ahead by stopping first (within 5 laps);
* ``undercut_chance``: the same for this car over the car in front;
* extras: ``in_pit_window``, ``typical_stint_laps``, ``team_cover_rate`` and the raw signals.

``pit_prob_k`` has two layers:
1. A *prior hazard* from earlier races (``summarize_race`` keeps every finished stint by compound;
   ``__init__`` turns the circuit's stints, shrunk toward all circuits, into "chance this stint ends
   within k laps given it has lasted ``a``"). Without history a stint-life default is used.
2. A logistic correction with in-race signals (SC / VSC, rivals stopping next to us, the team's habit
   of covering stops, tyre cliff risk, laps left, must-stop, traffic). Coefficients ``PIT_B`` were fitted
   on the 2025 season's labels (bench ``y_pit_k``) only.

Undercut: if the car behind (B) stops, it loses ``loss_now`` but then laps on fresh tyres while the
car in front (A) is still on old ones. A covers after about two laps, so B ends ahead if the
fresh-tyre gain over that time exceeds the gap. ``UC_B`` maps that margin (plus A's team habit) to a
probability, fitted on 2025. The chance that B stops first comes from the two cars' ``pit_prob_5``.

Everything is as-of ``state.t`` and deterministic.
"""

from __future__ import annotations

import bisect
import math

from ...state import RaceState
from ..engineer import Engineer
from ..types import Alert

DRY = ("SOFT", "MEDIUM", "HARD")
LIFE = {"SOFT": 18.0, "MEDIUM": 28.0, "HARD": 38.0}  # typical stint length without history, laps
HORIZONS = (1, 3, 5)
UC_HORIZON = 5  # laps within which the chaser must stop
UC_COVER_LAPS = 2  # laps the defender usually stays out after the chaser stops
RECENT_LAPS = 2  # a stop this recent counts as "just pitted"
COVER_WINDOW = 2  # a stop by the car directly behind within this many laps is a cover
SHRINK_CIRCUIT = 6.0  # pseudo-stints pulling a circuit's hazard toward all circuits
SHRINK_POOL = 5.0  # pseudo-stints pulling all circuits toward the default life
SHRINK_TEAM = 12.0  # pseudo-opportunities pulling a team's cover rate toward the field's
FEATS = ("lp", "sc", "vsc", "ahead_pit", "behind_pit", "mate_pit", "cover", "cliff", "late", "must",
         "stuck", "threat", "stops", "rem", "ratio")

# logistic on FEATS, per horizon (intercept first). Fitted on 2025 labels.
PIT_B: dict[int, tuple[float, ...]] = {  # fitted on the 2025 season, history prior from the other 2025 races
    1: (-3.392, 0.406, 0.938, -0.447, 0.456, 0.469, 0.651, -0.036, -0.045, -1.859, 0.095, -0.125, 0.104, -0.324, 0.837, 0.819),
    3: (-2.66, 0.421, 0.156, -0.744, 0.337, 0.337, 0.24, -0.024, 0.017, -2.345, 0.308, -0.163, 0.093, -0.186, 0.813, 0.742),
    5: (-2.356, 0.402, 0.098, 0.007, 0.265, 0.207, 0.081, -0.001, -0.081, -1.922, 0.371, -0.181, 0.067, -0.179, 0.825, 0.759),
}
# undercut success given the chaser stops first: intercept, margin (s), defender team's cover rate
UC_B = (-0.293, 0.319, -1.778)
# calibration of 'attacker stops first within UC_HORIZON': a + b * logit(raw)
FIRST_B = (-0.543, 0.989)


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _r(x, nd: int = 3):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def _default_p(compound: str, a: int, k: int) -> float:
    life = LIFE.get(compound, 22.0)
    return min(0.95, max(0.01, k / max(k + 1.0, life + 3.0 - a)))


class _Stints:
    """Finished and ongoing stints of one (circuit, compound), as sorted lengths."""

    __slots__ = ("done", "open")

    def __init__(self) -> None:
        self.done: list[int] = []  # stints that ended with a stop
        self.open: list[int] = []  # stints still running at the flag (censored)

    def sort(self) -> None:
        self.done.sort()
        self.open.sort()

    def lap(self, j: int) -> tuple[int, int]:
        """(stints that ended with a stop on their j-th lap, stints that reached their j-th lap)."""
        d, o = self.done, self.open
        ev = bisect.bisect_right(d, j) - bisect.bisect_left(d, j)
        risk = (len(d) - bisect.bisect_left(d, j)) + (len(o) - bisect.bisect_left(o, j))
        return ev, risk


class RivalsEngineer(Engineer):
    name = "rivals"
    requires = ("tyre", "pitstop")
    # No declared benchmark features: on 2025 none of these keys improved the core gbm_hazard (k=3 got worse),
    # so the core leaderboard stays as it was. The rivals models in bench/rivals.py name the keys they use.
    features = ()

    # ------------------------------------------------------------------ learning from finished races
    @classmethod
    def summarize_race(cls, final: RaceState, meta: dict) -> dict:
        """Stints by compound ([laps, ended with a stop]) and how often each team covered a stop."""
        stops: dict[str, list[int]] = {}
        for pe in final.pit_events:
            if not pe.under_red:
                stops.setdefault(pe.driver, []).append(pe.in_lap)
        laps: dict[str, dict[int, object]] = {}
        for x in final.laps:
            laps.setdefault(x.driver, {})[x.lap] = x
        stints: dict[str, list[list[int]]] = {}
        for drv, by_lap in laps.items():
            start = 0
            for s in sorted(set(stops.get(drv, ()))):
                rec = by_lap.get(s)
                if rec is not None and rec.compound and s - start >= 1:
                    stints.setdefault(rec.compound, []).append([s - start, 1])
                start = s
            d = final.drivers.get(drv)
            last = max(by_lap, default=0)
            if d is not None and d.running and last > start and by_lap.get(last) is not None:
                comp = by_lap[last].compound
                if comp:
                    stints.setdefault(comp, []).append([last - start, 0])
        # covers: a car within two places behind a rival that stops (green, not late) stops within COVER_WINDOW laps
        total = final.total_laps or 0
        teams: dict[str, list[int]] = {}
        for pe in final.pit_events:
            if pe.under_red or pe.status_at_entry != "1" or (total and pe.in_lap > total - 8) or pe.in_lap < 3:
                continue
            me = laps.get(pe.driver, {}).get(pe.in_lap - 1)
            if me is None or me.position is None:
                continue
            for x in final.drivers.values():
                rec = laps.get(x.number, {}).get(pe.in_lap - 1)
                if rec is None or not 1 <= (rec.position or 0) - me.position <= 2 or x.team == final.drivers[pe.driver].team:
                    continue
                mine = stops.get(x.number, [])
                if any(pe.in_lap - 1 <= s <= pe.in_lap for s in mine):  # stopped together: not a reaction
                    continue
                if laps.get(x.number, {}).get(pe.in_lap + COVER_WINDOW) is None:  # retired / race over
                    continue
                t = teams.setdefault(x.team, [0, 0])
                t[0] += 1
                t[1] += int(any(pe.in_lap < s <= pe.in_lap + COVER_WINDOW for s in mine))
        return {"stints": stints, "teams": teams, "circuit": meta.get("circuit_key")}

    # ------------------------------------------------------------------ setup from past races
    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self.circuit = ctx.meta.get("circuit_key")
        self._pool: dict[str, _Stints] = {}
        self._circ: dict[str, _Stints] = {}
        teams: dict[str, list[int]] = {}
        for race in ctx.past_races:
            extra = (race.extra or {}).get("rivals") or {}
            for comp, rows in (extra.get("stints") or {}).items():
                for length, ended in rows:
                    self._add(self._pool, comp, length, ended)
                    if race.circuit_key == self.circuit:
                        self._add(self._circ, comp, length, ended)
            for team, (n, c) in (extra.get("teams") or {}).items():
                t = teams.setdefault(team, [0, 0])
                t[0] += n
                t[1] += c
        for s in [*self._pool.values(), *self._circ.values()]:
            s.sort()
        n_all = sum(t[0] for t in teams.values())
        self._cover_all = (sum(t[1] for t in teams.values()) + 10 * 0.15) / (n_all + 10)
        self._teams = teams
        self._typical: dict[str, float] = {}
        for comp in DRY:
            src = self._circ.get(comp)
            if src is None or len(src.done) < 5:
                src = self._pool.get(comp)
            self._typical[comp] = float(src.done[len(src.done) // 2]) if src and len(src.done) >= 5 else LIFE[comp]

    @staticmethod
    def _add(table: dict[str, _Stints], comp: str, length: int, ended: int) -> None:
        s = table.setdefault(comp, _Stints())
        (s.done if ended else s.open).append(int(length))

    def prior_hazard(self, compound: str | None, age: int, k: int) -> float:
        """P(stint ends within k laps | it has lasted ``age`` laps): per-lap hazards from earlier races
        (circuit shrunk to all circuits shrunk to a default life), compounded over the k laps."""
        comp = compound or "UNKNOWN"
        pool, circ = self._pool.get(comp), self._circ.get(comp)
        stay = 1.0
        for j in range(age + 1, age + k + 1):
            h = _default_p(comp, j - 1, 1)
            if pool is not None:
                ev, risk = pool.lap(j)
                h = (ev + SHRINK_POOL * h) / (risk + SHRINK_POOL)
            if circ is not None:
                ev, risk = circ.lap(j)
                h = (ev + SHRINK_CIRCUIT * h) / (risk + SHRINK_CIRCUIT)
            stay *= 1.0 - min(0.9, h)
        return min(0.98, max(0.005, 1.0 - stay))

    def team_cover_rate(self, team: str) -> float:
        n, c = self._teams.get(team, (0, 0))
        return (c + SHRINK_TEAM * self._cover_all) / (n + SHRINK_TEAM)

    # ------------------------------------------------------------------ in-race
    def _signals(self, state: RaceState, number: str, view) -> dict | None:
        """Everything about one car that the hazard needs; cached per view (one moment)."""
        cache = view.__dict__.setdefault("_rivals", {})
        if number in cache:
            return cache[number]
        d = state.drivers.get(number)
        order = [x for x in state.running_order() if x.running]
        idx = {x.number: i for i, x in enumerate(order)}
        i = idx.get(number)
        if d is None or i is None:
            cache[number] = None
            return None
        ahead = order[i - 1] if i > 0 else None
        behind = order[i + 1] if i + 1 < len(order) else None
        tyre = view.car("tyre", number)
        age = in_stint = max(0, d.laps - d.stint_start_lap)  # laps on this set since it was fitted
        total = state.total_laps or 0
        remaining = (total - d.laps) if total else None
        # rivals that stopped in the last RECENT_LAPS laps, near us in the order at their entry
        ahead_pit = behind_pit = mate_pit = 0
        for k in range(len(state.pit_events) - 1, max(-1, len(state.pit_events) - 40), -1):
            pe = state.pit_events[k]
            if pe.driver == number or pe.under_red or pe.in_lap < d.laps - RECENT_LAPS + 1:
                continue
            other = state.drivers.get(pe.driver)
            if other is not None and other.team == d.team:
                mate_pit = 1
                continue
            pre = self.memory.order_at_pit.get(k, {})
            if pe.driver in pre and number in pre:
                delta = pre[pe.driver] - pre[number]
                if -2 <= delta < 0:
                    ahead_pit = 1
                elif 0 < delta <= 2:
                    behind_pit = 1
        status = state.track_status
        dry_used = {c for x in state.drivers.values() for c in x.compounds_used}
        race_dry = bool(dry_used) and dry_used <= set(DRY)
        must = int(race_dry and len(set(d.compounds_used) & set(DRY)) < 2)
        cover_rate = self.team_cover_rate(d.team)
        gap_a = d.interval if ahead is not None else None
        gap_b = behind.interval if behind is not None else None
        x = {
            "sc": int(status == "4"), "vsc": int(status in ("6", "7")),
            "ahead_pit": ahead_pit, "behind_pit": behind_pit, "mate_pit": mate_pit,
            "cover": cover_rate * ahead_pit,
            "cliff": float(tyre.get("cliff_risk") or 0.0),
            "must": must,
            "stuck": int(gap_a is not None and gap_a < 1.2),
            "threat": int(gap_b is not None and gap_b < 1.5),
            "stops": d.pit_stops,
            "rem": min(remaining, 40) / 40.0 if remaining is not None else 0.5,
            "ratio": min(2.0, age / self._typical.get(d.compound or "", LIFE.get(d.compound or "", 25.0))),
        }
        sig = {"d": d, "ahead": ahead, "behind": behind, "age": age, "in_stint": in_stint, "remaining": remaining,
               "x": x, "cover_rate": cover_rate, "gap_a": gap_a, "gap_b": gap_b}
        pit = {}
        for k in HORIZONS:
            prior = self.prior_hazard(d.compound, age, k)
            xs = dict(x, lp=_logit(prior), late=int(remaining is not None and remaining <= k + 1))
            b = PIT_B[k]
            z = b[0] + sum(b[1 + j] * xs[f] for j, f in enumerate(FEATS))
            p = _sig(z)
            if remaining is not None and remaining <= 0:
                p = 0.0
            pit[k] = (p, prior, xs)
        sig["pit"] = pit
        cache[number] = sig
        return sig

    def _fresh(self, tyre: dict) -> float | None:
        vals = [tyre.get(f"fresh_{c.lower()}_s") for c in DRY]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _margin(self, view, attacker: str, defender: str, gap: float | None) -> float | None:
        """Seconds the attacker would be ahead after the defender's cover delay, if it stopped now."""
        if gap is None:
            return None
        ta, td = view.car("tyre", attacker), view.car("tyre", defender)
        fresh, pace, deg = self._fresh(ta), td.get("pace_s"), td.get("deg_s_per_lap")
        if fresh is None or pace is None:
            return None
        deg = deg or 0.0
        gain = sum(pace + deg * j - fresh for j in range(1, UC_COVER_LAPS + 1))
        return gain - gap

    def _first(self, view, state, a: str, b: str) -> float | None:
        """P(a stops within UC_HORIZON laps and before b)."""
        sa, sb = self._signals(state, a, view), self._signals(state, b, view)
        if sa is None or sb is None:
            return None
        pa, pb = sa["pit"][5][0], sb["pit"][5][0]
        w = pa / (pa + pb) if pa + pb > 0 else 0.5
        return pa * (1 - pb) + pa * pb * w

    def _undercut(self, state, view, attacker: str | None, defender: str | None, gap: float | None):
        if attacker is None or defender is None:
            return None, None, None
        m = self._margin(view, attacker, defender, gap)
        first = self._first(view, state, attacker, defender)
        if first is None:
            return None, m, None
        first = _sig(FIRST_B[0] + FIRST_B[1] * _logit(first))
        if m is None:
            return None, m, first
        cover = self._signals(state, defender, view)["cover_rate"]
        succ = _sig(UC_B[0] + UC_B[1] * max(-30.0, min(30.0, m)) + UC_B[2] * cover)
        return first * succ, m, first

    def car(self, state, number, view):
        sig = self._signals(state, number, view)
        if sig is None:
            return {}
        d, ahead, behind = sig["d"], sig["ahead"], sig["behind"]
        pit = sig["pit"]
        threat, m_t, f_t = self._undercut(state, view, behind.number if behind else None, number, sig["gap_b"])
        chance, m_c, f_c = self._undercut(state, view, number, ahead.number if ahead else None, sig["gap_a"])
        typical = self._typical.get(d.compound or "", LIFE.get(d.compound or "", None))
        window = None
        if typical is not None and sig["remaining"] is not None:
            window = bool(sig["age"] >= 0.7 * typical and sig["remaining"] > 3)
        out = {
            "ahead": ahead.number if ahead else None,
            "behind": behind.number if behind else None,
            "gap_ahead": sig["gap_a"],
            "gap_behind": sig["gap_b"],
            "pit_prob_1": _r(pit[1][0], 4),
            "pit_prob_3": _r(pit[3][0], 4),
            "pit_prob_5": _r(pit[5][0], 4),
            "prior_pit_1": _r(pit[1][1], 4),
            "prior_pit_3": _r(pit[3][1], 4),
            "undercut_threat": _r(threat, 4),
            "undercut_chance": _r(chance, 4),
            "uc_margin_threat": _r(m_t, 3),
            "uc_margin_chance": _r(m_c, 3),
            "uc_first_threat": _r(f_t, 4),
            "uc_first_chance": _r(f_c, 4),
            "in_pit_window": window,
            "typical_stint_laps": typical,
            "team_cover_rate": _r(sig["cover_rate"], 4),
        }
        xs = pit[3][2]
        for f in ("sc", "vsc", "ahead_pit", "behind_pit", "mate_pit", "cover", "cliff", "must", "stuck", "threat", "stops", "rem", "ratio"):
            out[f] = _r(float(xs[f]), 4)
        out["lp_1"], out["lp_3"], out["lp_5"] = (_r(pit[k][2]["lp"], 4) for k in HORIZONS)
        return out

    def alerts(self, state, view):
        out = []
        for d in state.running_order():
            if not d.running or d.in_pit:
                continue
            sig = self._signals(state, d.number, view)
            if sig is None:
                continue
            threat = view.car(self.name, d.number).get("undercut_threat")
            if threat is not None and threat >= 0.35 and sig["behind"] is not None:
                out.append(Alert(
                    t=round(state.t, 3), engineer=self.name, code="undercut_threat",
                    severity="warn" if threat < 0.6 else "critical",
                    message=f"Car {sig['behind'].number} behind {d.number} may undercut ({threat:.0%})",
                    car=d.number, data={"threat": threat, "chaser": sig["behind"].number},
                ))
        return out
