"""What past races teach the strategy engineer: stint lengths, safety-car hazard, how hard it is to pass.

``learn`` reads one finished race (``StrategyEngineer.summarize_race``); ``Priors`` pools the races in
``ctx.past_races`` (only those that ended before this one started) into the tables the simulator samples from.
All values are plain JSON in ``RaceSummary.extra["strategy"]``.
"""

from __future__ import annotations

import bisect
import math

DRY = ("SOFT", "MEDIUM", "HARD")
LIFE = {"SOFT": 18.0, "MEDIUM": 28.0, "HARD": 38.0}  # typical stint length without history, laps
PASS_BINS = (-0.3, 0.0, 0.3, 0.6, 1.0)  # chaser's pace advantage (s/lap): bin edges, 6 bins
PASS_DEFAULT = (0.02, 0.05, 0.12, 0.25, 0.40, 0.60)  # per-lap pass chance within 1 s, by bin (no history)
SC_DEFAULT_RATE = 0.0075  # safety-car starts per lap, all circuits (no history)
VSC_DEFAULT_RATE = 0.0045
SC_DEFAULT_LAPS = 4.0
VSC_DEFAULT_LAPS = 2.0
SHRINK_CIRCUIT_LAPS = 120.0  # pseudo-laps pulling a circuit's SC rate toward the pool
SHRINK_PASS = 40.0  # pseudo-exposures pulling a circuit's pass chance toward the pool


def pass_bin(delta: float) -> int:
    return bisect.bisect_right(PASS_BINS, delta)


def learn(final, meta: dict) -> dict:
    """Stints by compound, safety-car events, and pass chances by pace advantage, from a finished race."""
    stops: dict[str, list[int]] = {}
    for pe in final.pit_events:
        if not pe.under_red:
            stops.setdefault(pe.driver, []).append(pe.in_lap)
    laps: dict[str, dict[int, object]] = {}
    for x in final.laps:
        laps.setdefault(x.driver, {})[x.lap] = x
    total = final.total_laps or 0

    stints: dict[str, list[list[int]]] = {}
    for drv, by_lap in laps.items():
        start = 0
        for s in sorted(set(stops.get(drv, ()))):
            rec = by_lap.get(s)
            if rec is not None and rec.compound in DRY and s - start >= 1:
                stints.setdefault(rec.compound, []).append([s - start, 1])
            start = s
        d = final.drivers.get(drv)
        last = max(by_lap, default=0)
        if d is not None and d.running and last > start and by_lap.get(last) is not None:
            comp = by_lap[last].compound
            if comp in DRY:
                stints.setdefault(comp, []).append([last - start, 0])

    # safety cars: start lap and length in leader laps
    leader = sorted((x for x in final.laps if x.position == 1), key=lambda x: x.lap)
    lead_t = [x.t_end for x in leader]
    events: dict[str, list] = {"sc": [], "vsc": []}
    cur: tuple[str, float] | None = None
    for t, code in list(final.status_log) + [(float("inf"), "1")]:
        kind = "sc" if code == "4" else "vsc" if code == "6" else None
        if cur is not None and kind != cur[0]:
            k0, t0 = cur
            i0 = bisect.bisect_left(lead_t, t0)
            i1 = bisect.bisect_left(lead_t, t) if math.isfinite(t) else len(lead_t)
            events[k0].append([i0 + 1, max(1, i1 - i0)])
            cur = None
        if kind is not None and cur is None:
            cur = (kind, t)

    # passing: adjacent cars within 1 s at a lap end, both on green laps without a stop around them
    exp = [0] * (len(PASS_BINS) + 1)
    swp = [0] * (len(PASS_BINS) + 1)
    by_lap_pos: dict[int, list] = {}
    for x in final.laps:
        if x.position is not None:
            by_lap_pos.setdefault(x.lap, []).append(x)
    for lap_no, recs in by_lap_pos.items():
        nxt = {x.driver: x for x in by_lap_pos.get(lap_no + 1, [])}
        recs.sort(key=lambda x: x.position)
        for a, b in zip(recs, recs[1:]):
            if b.position != a.position + 1 or b.interval is None or b.interval >= 1.0 or lap_no < 2:
                continue
            a1, b1 = nxt.get(a.driver), nxt.get(b.driver)
            if a1 is None or b1 is None or a1.lap_time is None or b1.lap_time is None:
                continue
            if any(r.track_status != "1" or r.is_in_lap or r.is_out_lap for r in (a, b, a1, b1)):
                continue
            k = pass_bin(a1.lap_time - b1.lap_time)
            exp[k] += 1
            swp[k] += int(b1.position is not None and a1.position is not None and b1.position < a1.position)
    return {"stints": stints, "sc": events["sc"], "vsc": events["vsc"], "laps": total,
            "pass": [exp, swp], "circuit": meta.get("circuit_key")}


class Priors:
    """Tables from the races that ended before this one started (circuit-specific, shrunk to all races)."""

    def __init__(self, past_races, circuit) -> None:
        self.circuit = circuit
        pool: dict[str, list[int]] = {c: [] for c in DRY}
        pool_open: dict[str, list[int]] = {c: [] for c in DRY}
        circ: dict[str, list[int]] = {c: [] for c in DRY}
        circ_open: dict[str, list[int]] = {c: [] for c in DRY}
        self.n_races = 0
        pool_laps = circ_laps = 0
        pool_sc = pool_vsc = circ_sc = circ_vsc = 0
        sc_len: list[int] = []
        vsc_len: list[int] = []
        pexp, pswp, cexp, cswp = [0] * 6, [0] * 6, [0] * 6, [0] * 6
        self.laps_seen: list[int] = []
        for race in past_races:
            ex = (race.extra or {}).get("strategy")
            if not ex:
                continue
            self.n_races += 1
            same = race.circuit_key == circuit
            for comp, rows in (ex.get("stints") or {}).items():
                if comp not in pool:
                    continue
                for length, ended in rows:
                    (pool if ended else pool_open)[comp].append(length)
                    if same:
                        (circ if ended else circ_open)[comp].append(length)
            n = race.laps or ex.get("laps") or 0
            if n:
                self.laps_seen.append(n)
            pool_laps += n
            pool_sc += len(ex.get("sc") or [])
            pool_vsc += len(ex.get("vsc") or [])
            sc_len += [d for _, d in (ex.get("sc") or [])]
            vsc_len += [d for _, d in (ex.get("vsc") or [])]
            if same:
                circ_laps += n
                circ_sc += len(ex.get("sc") or [])
                circ_vsc += len(ex.get("vsc") or [])
            e, s = ex.get("pass") or ([0] * 6, [0] * 6)
            for i in range(6):
                pexp[i] += e[i]
                pswp[i] += s[i]
                if same:
                    cexp[i] += e[i]
                    cswp[i] += s[i]
        use_circ = {c: len(circ[c]) >= 6 for c in DRY}
        self.stints = {c: sorted(circ[c] if use_circ[c] else pool[c]) for c in DRY}
        self.stints_open = {c: sorted(circ_open[c] if use_circ[c] else pool_open[c]) for c in DRY}
        self.life = {}
        for c in DRY:
            s = self.stints[c]
            self.life[c] = float(s[len(s) // 2]) if len(s) >= 5 else LIFE[c]
        # safety-car start rates per lap: circuit shrunk to pool shrunk to the default
        pr_sc = (pool_sc + 5 * SC_DEFAULT_RATE * 60) / (pool_laps + 5 * 60)
        pr_vsc = (pool_vsc + 5 * VSC_DEFAULT_RATE * 60) / (pool_laps + 5 * 60)
        self.sc_rate = (circ_sc + SHRINK_CIRCUIT_LAPS * pr_sc) / (circ_laps + SHRINK_CIRCUIT_LAPS)
        self.vsc_rate = (circ_vsc + SHRINK_CIRCUIT_LAPS * pr_vsc) / (circ_laps + SHRINK_CIRCUIT_LAPS)
        self.sc_len = (sum(sc_len) + 3 * SC_DEFAULT_LAPS) / (len(sc_len) + 3)
        self.vsc_len = (sum(vsc_len) + 3 * VSC_DEFAULT_LAPS) / (len(vsc_len) + 3)
        self.pass_p: list[float] = []
        for i in range(6):
            pool_p = (pswp[i] + 20 * PASS_DEFAULT[i]) / (pexp[i] + 20)
            self.pass_p.append((cswp[i] + SHRINK_PASS * pool_p) / (cexp[i] + SHRINK_PASS))
        for i in range(1, 6):  # a bigger pace advantage never passes less
            self.pass_p[i] = max(self.pass_p[i], self.pass_p[i - 1])
