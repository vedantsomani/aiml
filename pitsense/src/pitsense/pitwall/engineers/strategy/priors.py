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
RED_DEFAULT_RATE = 0.0008  # red flags per lap (no history)
SHRINK_RED_LAPS = 600.0
PHASE_EDGES = (0.25, 0.5, 0.75)  # lap phase bins (share of the distance) for the neutralisation hazard; lap 1 is excluded
SHRINK_PHASE_LAPS = 400.0  # pseudo-laps pulling a phase's hazard multiplier toward 1
PASS_OFF_SD = 0.6  # prior sd of a circuit's overtaking-difficulty shift (logit scale)
AIR_DEFAULT = 0.25  # s/lap lost running within 1 s of the car ahead (no history)
SHRINK_AIR = 30.0  # pseudo-weight pulling a circuit's dirty-air loss toward the pool
CLEAR_S = 2.0  # a lap started this far behind the car ahead is in clear air


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
    events["red"] = []
    cur: tuple[str, float] | None = None
    for t, code in list(final.status_log) + [(float("inf"), "1")]:
        kind = "sc" if code == "4" else "vsc" if code in ("6", "7") else "red" if code == "5" else None
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
    out = {"stints": stints, "sc": events["sc"], "vsc": events["vsc"], "laps": total,
           "pass": [exp, swp], "circuit": meta.get("circuit_key"), "v": 2, "red": events["red"],
           "pass2": _learn_pass_free(by_lap_pos, laps), "air": _learn_air(laps)}
    from .wetmodel import dry_reference, learn_wet

    ref = dry_reference(final)
    if ref is not None:  # circuit's dry pace for the wet model (added key; dry-race planning does not read it)
        out["ref10"] = ref

    wet = learn_wet(final, meta)
    if wet:  # only races that used inters or wets (dry races' summaries are unchanged)
        out["wet"] = wet
    return out


def _green(r) -> bool:
    return r is not None and r.lap_time is not None and r.track_status == "1" and not r.is_in_lap and not r.is_out_lap


def _free_pace(by_lap: dict, lap: int, min_gap: float) -> float | None:
    """Median lap time of the car's recent green laps (up to 8 before ``lap``, the 5 latest used) that started at
    least ``min_gap`` behind the car ahead (0: any)."""
    xs = []
    for k in range(lap - 1, max(1, lap - 9), -1):
        r, prev = by_lap.get(k), by_lap.get(k - 1)
        if not _green(r) or prev is None:
            continue
        if min_gap <= 0 or prev.position == 1 or (prev.interval is not None and prev.interval >= min_gap):
            xs.append(r.lap_time)
        if len(xs) == 5:
            break
    if len(xs) < 2:
        return None
    xs.sort()
    return xs[len(xs) // 2]


def _learn_pass_free(by_lap_pos: dict, laps: dict) -> list:
    """Passes by the chaser's free-air pace advantage (both cars' recent clear-air laps), adjacent cars within 1 s."""
    exp = [0] * (len(PASS_BINS) + 1)
    swp = [0] * (len(PASS_BINS) + 1)
    for lap_no, recs in by_lap_pos.items():
        if lap_no < 3:
            continue
        nxt = {x.driver: x for x in by_lap_pos.get(lap_no + 1, [])}
        recs = sorted(recs, key=lambda x: x.position)
        for a, b in zip(recs, recs[1:]):
            if b.position != a.position + 1 or b.interval is None or b.interval >= 1.0:
                continue
            a1, b1 = nxt.get(a.driver), nxt.get(b.driver)
            if a1 is None or b1 is None or a1.position is None or b1.position is None:
                continue
            if any(r.track_status != "1" or r.is_in_lap or r.is_out_lap for r in (a, b, a1, b1)):
                continue
            pa, pb = _free_pace(laps[a.driver], lap_no + 1, 0.0), _free_pace(laps[b.driver], lap_no + 1, 1.0)
            if pa is None or pb is None:
                continue
            k = pass_bin(pa - pb)
            exp[k] += 1
            swp[k] += int(b1.position < a1.position)
    return [exp, swp]


def _learn_air(laps: dict) -> list:
    """Lap time lost running within 1 s of the car ahead: per stint, lap time ~ 1 + lap + close (least squares),
    pooled with inverse-variance-like weights. Returns [sum w*coef, sum w]."""
    import numpy as np

    num = den = 0.0
    for by_lap in laps.values():
        stints: dict[int, list] = {}
        for k, r in by_lap.items():
            prev = by_lap.get(k - 1)
            if k < 3 or not _green(r) or prev is None or prev.position is None:
                continue
            close = prev.position > 1 and prev.interval is not None and prev.interval < 1.0
            clear = prev.position == 1 or (prev.interval is not None and prev.interval >= CLEAR_S)
            if close or clear:
                stints.setdefault(r.stint, []).append((k, r.lap_time, 1.0 if close else 0.0))
        for rows in stints.values():
            if len(rows) < 8:
                continue
            x = np.array(rows)
            n1 = x[:, 2].sum()
            n0 = len(x) - n1
            if n1 < 2 or n0 < 2:
                continue
            med = np.median(x[:, 1])
            ok = np.abs(x[:, 1] - med) < 3.0  # traffic with lapped cars, mistakes
            x = x[ok]
            if len(x) < 8 or x[:, 2].sum() < 2 or (1 - x[:, 2]).sum() < 2:
                continue
            X = np.column_stack([np.ones(len(x)), x[:, 0] - x[:, 0].mean(), x[:, 2]])
            coef = np.linalg.lstsq(X, x[:, 1], rcond=None)[0][2]
            w = 1.0 / (1.0 / x[:, 2].sum() + 1.0 / (1 - x[:, 2]).sum())
            num += w * float(np.clip(coef, -1.0, 2.0))
            den += w
    return [round(float(num), 4), round(float(den), 4)]


def _phase_bin(lap: int, n: int) -> int:
    return bisect.bisect_left(PHASE_EDGES, lap / max(n, 1))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _circuit_shift(base: list[float], exp: list[int], swp: list[int]) -> float:
    """Logit shift of the pool's pass chances that best fits a circuit's passes (normal prior, Newton steps)."""
    o = 0.0
    z = [math.log(max(p, 1e-4) / max(1 - p, 1e-4)) for p in base]
    for _ in range(12):
        g, h = -o / PASS_OFF_SD ** 2, -1.0 / PASS_OFF_SD ** 2
        for zi, e, s in zip(z, exp, swp):
            if e:
                p = _sigmoid(zi + o)
                g += s - e * p
                h -= e * p * (1 - p)
        step = g / h
        o -= step
        if abs(step) < 1e-6:
            break
    return max(-2.5, min(2.5, o))


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
        nb = len(PHASE_EDGES) + 1
        hz_ev = {k: [0] * nb for k in ("sc", "vsc", "red")}  # pool event starts by phase (laps >= 2)
        hz_exp = [0] * nb
        hz_cev = {k: 0 for k in ("sc", "vsc", "red")}
        hz_cexp = 0
        red_exp = red_cexp = 0
        q_exp, q_swp, q_cexp, q_cswp = [0] * 6, [0] * 6, [0] * 6, [0] * 6
        air_n = air_d = air_cn = air_cd = 0.0
        loss = {k: [] for k in ("green", "sc", "vsc")}
        closs = {k: [] for k in ("green", "sc", "vsc")}
        for race in past_races:
            ex = (race.extra or {}).get("strategy")
            if not ex:
                continue
            self.n_races += 1
            same = race.circuit_key == circuit
            v2 = (ex.get("v") or 0) >= 2
            n_laps = race.laps or ex.get("laps") or 0
            if n_laps:
                kinds = ("sc", "vsc", "red") if v2 else ("sc", "vsc")
                covered = set()
                for k in kinds:
                    for st, ln in ex.get(k) or []:
                        covered.update(range(st + 1, st + max(1, ln)))
                        if 2 <= st <= n_laps:
                            hz_ev[k][_phase_bin(st, n_laps)] += 1
                            hz_cev[k] += int(same)
                green_laps = [l for l in range(2, n_laps + 1) if l not in covered]
                for l in green_laps:
                    hz_exp[_phase_bin(l, n_laps)] += 1
                hz_cexp += len(green_laps) if same else 0
                if v2:
                    red_exp += len(green_laps)
                    red_cexp += len(green_laps) if same else 0
            if v2:
                e2, s2 = ex.get("pass2") or ([0] * 6, [0] * 6)
                for i in range(6):
                    q_exp[i] += e2[i]
                    q_swp[i] += s2[i]
                    if same:
                        q_cexp[i] += e2[i]
                        q_cswp[i] += s2[i]
                an, ad = ex.get("air") or (0.0, 0.0)
                air_n, air_d = air_n + an, air_d + ad
                if same:
                    air_cn, air_cd = air_cn + an, air_cd + ad
            for k in loss:
                vals = [float(x) for x in (getattr(race, f"pit_loss_{k}", None) or []) if isinstance(x, (int, float))]
                loss[k] += vals
                if same:
                    closs[k] += vals
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
        self.pass_p_bin = list(self.pass_p)  # the older table (pace advantage from the lap itself)
        # passing by free-air pace advantage: the pool's curve, shifted on the logit scale by the circuit's difficulty
        self.pass_shift = 0.0
        if sum(q_exp) >= 300:
            base = [(q_swp[i] + 20 * PASS_DEFAULT[i]) / (q_exp[i] + 20) for i in range(6)]
            for i in range(1, 6):
                base[i] = max(base[i], base[i - 1])
            self.pass_shift = _circuit_shift(base, q_cexp, q_cswp)
            z = [math.log(max(p, 1e-4) / max(1 - p, 1e-4)) for p in base]
            self.pass_p = [_sigmoid(zi + self.pass_shift) for zi in z]
        # neutralisation hazard per green lap (laps 2+): circuit level, times a pooled lap-phase multiplier
        pool_exp = max(sum(hz_exp), 1)
        self.hz_base, self.hz_mult = {}, {}
        for k, dflt in (("sc", SC_DEFAULT_RATE), ("vsc", VSC_DEFAULT_RATE), ("red", RED_DEFAULT_RATE)):
            ev = sum(hz_ev[k])
            ex_k = red_exp if k == "red" else pool_exp
            pool_k = (ev + 300 * dflt) / (ex_k + 300)
            shrink = SHRINK_RED_LAPS if k == "red" else SHRINK_CIRCUIT_LAPS
            cex = red_cexp if k == "red" else hz_cexp
            self.hz_base[k] = (hz_cev[k] + shrink * pool_k) / (cex + shrink)
            raw_k = ev / max(ex_k, 1)
            if k == "red" or ev < 10:
                self.hz_mult[k] = [1.0] * nb
            else:
                self.hz_mult[k] = [((hz_ev[k][b] + SHRINK_PHASE_LAPS * raw_k) / (hz_exp[b] + SHRINK_PHASE_LAPS)) / raw_k
                                   for b in range(nb)]
        self.sc_lens = sorted(int(x) for x in sc_len if x >= 1)
        self.vsc_lens = sorted(int(x) for x in vsc_len if x >= 1)
        self.air = (air_cn + SHRINK_AIR * ((air_n + 10 * AIR_DEFAULT) / (air_d + 10))) / (air_cd + SHRINK_AIR)
        # pit loss by circuit and track status: circuit mean shrunk to the pool median (SC / VSC as a ratio to green)
        self.loss = {}
        med = lambda v, d: sorted(v)[len(v) // 2] if v else d  # noqa: E731
        g_pool = med(loss["green"], 22.0)
        cg = closs["green"]
        self.loss["green"] = (sum(cg) + 6 * g_pool) / (len(cg) + 6)
        for k, d in (("sc", 0.55), ("vsc", 0.7)):
            r_pool = med(loss[k], d * g_pool) / max(g_pool, 1.0)
            ck = closs[k]
            r_c = (sum(x / max(self.loss["green"], 1.0) for x in ck) + 6 * r_pool) / (len(ck) + 6)
            self.loss[k] = r_c * self.loss["green"]

    def hazard(self, A: int, total: int):
        """Per-lap start probability of a red flag, SC and VSC for laps A+1..total: array [3, R]."""
        import numpy as np

        R = max(total - A, 0)
        out = np.zeros((3, R))
        for i, k in enumerate(("red", "sc", "vsc")):
            mult = self.hz_mult[k]
            out[i] = [self.hz_base[k] * mult[_phase_bin(A + 1 + j, total)] for j in range(R)]
        return out

    def fill(self, F) -> None:
        """Put the tables the simulator samples from on a FieldIn (after its laps and losses are set)."""
        F.sc_rate, F.vsc_rate, F.sc_len, F.vsc_len = self.sc_rate, self.vsc_rate, self.sc_len, self.vsc_len
        F.pass_p = tuple(self.pass_p)
        F.haz = self.hazard(F.A, F.total)
        F.sc_lens = tuple(self.sc_lens)
        F.vsc_lens = tuple(self.vsc_lens)
        F.air = float(self.air)
        for k, v in self.loss.items():
            if F.loss.get(k) is None:
                F.loss[k] = v
