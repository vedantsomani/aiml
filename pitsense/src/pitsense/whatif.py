"""Ask the pit wall: parse a question, and answer "what if" questions on the strategy simulator.

Two halves, both as-of (everything comes from the race state at ``state.t``; nothing from the future):

* :func:`parse_question` turns text (typed, or a Whisper transcript) into a :class:`Question`: a what-if
  (box now / in N laps / for a compound, stay out to the end, a safety car in N laps, the effect against a
  rival car) or a fact question (gap, tyre age, pit window, plan B, ...). Regexes decide; whatever they do
  not catch goes to the voice's free-form router (``voice.composer.free_kind``).
* :func:`what_if` runs the strategy engineer's simulator once (the same 288 futures it uses for its own
  calls: ``analysis.build_field``, ``Draws``, ``simulate_field``) and scores the asked plans on those futures
  with ``evaluate_plans``, so the numbers are paired: the same futures for every plan, deterministic.

Public strategy functions only; nothing in ``strategy/`` or ``head.py`` is touched.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .pitwall.engineers.strategy import analysis
from .pitwall.engineers.strategy.engineer import get_analysis
from .pitwall.engineers.strategy.priors import DRY
from .pitwall.engineers.strategy.run import evaluate_plans, simulate_field
from .pitwall.engineers.strategy.sim import K_STOPS, NOSTOP, Draws

N_SIMS = 288
GAPS2 = (8, 12, 16, 21, 27)  # second-stop gaps tried when one stop cannot cover the stint
CIDX = {c: i for i, c in enumerate(DRY)}


# ============================================================================ parsing
@dataclass(frozen=True)
class Spec:
    """One strategy option in a question."""

    kind: str  # "now" | "in" | "lap" | "stay" | "plan" (keep the current plan: wait)
    laps: int | None = None  # "in": laps from now (0 = this lap)
    lap: int | None = None  # "lap": absolute in-lap
    compound: str | None = None  # SOFT | MEDIUM | HARD

    def label(self) -> str:
        c = f" for {self.compound}" if self.compound else ""
        if self.kind == "stay":
            return "Stay out to the end"
        if self.kind == "plan":
            return "Keep the current plan"
        if self.kind == "now":
            return f"Box now{c}"
        if self.kind == "lap":
            return f"Box on lap {self.lap}{c}"
        return f"Box in {self.laps} lap{'s' if self.laps != 1 else ''}{c}"


@dataclass(frozen=True)
class Question:
    text: str
    kind: str  # "whatif" | "sc" | "fact"
    options: tuple[Spec, ...] = ()
    sc_in: int = 0  # "sc": laps until the neutralisation starts (0 = this lap)
    vsc: bool = False
    rival: str | None = None  # car number
    rival_ref: str | None = None  # "ahead" | "behind" (resolved from the tower)
    fact: str | None = None  # gap | tyre_age | pit_window | plan_b | why | sc_prob | free
    target: str | None = None  # gap: the other car

    def to_dict(self) -> dict:
        return {"kind": self.kind, "fact": self.fact, "options": [o.label() for o in self.options],
                "sc_in": self.sc_in if self.kind == "sc" else None, "rival": self.rival, "rival_ref": self.rival_ref}


_WORDNUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_COMP = {"soft": "SOFT", "softs": "SOFT", "medium": "MEDIUM", "mediums": "MEDIUM", "mediam": "MEDIUM", "hard": "HARD", "hards": "HARD"}
_COLOUR = {"red": "SOFT", "reds": "SOFT", "yellow": "MEDIUM", "yellows": "MEDIUM", "white": "HARD", "whites": "HARD"}
_STOP_WORD = r"(?:box|boxing|boxed|pit|pits|pitting|pitstop|stop|stopping|come in|comes in|bring (?:him|her|us|it|the car) in|dive in|undercut|overcut|switch to|go onto|go on to|fit|change (?:the )?tyres?)"
_STAY = (r"stay(?:ing)? out|stay on (?:the )?(?:track|these|this|current|old|same)|no (?:more )?(?:pit ?)?stops?|never (?:box|pit)|"
         r"(?:don'?t|do not|not|without|skip|avoid) (?:a |the |another )?(?:box|boxing|pit|pitting|stop|stopping)|"
         r"(?:to|until|till) the (?:end|flag|finish|chequered|checkered)|run (?:it )?(?:to|until) the end|go (?:all the way )?to the end|extend (?:the )?(?:stint|this)|stretch")
_NOW = r"\b(?:now|right now|this lap|immediately|straight ?away|at once)\b"
_NEXT = r"\bnext lap\b|\bone lap\b|\b1 lap\b"
_SC = r"\b(?:safety car|safety|sc|s c|vsc|v s c|virtual safety car|virtual|neutrali[sz]ation|full course|caution)\b"
_SPLIT = r"\s(?:or|vs\.?|versus|instead of|rather than|compared (?:to|with)|as opposed to|against staying|and then)\s"
_RIVAL_CUE = r"(?:to|about|against|vs\.?|versus|relative to|ahead of|behind|beat|overtake|pass|cover|covering|keep ahead of|stay ahead of|hold off|defend (?:from|against)|affect|impact on|effect on|catch|undercut|overcut|than|from|on|for|with|of)"
_SURN_SKIP = {"what", "when", "where", "should", "would", "could", "this", "that", "they", "them", "then", "tyres", "tires", "laps", "lap", "pits", "pit", "stop", "stay", "soft", "hard", "medium", "safety", "next", "stint", "gap", "wall", "plan", "race", "position", "ahead", "behind", "leader", "chasing", "overtake", "pass", "cover", "against", "versus"}


def _norm(text: str) -> str:
    q = text.lower().replace("’", "'")
    q = re.sub(r"[^a-z0-9#'. ]+", " ", q)
    q = re.sub(r"\b(" + "|".join(sorted(_WORDNUM, key=len, reverse=True)) + r")\b", lambda m: str(_WORDNUM[m.group(1)]), q)
    q = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _compound(q: str) -> str | None:
    m = re.search(r"\b(soft|softs|medium|mediums|mediam|hard|hards)\b", q)
    if m:
        return _COMP[m.group(1)]
    m = re.search(r"\b(reds?|yellows?|whites?)\s+(?:tyres?|tires?|compound|set|ones?)\b", q)
    return _COLOUR[m.group(1)] if m else None


def _rival(raw: str, q: str, tla_of: dict[str, str], me: str | None) -> tuple[str | None, str | None]:
    """(car number, "ahead"/"behind") the question is about, if any. Never the asked car itself."""
    tla2num = {t.upper(): n for n, t in tla_of.items() if t}
    for m in re.finditer(r"\b([A-Z]{3})\b", raw):  # typed in capitals: VER
        n = tla2num.get(m.group(1))
        if n and n != me:
            return n, None
    m = re.search(r"\b(?:car|number|no|#)\s*#?(\d{1,2})\b", q)
    if m and m.group(1) in tla_of and m.group(1) != me:
        return m.group(1), None
    for m in re.finditer(r"\b([a-z]{3})\b", q):  # lowercase TLA after a cue word: "against ver"
        n = tla2num.get(m.group(1).upper())
        if n and n != me and re.search(_RIVAL_CUE + r"\s+(?:the\s+)?" + m.group(1) + r"\b", q):
            return n, None
    for m in re.finditer(r"\b([a-z]{4,})\b", q):  # surname: Verstappen -> VER, Leclerc -> LEC
        w = m.group(1)
        if w in _SURN_SKIP:
            continue
        n = tla2num.get(w[:3].upper())
        if n and n != me and re.search(_RIVAL_CUE + r"\s+(?:the\s+)?(?:\w+\s+)?" + w + r"\b", q):
            return n, None
    if re.search(r"\b(?:car|driver|guy|one|rival)\s+(?:in front|ahead)|\b(?:ahead of|beat|pass|overtake|catch|undercut)\s+the car\b|the car (?:in front|ahead)", q):
        return None, "ahead"
    if re.search(r"\b(?:car|driver|guy|one|rival)\s+behind|the car behind|\bchasing\b|\bhold off\b|\bdefend\b|\bcover\b", q):
        return None, "behind"
    return None, None


def _spec(part: str) -> Spec | None:
    """One option from a clause. None if the clause names no plan."""
    comp = _compound(part)
    if re.search(_STAY, part):
        return Spec("stay")
    n = re.search(r"\b(?:in|after|within)\s+(\d{1,2})\s+laps?\b", part)
    ab = re.search(r"\b(?:on|at|by|around|end of)?\s*lap\s+(\d{1,2})\b", part)
    if n:
        k = int(n.group(1))
        return Spec("now" if k == 0 else "in", laps=k, compound=comp)
    if re.search(_NEXT, part):
        return Spec("in", laps=1, compound=comp)
    if ab and re.search(_STOP_WORD, part):
        return Spec("lap", lap=int(ab.group(1)), compound=comp)
    if re.search(r"\b(?:wait|later|delay|hold off|not yet|extend|stretch|longer|keep going|as planned|stick to)\b", part):
        return Spec("plan", compound=comp)
    if re.search(_NOW, part) or re.search(_STOP_WORD, part) or comp:
        return Spec("now", laps=0, compound=comp)
    return None


def parse_question(text: str, tla_of: dict[str, str] | None = None, car: str | None = None) -> Question:
    """Map a question to a what-if or a fact. ``tla_of``: car number -> TLA of the running field."""
    tla_of = tla_of or {}
    raw = text or ""
    q = _norm(raw)
    rival, ref = _rival(raw, q, tla_of, car)
    box = bool(re.search(r"\b" + _STOP_WORD + r"\b", q))
    iff = bool(re.search(r"\b(?:what if|if|suppose|imagine|should|shall|can we|could we|would|do we|do you want|how about|what about|compare)\b", q))
    sc = re.search(_SC, q) and not re.search(r"\bsc\b.*\bnumber\b", q)

    if sc and re.search(r"\b(?:chance|probab\w*|likelihood|odds|how likely|risk)\b", q):
        return Question(raw, "fact", fact="sc_prob", rival=rival)
    if sc:
        k = 0
        m = re.search(r"\b(?:in|after|within)\s+(\d{1,2})\s+laps?\b", q)
        if m:
            k = int(m.group(1))
        elif re.search(_NEXT, q):
            k = 1
        spec = _spec(re.sub(_SC, " ", q)) if box and not re.search(r"\b(?:what if|if)\s+(?:a |the )?(?:" + _SC + ")", q) else None
        return Question(raw, "sc", options=(spec,) if spec and box else (), sc_in=k,
                        vsc=bool(re.search(r"\bvsc\b|v s c|virtual", q)), rival=rival, rival_ref=ref)
    if re.search(r"^(?:why|what is the reason|what are the reasons|explain)\b", q):
        return Question(raw, "fact", fact="why")
    if re.search(_STAY, q) and not re.search(r"\bwhen\b", q):
        parts = re.split(_SPLIT, q, maxsplit=1)
        opts = tuple(s for s in (_spec(p) for p in parts) if s)
        return Question(raw, "whatif", options=opts[:2] or (Spec("stay"),), rival=rival, rival_ref=ref)

    # facts that mention boxing but are not what-ifs
    if re.search(r"\bplan b\b|back ?up|fall ?back|alternative|second plan|other plan", q):
        return Question(raw, "fact", fact="plan_b")
    if re.search(r"\b(?:rejoin|come out|come back out|where (?:would|will|do|does|are)\b.*\b(?:we|he|she|i|they)?\s*(?:come|end|exit))|\bposition after\b", q):
        return Question(raw, "fact", fact="free")
    if re.search(r"pit loss|stop cost|cost of (?:a |the )?(?:stop|pit)|how much (?:time )?.*(?:lose|cost).*(?:stop|pit|box)|time (?:lost|loss) .*(?:pit|stop)", q):
        return Question(raw, "fact", fact="free")
    if re.search(r"\b(?:tyres?|tires?|stint|set|rubber)\b", q) and re.search(r"\b(?:age|old|how many laps|how long|life|worn|wear|deg|degradation|cliff|condition|state)\b", q) and not re.search(r"\bif\b|what about", q):
        return Question(raw, "fact", fact="tyre_age" if re.search(r"\b(?:age|old|how many laps|how long)\b", q) else "free")
    if re.search(r"\bwhy\b|\breasons?\b|\bbecause\b|\bexplain\b", q):
        return Question(raw, "fact", fact="why")
    if re.search(r"\bwhen\b.*" + _STOP_WORD + r"|\bwhich lap\b|\bwhat lap\b|pit window|next (?:pit )?stop|\bplan a\b|\bthe plan\b|\bour plan\b|\bstrategy\b|\bwindow\b", q) and not re.search(r"\bwhat if\b", q):
        return Question(raw, "fact", fact="pit_window")
    if re.search(r"\b(?:gap|interval|how far|how close|distance|margin)\b", q) and not re.search(r"\bif\b|what about", q):
        return Question(raw, "fact", fact="gap", rival=rival, rival_ref=ref, target=rival)
    if re.search(r"^(?:who|which car)\b.*\b(?:ahead|behind|in front|chasing)\b|\bwho is (?:ahead|behind|in front|chasing)\b", q):
        return Question(raw, "fact", fact="gap", rival=rival, rival_ref="behind" if re.search(r"behind|chasing", q) else "ahead")

    # what-ifs
    timing = bool(re.search(_NOW + r"|" + _NEXT + r"|\b(?:in|after|within)\s+\d{1,2}\s+laps?\b|\blap\s+\d{1,2}\b|\bwait\b|\blater\b", q))
    comp = _compound(q)
    if box or comp or (rival and (iff or timing)):
        parts = [p for p in re.split(_SPLIT, q, maxsplit=1)]
        opts: list[Spec] = []
        if len(parts) == 2:
            a, b = (_spec(p) for p in parts)
            if a and b:
                # "box now for hards or mediums": the second clause only names a compound -> one option
                if b.kind == "now" and not re.search(_NOW + r"|" + _STOP_WORD, parts[1]) and a.kind in ("now", "in", "lap"):
                    opts = [a]
                else:
                    if b.compound is None and b.kind in ("now", "in", "lap") and a.compound:
                        b = Spec(b.kind, b.laps, b.lap, None)
                    opts = [a, b]
            elif a or b:
                opts = [a or b]
        else:
            s = _spec(q)
            if s:
                opts = [s]
        if opts:
            return Question(raw, "whatif", options=tuple(opts), rival=rival, rival_ref=ref)
        if rival or ref:
            return Question(raw, "whatif", options=(Spec("now", laps=0),), rival=rival, rival_ref=ref)

    from .voice.composer import free_kind  # the voice's free-form router for everything else

    k = free_kind(q)
    if k in ("gap_ahead", "gap_behind"):
        return Question(raw, "fact", fact="gap", rival_ref="ahead" if k == "gap_ahead" else "behind")
    if k == "tyre":
        return Question(raw, "fact", fact="tyre_age")
    if k == "plan":
        return Question(raw, "fact", fact="pit_window")
    if k == "why":
        return Question(raw, "fact", fact="why")
    return Question(raw, "fact", fact="free")


# ============================================================================ the engine
def _legal(F, info, c: int, car: str, plan: tuple) -> bool:
    """The plan generator's own legality rules (two compounds, stint limits, tyre life), restated."""
    A, total, life = F.A, F.total, F.life
    used = {j for j in range(3) if F.used[c, j]}
    comps = {cj for _, cj in plan}
    if F.must[c] and len(used | comps) < 2:
        return False
    if info["stops_done"].get(car, 0) + len(plan) < F.reg_min_stops:
        return False
    stint_left = info["stint_left"].get(car)
    laps = [l for l, _ in plan]
    if plan and stint_left is not None and laps[0] > A + max(stint_left, 0):
        return False
    if not plan and stint_left is not None and stint_left < F.R:
        return False
    bounds = laps + [total]
    for k, (l, cj) in enumerate(plan):
        nxt = bounds[k + 1]
        if nxt - l > 1.35 * life[cj] or (nxt - l < 4 and nxt != total):
            return False
        if F.max_stint and nxt - l > F.max_stint:
            return False
    if plan and plan[-1][0] > total - 3:
        return False
    return all(b > a for a, b in zip(laps, laps[1:]))


def _variants(F, info, c: int, car: str, first_lap: int, comps) -> tuple[list[tuple], bool]:
    """Plans that stop at ``first_lap`` (one stop, or two when the stint is too long). (plans, all_legal)."""
    total = F.total
    plans, loose = [], []
    for cj in comps:
        one = ((first_lap, cj),)
        (plans if _legal(F, info, c, car, one) else loose).append(one)
        for g in GAPS2:
            l2 = first_lap + g
            if l2 > total - 3:
                continue
            for c2 in range(3):
                two = ((first_lap, cj), (l2, c2))
                if _legal(F, info, c, car, two):
                    plans.append(two)
    if plans:
        return plans, True
    return loose, False


def _arrays(plans: list[tuple], S: int):
    P = len(plans)
    stop = np.full((P, K_STOPS), NOSTOP, dtype=np.int64)
    comp = np.zeros((P, K_STOPS), dtype=np.int64)
    for i, p in enumerate(plans):
        for k, (l, cj) in enumerate(p[:K_STOPS]):
            stop[i, k], comp[i, k] = l, cj
    return np.repeat(stop[:, None, :], S, 1), np.repeat(comp[:, None, :], S, 1)


def _txt(plan: tuple) -> str:
    return ", ".join(f"L{l} {DRY[cj]}" for l, cj in plan) if plan else "no stop"


@dataclass
class _Field:
    F: object
    info: dict
    D: object
    run: object
    c: int
    names: list
    ms: float = 0.0
    extra: dict = field(default_factory=dict)


def _field_for(wall, state, car: str, S: int):
    """The simulated field and futures for ``car`` as of now, cached for this moment. (obj, None) or (None, why)."""
    ctx, memory = wall.ctx, wall.memory
    d = state.drivers.get(car)
    if d is None:
        return None, f"no car {car}"
    if not d.running:
        return None, f"car {car} is out of the race"
    if d.laps < 2:
        return None, "too early: no laps to read pace from"
    view = wall.view(state)
    if not bool(view.race("rules").get("race_dry", True)) or bool(view.race("weather").get("wet_running")):
        return None, "wet conditions: the simulator only models dry tyres"
    key = (id(state), state.t, d.laps, S)
    hit = ctx.__dict__.get("_whatif_field")
    if hit is not None and hit[0] == key:
        return hit[1], None
    t0 = time.perf_counter()
    A = d.laps
    pri = analysis.priors_for(ctx)
    F, names, info = analysis.build_field(state, view, memory, ctx, pri, A)
    if F.R <= 0 or F.C < 2:
        return None, "race is over"
    if car not in names:
        return None, "no timing for the car yet"
    D = Draws(F, S, analysis.seed_for(ctx, "", 0))
    run = simulate_field(F, D)
    fld = _Field(F, info, D, run, names.index(car), names, (time.perf_counter() - t0) * 1000)
    ctx.__dict__["_whatif_field"] = (key, fld)
    return fld, None


def _plan_a(wall, state, car: str, F) -> tuple | None:
    """The head's current plan A for the car (cached for focus cars), as ((lap, compound index), ...)."""
    view = wall.view(state)
    res = get_analysis(wall.ctx, wall.memory, state, view).get(car)
    if res is None:
        res = analysis.analyse(state, view, wall.ctx, wall.memory, [car], light=True).get(car)
    if res is None or not res.ok or res.plan_a is None:
        return None
    return tuple((int(l), CIDX[cn]) for l, cn in res.plan_a.stops)


def _stat(pos: np.ndarray) -> dict:
    return {"exp_pos": round(float(pos.mean()), 2), "sd": round(float(pos.std()), 2),
            "p10": round(float(np.percentile(pos, 10)), 1), "p90": round(float(np.percentile(pos, 90)), 1),
            "p_top10": round(float((pos <= 10).mean()), 3), "p_podium": round(float((pos <= 3).mean()), 3)}


def _fmt_num(x) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 3)


def what_if(wall, state, car: str, q: Question, *, S: int = N_SIMS) -> dict:
    """Answer a what-if question for ``car`` as of ``state``. Deterministic; about a second."""
    t0 = time.perf_counter()
    d = state.drivers.get(car)
    base = {"car": car, "tla": d.tla if d else car, "t": round(state.t, 1), "question": q.text, "parsed": q.to_dict(), "kind": q.kind}
    fld, why = _field_for(wall, state, car, S)
    if fld is None:
        return {**base, "ok": False, "why": why}
    F, info, D, run, c = fld.F, fld.info, fld.D, fld.run, fld.c
    A, total = F.A, F.total
    base.update(lap=A + 1, total=total, position=d.position, compound=d.compound, tyre_age=d.tyre_age)
    tla = {n: state.drivers[n].tla for n in F.cars}

    rc, rival = None, None  # the rival car, as an index into the field
    rn = q.rival
    if rn is None and q.rival_ref:
        order = [x for x in state.running_order() if x.running and x.number in F.cars]
        nums = [x.number for x in order]
        if car in nums:
            i = nums.index(car) + (-1 if q.rival_ref == "ahead" else 1)
            if 0 <= i < len(nums):
                rn = nums[i]
    if q.rival or q.rival_ref:
        if rn is None or rn not in F.cars or rn == car:
            return {**base, "ok": False, "why": f"no rival car to compare with ({'the car ' + q.rival_ref if q.rival_ref else q.rival})"}
        rc, rival = F.cars.index(rn), rn

    plan_a = _plan_a(wall, state, car, F)
    notes: list[str] = []
    legal_all = True
    use = D
    run_use = run
    sc_note = None

    # --- scenario futures: a safety car / VSC starting in N laps
    if q.kind == "sc":
        if F.sc_now:
            sc_note = "a neutralisation is already out; answered on the live futures"
            q = Question(q.text, "whatif", q.options or (Spec("now", laps=0),), rival=q.rival, rival_ref=q.rival_ref)
        else:
            s0 = min(max(q.sc_in, 0), max(F.R - 3, 0))
            use = Draws(F, S, analysis.seed_for(wall.ctx, "", 0), force_sc=(s0, s0, 0.0 if q.vsc else 1.0))
            run_use = simulate_field(F, use)

    # --- the options to score
    opts = list(q.options)
    if q.kind == "sc":
        l0 = A + 1 + min(max(q.sc_in, 0), max(F.R - 3, 0))
        spec = opts[0] if opts else Spec("lap", lap=l0)
        opts = [Spec("lap", lap=l0, compound=spec.compound)]
    elif not opts:
        opts = [Spec("now", laps=0)]
    groups: list[tuple[str, list[tuple]]] = []  # (label, variant plans)
    for sp in opts[:2]:
        if sp.kind == "stay":
            plans = [()]
            if not _legal(F, info, c, car, ()):
                legal_all = False
                notes.append("staying out is not allowed here: " + ("the car still has to fit a second compound" if F.must[c] else "the stint is too long or the minimum stops are not met"))
        elif sp.kind == "plan":
            plans = [plan_a] if plan_a is not None else []
        else:
            lap = (A + 1 + (sp.laps or 0)) if sp.kind in ("now", "in") else int(sp.lap)
            lap = max(lap, A + 1)
            if lap > total - 1:
                return {**base, "ok": False, "why": f"lap {lap} is past the end of the race"}
            comps = [CIDX[sp.compound]] if sp.compound else [0, 1, 2]
            plans, ok = _variants(F, info, c, car, lap, comps)
            if not ok:
                legal_all = False
                notes.append(f"a stop on lap {lap} falls outside the usual limits (too late, or a stint longer than the tyres last)")
        if not plans:
            return {**base, "ok": False, "why": "no plan to compare: the head has none yet"}
        lab = sp.label()
        if q.kind == "sc":
            lab = f"Box under the {'VSC' if q.vsc else 'safety car'} on lap {sp.lap}" + (f" for {sp.compound}" if sp.compound else "")
        groups.append((lab, plans))

    # --- one pass over all plans on the same futures
    allplans: list[tuple] = []
    index: dict[tuple, int] = {}

    def add(p: tuple) -> int:
        if p not in index:
            index[p] = len(allplans)
            allplans.append(p)
        return index[p]

    gidx = [[add(p) for p in plans] for _, plans in groups]
    a_i = add(plan_a) if plan_a is not None else None
    stop, comp = _arrays(allplans, S)
    pos, tm, soft = evaluate_plans(F, use, run_use, c, stop, comp, S)
    util = (soft + analysis.SETTINGS["w_time"] * (tm - tm.mean())).mean(1)

    def best(ix: list[int]) -> int:
        return min(ix, key=lambda i: (util[i], pos[i].mean()))

    def summarize(i: int, label: str) -> dict:
        out = {"label": label, "stops": [[int(l), DRY[cj]] for l, cj in allplans[i]], "plan": _txt(allplans[i]), **_stat(pos[i]),
               "race_time_s": round(float(tm[i].mean() - tm.mean()), 1)}
        if rc is not None:
            out["p_ahead_rival"] = round(float((tm[i] < run_use.final[:S, rc]).mean()), 3)
        return out

    chosen = [best(ix) for ix in gidx]
    res: dict = {**base, "ok": True, "scenario": summarize(chosen[0], groups[0][0]), "legal": legal_all, "notes": notes, "n_sims": S}
    vi = chosen[1] if len(chosen) == 2 else a_i
    versus = None
    if vi is not None:
        label = groups[1][0] if len(chosen) == 2 else ("Stay on plan A" if q.kind == "sc" else "Plan A") + f" ({_txt(allplans[vi])})"
        versus = summarize(vi, label)
    scen = res["scenario"]
    if q.kind == "sc":
        res["sc"] = {"in_laps": q.sc_in, "vsc": q.vsc, "stop_lap": opts[0].lap,
                     "pit_loss_sc_s": _fmt_num(F.loss["sc"]), "pit_loss_green_s": _fmt_num(F.loss["green"])}
        if plan_a is not None:  # the same plan A on the normal futures: what the safety car does to us
            stop0, comp0 = _arrays([plan_a], S)
            res["sc"]["plan_a_without_sc_pos"] = round(float(evaluate_plans(F, D, run, c, stop0, comp0, S)[0].mean()), 2)
    res["versus"] = versus
    if vi is not None:
        ps, pv = pos[chosen[0]], pos[vi]
        res["p_gain"] = round(float((ps < pv).mean()), 3)
        res["p_loss"] = round(float((ps > pv).mean()), 3)
        res["p_same"] = round(float((ps == pv).mean()), 3)
        res["delta_pos"] = round(float(pv.mean() - ps.mean()), 2)  # places gained (positive = the scenario is better)
        res["delta_time_s"] = round(float(tm[chosen[0]].mean() - tm[vi].mean()), 1)
    if rc is not None:
        res["rival"] = {"car": rival, "tla": tla.get(rival, rival), "p_stops_in_3": _fmt_num(F.pp[rc][1]),
                        "gap_now_s": _fmt_num(F.x0[rc] - F.x0[c])}
    if sc_note:
        notes.append(sc_note)

    # --- the reasons, most useful first
    reasons: list[str] = []
    if res.get("delta_pos") is not None and versus is not None:
        dp, dt = res["delta_pos"], res.get("delta_time_s")
        reasons.append(f"{abs(dp):.1f} places {'better' if dp > 0 else 'worse' if dp < 0 else 'no different'} than {_low(versus['label'].split(' (')[0])}"
                       + (f" and {abs(dt):.0f} s {'quicker' if dt < 0 else 'slower'} over the race" if dt is not None and abs(dt) >= 1 else ""))
    if q.kind == "sc":
        reasons.append(f"a stop under the {'VSC' if q.vsc else 'safety car'} costs about {F.loss['vsc' if q.vsc else 'sc']:.0f} s against {F.loss['green']:.0f} s under green")
    else:
        reasons.append(f"a stop costs about {F.loss['green'] + F.team_off[c]:.0f} s now")
    age = int(F.age0[c])
    reasons.append(f"{DRY[F.comp0[c]]} tyres are {age} laps old" + (f", cliff risk {int(round(100 * F.cliff_risk[c]))} percent" if F.cliff_risk[c] >= 0.1 else ""))
    if rc is not None:
        r = res["rival"]
        if scen.get("p_ahead_rival") is not None:
            reasons.append(f"{r['tla']} would be behind us in {int(round(100 * scen['p_ahead_rival']))} percent of futures"
                           + (f" against {int(round(100 * versus['p_ahead_rival']))} percent on the other option" if versus and versus.get("p_ahead_rival") is not None else ""))
        if r["p_stops_in_3"] is not None and r["p_stops_in_3"] >= 0.2:
            reasons.append(f"{r['tla']} is likely to stop within three laps ({int(round(100 * r['p_stops_in_3']))} percent)")
    if q.kind != "sc":
        reasons.append(f"a safety car in the next five laps is {int(round(100 * (1 - (1 - F.sc_rate) ** 5)))} percent likely")
    res["reasons"] = reasons[:5]
    res["ms"] = round((time.perf_counter() - t0) * 1000)
    res["sim_ms"] = round(fld.ms)
    res["answer"] = compose(res)
    return res


# ============================================================================ the words
def _low(t: str) -> str:
    return t[:1].lower() + t[1:]


def _pct(x) -> int:
    return int(round(100 * x))


def compose(r: dict) -> str:
    """The spoken answer: two or three short sentences, every number taken from the result."""
    if not r.get("ok"):
        return f"I cannot run that for {r.get('tla', 'the car')}: {r.get('why', 'no data')}."
    s, v, tla = r["scenario"], r.get("versus"), r["tla"]
    out: list[str] = []
    if "sc" in r:
        sc = r["sc"]
        kind = "VSC" if sc["vsc"] else "safety car"
        when = "this lap" if sc["in_laps"] == 0 else "next lap" if sc["in_laps"] == 1 else f"in {sc['in_laps']} laps"
        out.append(f"With a {kind} {when}, boxing under it ends position {s['exp_pos']:.1f} for {tla}"
                   + (f", against position {v['exp_pos']:.1f} staying on plan A." if v else "."))
        if r.get("p_gain") is not None:
            out.append(f"That gains places in {_pct(r['p_gain'])} percent of futures and loses in {_pct(r['p_loss'])} percent.")
        if "plan_a_without_sc_pos" in r["sc"]:
            out.append(f"Without it, plan A finishes position {r['sc']['plan_a_without_sc_pos']:.1f}.")
    else:
        lab = s["label"]
        if v is not None:
            out.append(f"{lab}: expected position {s['exp_pos']:.1f}, against {v['exp_pos']:.1f} for {_low(v['label'].split(' (')[0])}.")
            if r.get("p_gain") is not None:
                out.append(f"It gains places in {_pct(r['p_gain'])} percent of futures and loses in {_pct(r['p_loss'])} percent.")
        else:
            out.append(f"{lab}: expected position {s['exp_pos']:.1f}, podium chance {_pct(s['p_podium'])} percent.")
    if "rival" in r and s.get("p_ahead_rival") is not None:
        rv = r["rival"]["tla"]
        tail = f", against {_pct(v['p_ahead_rival'])} percent on the other option" if v and v.get("p_ahead_rival") is not None else ""
        out.append(f"{tla} finishes ahead of {rv} in {_pct(s['p_ahead_rival'])} percent of futures{tail}.")
    if r.get("notes"):
        out.append(r["notes"][0][0].upper() + r["notes"][0][1:] + ".")
    elif len(out) < 3 and r.get("reasons"):
        out.append(r["reasons"][1 if len(r["reasons"]) > 1 else 0][0].upper() + r["reasons"][1 if len(r["reasons"]) > 1 else 0][1:] + ".")
    return " ".join(out)
