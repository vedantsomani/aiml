"""Rival radio: what the other teams tell their drivers, as strategy intelligence.

Reads ``state.feeds.radio.messages(car)`` for every car (transcript known 15 s after the message, so
the answer at ``t`` never depends on later audio). Without the radio feed (benchmark rows built from the
timing archive alone) every key keeps its sentinel.

Each message is classified by ``classify(text)`` into intents with a phrase/regex model:

* ``box``        "box box", "box this lap", "we're going to box", "pit this lap"
* ``extend``     "stay out", "extend", "go long", "don't box" (a negated box is an extend)
* ``planchange`` "Plan B", "Strategy A", "box opposite", "one-stop"
* ``tyres_gone`` "tyres are gone", "no grip", "graining", "deg", "cliff"
* ``push`` / ``save``  "push", "attack" / "lift and coast", "fuel save", "manage the tyres"
* ``cover``      "cover", "undercut", "they've pitted"
* ``weather``    "rain", "drops", "inters"
* ``problem``    the mechanic phrase table (mechanic_radio.classify)

Negation ("don't box", "no need to push", "not gone") voids a hit or flips box into extend; a question
("should we box?", "do you want to ...", "why didn't we box?") counts 0.35; "if/in case" is hypothetical
(0.25). The newest box/extend message wins: "stay out" after "box box" cancels the box intent. An
intent is spent when the car pits after it.

Per-car keys (sentinels in brackets): ``rr_box_intent`` (0-1, half-life 120 s, 0 once the car has pitted),
``rr_extend``, ``rr_push``, ``rr_save``, ``rr_tyres_gone``, ``rr_planchange``, ``rr_cover``, ``rr_weather``,
``rr_problem`` (0), ``rr_n`` (clips known so far, 0), ``rr_age_s`` (seconds since the last classified clip, -1),
``rr_last_t`` (message time, -1), ``rr_last_quote`` (""), ``rr_last_intent`` ("").
Alerts are aimed at the focus team (``ctx.team``): "LEC's engineer: 'box box' (lap 23); he's 1.2 s ahead of you".
Docs: docs/engineers/rivalradio.md.
"""

from __future__ import annotations

import math
import re

from ..engineer import Engineer
from ..types import Alert
from .mechanic_radio import classify as classify_problem

HALF_LIFE_S = {"box": 120.0, "extend": 240.0, "push": 300.0, "save": 300.0, "tyres_gone": 600.0,
               "planchange": 600.0, "cover": 240.0, "weather": 600.0, "problem": 900.0}
MAX_AGE_S = 900.0
ALERT_BOX = 0.5
MIN_HIT = 0.25  # a clip counts as "classified" at this score

# (intent, weight, pattern); run per sentence on lower-case text
_P: list[tuple[str, float, re.Pattern]] = [(i, w, re.compile(p)) for i, w, p in [
    ("box", 0.9, r"\bbox,? box\b"),
    ("box", 0.9, r"\b(?:box|pit|pitting|boxing|come in|coming in|in) this lap\b"),
    ("box", 0.9, r"\b(?:box|pit) now\b"),
    ("box", 0.75, r"\b(?:we(?:'re| are| will|'ll)|let's|going to|gonna|we) (?:be )?(?:box|boxing|pit|pitting)\b"),
    ("box", 0.7, r"\b(?:box|pit|boxing|pitting) (?:next lap|at the end of (?:this|the) lap|in (?:one|two|three|\d) laps?|end of lap|on entry)\b"),
    ("box", 0.7, r"\b(?:going to|gonna|planning to|plan to|will) (?:box|pit)\b"),
    ("box", 0.35, r"\bbox to overtake\b"),
    ("box", 0.5, r"\bbox\b(?! to overtake)"),
    ("extend", 0.9, r"\bstay out\b(?! of)|\bstaying out\b|\bstay on (?:track|the track)\b"),
    ("extend", 0.8, r"\bextend\w*\b|\bgo(?:ing)? long\b"),
    ("extend", 0.7, r"\b(?:leave|leaving|left) (?:you |him )?out\b|\bkeep(?:ing)? you out\b|\bno stop\b"),
    ("extend", 0.5, r"\b(?:to|until) the (?:end|flag)\b|\bone[- ]stop(?:per)?\b"),
    ("planchange", 0.9, r"\bbox opposite\b|\bopposite strategy\b|\bswitch(?:ing)?(?: to)? (?:plan|strategy)\b|\b(?:change|changing|changed) (?:to )?(?:plan|strategy)\b"),
    ("planchange", 0.7, r"\b(?:plan|strategy) (?:b|c|d|bravo|charlie|delta|two|three)\b"),
    ("planchange", 0.3, r"\b(?:plan|strategy) (?:a|alpha|one)\b"),
    ("planchange", 0.5, r"\b(?:two|2)[- ]stop(?:per)?\b|\bplan [a-d] minus\b|\btarget (?:lap|plus)\b"),
    ("tyres_gone", 0.9, r"\b(?:tyres?|tires?|rears?|fronts?|mediums?|softs?|hards?)(?: are| is| have| has| feel| feels|'s)? (?:completely |totally |really |pretty |all )?(?:gone|dead|finished|destroyed|shot|done|dying|falling off|cooked|dropping off)\b"),
    ("tyres_gone", 0.7, r"\bno grip\b|\b(?:lost|lose|losing|lack of) (?:the |all |any )?(?:rear |front )?grip\b|\b(?:off|over) the cliff\b|\bcliff\b"),
    ("tyres_gone", 0.6, r"\bgrain\w*\b|\bblister\w*\b|\bthermal deg\w*\b"),
    ("tyres_gone", 0.4, r"\bdeg\b|\bdegradation\b|\bdegrading\b|\bsliding\b"),
    ("push", 0.8, r"\b(?:push|pushing|attack|attacking|go for it|flat out)\b(?! (?:you )?about)"),
    ("save", 0.9, r"\blift[- ]and[- ]coast\b|\bfuel sav\w+\b|\bsav(?:e|ing) (?:the )?(?:fuel|tyres?|tires?|battery|engine|brakes?)\b"),
    ("save", 0.7, r"\bmanag(?:e|ing) (?:the )?(?:tyres?|tires?|fuel|pace|gap|temps?)\b|\blook(?:ing)? after (?:the |your )?(?:tyres?|tires?)\b|\bshort[- ]?shift\w*\b|\blift (?:earlier|a bit|off|more)\b|\bcoast\w*\b"),
    ("save", 0.5, r"\bsaving\b|\bconserv\w+\b"),
    ("cover", 0.8, r"\bcover(?:ing|ed)?\b|\bundercut\w*\b|\bovercut\w*\b"),
    ("cover", 0.7, r"\b(?:he|they|[a-z]{3,12}|cars?)(?:'s|'ve| has| have| had)? (?:just )?(?:pitted|boxed|pitting)\b|\bpeople are (?:starting to )?pit(?:ting)?\b|\bhe(?:'s| is) (?:boxing|pitting|in the pits|in the pit lane)\b"),
    ("weather", 0.8, r"\b(?:rain|raining|drops|drizzle|inters?|intermediates?|wets)\b"),
]]

_NEG_BEFORE = re.compile(
    r"\b(?:don't|do not|dont|doesn't|didn't|did not|won't|will not|wouldn't|can't|cannot|not|never|no need to|no|without|nothing)\s+(?:\w+\s+){0,2}$"
    r"|\bnot (?:going to|gonna|planning to)\s*$")
_HYPO = re.compile(r"\b(?:if|in case|what if|unless)\b[^,.;]{0,45}$|\b(?:would|could|might|may|should)\s+(?:\w+\s+)?$")
_QUESTION_START = re.compile(r"^(?:so |and |ok(?:ay)?,? |right,? )?(?:should|shall|do you|do we|did you|did we|are you|are we|can we|can you|could we|could you|would you|why|how|what|when|any)\b")
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RED_BOX = re.compile(r"\bred box\b|\bbox (?:is|was|has|had)\b")


def classify(text: str | None) -> dict[str, tuple[float, str]]:
    """{intent: (score, quote)} for one transcript; only intents with score >= MIN_HIT.

    "problem" comes from the mechanic phrase table. A negated "box" is reported as "extend"."""
    out: dict[str, tuple[float, str]] = {}
    if not text:
        return out

    def put(intent: str, s: float, quote: str) -> None:
        if s >= MIN_HIT and s > out.get(intent, (0.0, ""))[0]:
            out[intent] = (round(s, 3), quote.strip()[:160])

    for sent in _SPLIT.split(text.strip()):
        low = sent.lower().replace("’", "'")
        if len(low) < 3:
            continue
        question = low.rstrip().endswith("?") or bool(_QUESTION_START.match(low.strip()))
        q = 0.35 if question else 1.0
        for intent, w, pat in _P:
            for m in pat.finditer(low):
                before = low[max(0, m.start() - 50):m.start()]
                if intent == "box" and _RED_BOX.search(low[max(0, m.start() - 4):m.end() + 6]):
                    continue
                neg = bool(_NEG_BEFORE.search(before))
                if intent == "tyres_gone" and re.match(r"^\s*(?:isn't|is not|aren't|not)\b", low[m.end():m.end() + 10]):
                    neg = True
                if neg:
                    if intent == "box":  # "don't box" is an instruction to stay out
                        put("extend", 0.75 * q, sent)
                    elif intent == "push":
                        put("save", 0.6 * q, sent)
                    continue
                hypo = bool(_HYPO.search(before))
                put(intent, 0.25 if hypo else w * q, sent)
    pi, ps, pq = classify_problem(text)
    if pi and ps >= 0.3:
        out["problem"] = (ps, pq)
    # a bare "box" next to a plan-change phrase ("box opposite"): the plan change is the message
    if "box" in out and "planchange" in out and out["box"][0] <= 0.5 and out["planchange"][0] >= 0.9:
        del out["box"]
    return out


def _decay(intent: str, dt: float) -> float:
    return 0.0 if dt > MAX_AGE_S else math.exp(-math.log(2) * max(0.0, dt) / HALF_LIFE_S[intent])


class RivalRadio(Engineer):
    name = "rivalradio"
    requires = ()  # alerts use the rivals engineer's gaps when it is on the wall
    features = ("rr_box_intent", "rr_extend", "rr_push", "rr_save", "rr_tyres_gone", "rr_planchange",
                "rr_cover", "rr_weather", "rr_problem", "rr_n", "rr_age_s")
    in_bench = True

    def __init__(self, ctx, memory) -> None:
        super().__init__(ctx, memory)
        self._cache: dict[str, dict[str, tuple[float, str]]] = {}

    @staticmethod
    def sentinels() -> dict:
        return {"rr_box_intent": 0.0, "rr_extend": 0.0, "rr_push": 0.0, "rr_save": 0.0, "rr_tyres_gone": 0.0,
                "rr_planchange": 0.0, "rr_cover": 0.0, "rr_weather": 0.0, "rr_problem": 0.0, "rr_n": 0,
                "rr_age_s": -1.0, "rr_last_t": -1.0, "rr_last_quote": "", "rr_last_intent": ""}

    def _clips(self, state, car: str) -> list[tuple[float, float, dict]]:
        """(t, known_at, intents) of the car's transcripts known now, oldest first."""
        out = []
        for m in state.feeds.radio.messages(car):
            if m.text is None:
                continue
            c = self._cache.get(m.path)
            if c is None:
                c = self._cache[m.path] = classify(m.text)
            out.append((m.t, m.known_at, c))
        return out

    @staticmethod
    def _last_pit_t(state, car: str) -> float:
        t = -1.0
        for pe in state.pit_events:
            if pe.driver == car and pe.in_t > t:
                t = pe.in_t
        return t

    def _lap_at(self, car: str, t: float) -> int | None:
        laps = self.memory.index.by_driver.get(car, {})
        for lap in sorted(laps):
            if laps[lap].t_end >= t:
                return lap
        return (max(laps) + 1) if laps else None

    def car(self, state, number, view):
        out = self.sentinels()
        if state.feeds.radio.now <= 0:
            return out
        clips = self._clips(state, number)
        out["rr_n"] = len(clips)
        if not clips:
            return out
        pit_t = self._last_pit_t(state, number)
        now = state.t
        # box/extend: the newest message carrying either signal decides; a later pit spends it
        for t, _, c in reversed(clips):
            b, e = c.get("box", (0.0, ""))[0], c.get("extend", (0.0, ""))[0]
            if max(b, e) < 0.4:
                continue
            if t > pit_t:
                if b > e:
                    out["rr_box_intent"] = round(b * _decay("box", now - t), 3)
                else:
                    out["rr_extend"] = round(e * _decay("extend", now - t), 3)
            break
        for intent, key in (("push", "rr_push"), ("save", "rr_save"), ("tyres_gone", "rr_tyres_gone"),
                            ("planchange", "rr_planchange"), ("cover", "rr_cover"), ("weather", "rr_weather"),
                            ("problem", "rr_problem")):
            best = 0.0
            for t, _, c in clips:
                if intent in c and (t > pit_t or intent in ("weather", "problem")):
                    best = max(best, c[intent][0] * _decay(intent, now - t))
            out[key] = round(best, 3)
        if out["rr_push"] and out["rr_save"]:  # the newest of the two wins
            tp = max((t for t, _, c in clips if "push" in c), default=-1)
            ts = max((t for t, _, c in clips if "save" in c), default=-1)
            out["rr_save" if tp > ts else "rr_push"] = 0.0
        last = None
        for t, _, c in clips:
            hits = {k: v for k, v in c.items() if v[0] >= 0.4}
            if hits:
                k = max(hits, key=lambda k: hits[k][0])
                last = (t, hits[k][1], k)
        if last is not None:
            out.update(rr_last_t=round(last[0], 1), rr_age_s=round(max(0.0, now - last[0]), 1),
                       rr_last_quote=last[1], rr_last_intent=last[2])
        return out

    # ------------------------------------------------------------------ alerts
    def _relation(self, state, view, focus: list[str], n: str) -> str:
        best = None
        for f in focus:
            try:
                r = view.car("rivals", f)
            except KeyError:
                r = {}
            who = "you" if len(focus) == 1 else (state.drivers[f].tla or f)
            if r.get("ahead") == n and r.get("gap_ahead") is not None:
                cand = (r["gap_ahead"], f"he's {r['gap_ahead']:.1f} s ahead of {who}")
            elif r.get("behind") == n and r.get("gap_behind") is not None:
                cand = (r["gap_behind"], f"he's {r['gap_behind']:.1f} s behind {who}")
            else:
                continue
            if best is None or cand[0] < best[0]:
                best = cand
        if best:
            return best[1]
        me = [state.drivers[f] for f in focus if f in state.drivers and state.drivers[f].position]
        d = state.drivers.get(n)
        if me and d is not None and d.position:
            diff = d.position - me[0].position
            if diff:
                return f"P{d.position}, {abs(diff)} place{'s' if abs(diff) > 1 else ''} {'behind' if diff > 0 else 'ahead of'} you"
        return ""

    def alerts(self, state, view):
        if state.feeds.radio.now <= 0:
            return []
        focus = self.ctx.team.focus(state)
        out = []
        for n in sorted(state.drivers, key=lambda x: int(x) if x.isdigit() else 999):
            if n in focus or state.drivers[n].retired:
                continue
            v = view.car(self.name, n)
            if not v["rr_n"] or v["rr_last_t"] < 0:
                continue
            lap = self._lap_at(n, v["rr_last_t"])
            tla = state.drivers[n].tla or n
            rel = self._relation(state, view, focus, n) if focus else ""
            tail = (f"; {rel}" if rel else "") + (f" (lap {lap})" if lap else "")
            q = v["rr_last_quote"]
            since = v["rr_last_t"] + 15.0
            if v["rr_box_intent"] >= ALERT_BOX:
                out.append(Alert(state.t, self.name, "rival_box", "warn", f"{tla}'s engineer: '{q}'{tail}",
                                 car=n, since=since, data={"intent": v["rr_box_intent"], "age_s": v["rr_age_s"]}))
            elif v["rr_tyres_gone"] >= 0.6:
                out.append(Alert(state.t, self.name, "rival_tyres", "info", f"{tla} on the radio, tyres going away: '{q}'{tail}",
                                 car=n, since=since, data={"tyres_gone": v["rr_tyres_gone"]}))
            elif v["rr_planchange"] >= 0.6:
                out.append(Alert(state.t, self.name, "rival_plan", "info", f"{tla}'s team talks strategy change: '{q}'{tail}",
                                 car=n, since=since, data={"planchange": v["rr_planchange"]}))
            elif v["rr_extend"] >= 0.6:
                out.append(Alert(state.t, self.name, "rival_extend", "info", f"{tla}'s engineer: staying out, '{q}'{tail}",
                                 car=n, since=since, data={"extend": v["rr_extend"]}))
        return out
