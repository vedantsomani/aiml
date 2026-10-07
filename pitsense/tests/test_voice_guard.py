"""The voice guard beyond vocabulary: a gap, a side, a lap and a tyre in the output must be the input's own for that
driver, lap and call, not just a number, name or compound found somewhere in the input.

The vocabulary check let "LEC is 2.1 ahead" through when 2.1 was the gap to the car behind, "box on lap 36" when 36
was the laps left, "box for mediums" when mediums were the tyres already on. The template composer is the fallback
when the guard refuses a model's text, so it has to pass every check the guard has.
"""

from __future__ import annotations

import dataclasses
import random
import re

import pytest

from pitsense.pitwall.types import Call, Plan, PlanStop, Reason
from pitsense.voice import composer, guard
from pitsense.voice.composer import PLURAL
from pitsense.voice.facts import ACTIONS, INTENTS, KNOWN_REASONS, Facts, Neighbour, PlanF, from_values
from pitsense.voice.synth import FREE_QUESTIONS

CALL = Facts(
    "BOX", "16", "LEC", lap=21, total=57, pos=3, cmp="MEDIUM", age=18, stops=1, fit="HARD", conf=80,
    ahead=Neighbour("1", "VER", "1.2"), behind=Neighbour("4", "NOR", "2.4"), loss="21.5", rejoin=5, cliff=60,
    deg="0.06", uc=40, sc="sc", pen=5, rain=40, must=True,
    plan_a=PlanF(((22, "HARD"),), 4), plan_b=PlanF(((25, "SOFT"), (40, "MEDIUM")), 3, "if sc comes before lap 30"),
    reasons=(("tyre_cliff", "60", None), ("rejoin_if_box_now", "5", None),
             ("other", None, "box called on lap 20 and not taken holding it")),
)
INP = CALL.text("radio")  # LAP 21 57 36, TY MEDIUM, FIT HARD, AH 1 VER 1.2, BH 4 NOR 2.4, PA 1 22 HARD, PB 2 25 SOFT 40 MEDIUM, IN 1, ...


def vocabulary_errors(errs):
    """The errors the old, vocabulary-only guard gave: a number, name or compound the input does not hold."""
    return [e for e in errs if e.startswith(("number ", "name ")) or re.fullmatch(r"compound \w+", e)]


@pytest.mark.parametrize("out", [
    "LEC, box this lap for hards.",
    "LEC, box this lap for the new hards.",
    "LEC, box on lap 22 for hards.",  # the plan's stop
    "LEC, box on lap 21, hards going on.",  # this lap, for a BOX
    "VER is 1.2 seconds ahead of LEC.",
    "NOR is 2.4 seconds behind LEC.",
    "VER is 1.2 seconds ahead and NOR is 2.4 seconds behind; a stop costs 21.5 seconds and rejoins P5.",
    "The gap to VER is 1.2 seconds, he is ahead.",
    "1.2 seconds to VER, ahead.",
    "LEC has NOR 2.4 seconds behind.",
    "NOR is behind by 2.4 seconds.",
    "Gap to VER: 1.2 seconds, ahead.",
    "At lap 21 of 57, LEC is P3, currently on 18 lap old mediums.",  # the tyres it is on, not the ones to fit
    "Plan A stops on lap 22 for hards, finishing around P4.",
    "Plan B stops on lap 25 for softs, then on lap 40 for mediums, finishing around P3.",
    "LEC pits on lap 22 under Plan A, taking hards.",
    "Plan B applies if sc comes before lap 30.",  # the trigger's own words are not a claim
    "LEC, box this lap, box called on lap 20 and not taken holding it.",  # nor is a reason's
    "LEC, box this lap, we gain 3.1 seconds on the cars ahead.",  # a gain, not a gap (with R undercut_gain 3.1)
])
def test_what_the_input_says_passes(out):
    assert guard.fact_errors(INP + " | R undercut_gain 3.1", out) == []


WRONG = [
    # wrong direction: the driver is on the other side
    ("VER is 1.2 seconds behind LEC.", "VER is not behind"),
    ("NOR is 2.4 seconds ahead of LEC.", "NOR is not ahead"),
    ("The gap to VER is 1.2 seconds, he is behind.", "VER is not behind"),
    ("1.2 seconds to VER, behind.", "VER is not behind"),
    ("NOR is ahead by 2.4 seconds.", "NOR is not ahead"),
    # wrong driver: the gap of the other neighbour, or a number from somewhere else in the input
    ("NOR is 1.2 seconds behind LEC.", "gap 1.2 is not NOR's"),
    ("VER is 2.4 seconds ahead of LEC.", "gap 2.4 is not VER's"),
    ("LEC has VER 21.5 seconds ahead.", "gap 21.5 is not VER's"),
    ("Gap to NOR: 1.2 seconds, behind.", "gap 1.2 is not NOR's"),
    ("LEC is 2.4 ahead.", "names no driver"),  # the gap to the car behind, put on the car itself
    ("LEC is 1.2 seconds ahead.", "names no driver"),
    # wrong lap: a lap the call does not name
    ("LEC, box on lap 36.", "lap 36"),  # the laps left
    ("LEC, box on lap 18.", "lap 18"),  # the tyre age
    ("LEC, box on lap 57.", "lap 57"),  # the race length
    ("LEC, box on lap 25.", "lap 25"),  # plan B's stop, not the call's
    ("At lap 22 of 57, LEC is P3.", "lap 22"),
    ("Plan A stops on lap 22 for hards, then on lap 40 for mediums.", "lap 40"),  # plan B's second stop
    ("Plan B stops on lap 30.", "lap 30"),  # the trigger's lap
    # wrong compound: the tyre it is on, or a tyre of another plan
    ("LEC, box this lap for mediums.", "compound medium"),
    ("LEC, box this lap onto softs.", "compound soft"),
    ("LEC, box on lap 22 for softs.", "compound soft"),
    ("LEC, box this lap, mediums going on.", "compound medium"),
    ("LEC, box this lap for the new mediums.", "compound medium"),
    ("LEC, box this lap and put on softs.", "compound soft"),
    ("LEC, box this lap, switch to mediums.", "compound medium"),
    ("Plan B stops on lap 25 for mediums, then on lap 40 for softs.", "compound medium"),
]


@pytest.mark.parametrize("out,error", WRONG, ids=[o for o, _ in WRONG])
def test_wrong_claims_are_rejected_though_every_word_is_in_the_input(out, error):
    errs = guard.fact_errors(INP, out)
    assert any(error in e for e in errs), errs
    assert vocabulary_errors(errs) == []  # the old guard let each of these through


def test_check_reports_the_relation():
    assert guard.check("radio", INP, "LEC, box this lap for hards.", "BOX") == []
    assert guard.check("radio", INP, "LEC, box this lap for mediums.", "BOX") == ["compound medium is not the call's tyre"]
    assert guard.check("ask_gap", INP, "VER is 1.2 seconds behind LEC.", "BOX") == ["VER is not behind"]


def test_a_token_the_vocabulary_already_rejected_is_not_reported_twice():
    assert guard.fact_errors(INP, "LEC, box on lap 24 for hards.") == ["number 24"]
    assert guard.fact_errors(INP, "VER is 1.3 seconds ahead of LEC.") == ["number 1.3"]
    assert guard.fact_errors(INP.replace("SOFT", "X"), "LEC, box this lap onto softs.") == ["compound soft"]
    assert guard.fact_errors(INP, "HAM is ahead.") == ["name HAM"]


def test_a_stop_may_be_named_for_this_lap_only_when_the_call_is_a_box():
    for action, ok in (("BOX", True), ("PREPARE_BOX", False), ("STAY_OUT", False)):
        f = dataclasses.replace(CALL, action=action, plan_a=PlanF(((23, "HARD"),), 4))
        assert (guard.fact_errors(f.text("radio"), "LEC, box on lap 21.") == []) is ok, action
        assert guard.fact_errors(f.text("radio"), "LEC, box on lap 23.") == []


def test_rivals_without_a_known_tla_are_written_by_car_number():
    f = dataclasses.replace(CALL, tla="16", ahead=Neighbour("44", "44", "1.2"), behind=Neighbour("4", "4", "2.4"))
    for task, target in (("brief", None), ("ask_gap", "44"), ("ask_gap", "4"), ("free", "how far is the car behind")):
        assert guard.check(task, f.text(task, target), composer.compose(f, task, target), f.action) == [], task
    assert guard.fact_errors(f.text("radio"), "16 is 1.2 seconds ahead.")  # a gap with a side and no driver is still refused


class Backend:
    """A model backend that always writes ``text``."""

    name = "stub"

    def __init__(self, text):
        self.text = text

    def decode(self, inputs):
        return [self.text] * len(inputs)


def test_voice_serves_the_template_when_the_model_gets_a_relation_wrong():
    from pitsense.voice.api import Voice

    v = Voice(use_model=False)
    v.backends = [Backend("VER is 2.4 seconds ahead of LEC.")]  # NOR's gap on VER
    r = v.write(CALL, "ask_gap", "1")
    assert (r.source, r.fallback, r.text) == ("template", True, composer.answer(CALL, "gap", "1"))
    assert r.errors == ("gap 2.4 is not VER's",) and r.raw == "VER is 2.4 seconds ahead of LEC."
    v.backends = [Backend("VER is 1.2 seconds ahead of LEC.")]
    r = v.write(CALL, "ask_gap", "1")
    assert (r.source, r.fallback, r.text) == ("stub", False, "VER is 1.2 seconds ahead of LEC.")


def test_odd_input_and_output_never_raise():
    odd = ("", "radio S000000", "radio S0 | AH | PA x | PB 3 | LAP 3 | ACT | CAR | R other ~ ", INP[:60], INP.replace(" | ", " |  | "))
    for inp in odd:
        for out in ("", "....", "VER is 1.2 seconds ahead", "Plan A stops on lap 3 for hards", "lap 21 of 57", "Plan"):
            assert isinstance(guard.fact_errors(inp, out), list)


# ---------------------------------------------------------------------------------------- the template fallback
TLAS = (("16", "LEC"), ("44", "HAM"), ("1", "VER"), ("4", "NOR"), ("63", "RUS"), ("81", "PIA"), ("55", "SAI"), ("14", "ALO"))
COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
HEAD_TEXTS = (  # reason texts as the head of strategy writes them: laps, compounds and "ahead" / "behind" in free text
    "best plan stops on lap 22 (HARD)", "BOX called on lap 20 and not taken: holding it",
    "BOX for INTERS: rain flag up for 12 min and no wet-tyre model", "pit lane is closed: box as soon as it opens",
    "field: 12 on slicks, 5 on inters, 1 on wets", "stopping now averages P4.2; waiting P5.0",
    "crossover reached: inters 1.5 s/lap faster", "fuel saving is needed", "the car behind is catching us at 0.4 s/lap",
)
TRIGGERS = ("if sc comes before lap {lap}", "if rain starts within 10 minutes", "if the undercut threat passes 40 percent",
            "if the tyres fall off before lap {lap}", "if the car behind pits first", "SC/VSC within 5 laps: box that lap for HARD", None)


def random_plan(rng, name, lap, total, trigger=None):
    stops, at = [], lap + rng.randint(0, 12)
    for _ in range(rng.choice((0, 1, 1, 2))):
        if at < total:
            stops.append(PlanStop(at, rng.choice(COMPOUNDS)))
        at += rng.randint(8, 20)
    return Plan(name, tuple(stops), float(rng.randint(1, 20)) if rng.random() < 0.7 else None, trigger)


def random_reasons(rng):
    out = []
    for _ in range(rng.choice((0, 1, 2, 3))):
        if rng.random() < 0.3:
            out.append(Reason("note", rng.choice(HEAD_TEXTS), rng.choice((None, 1.5, True))))
        else:
            code = rng.choice(KNOWN_REASONS)
            value = rng.choice((None, round(rng.uniform(0.5, 30), 1), rng.randint(1, 80), rng.choice(("44", "1", "4"))))
            out.append(Reason(code, code, value))
    return tuple(out)


def random_facts(rng: random.Random, k: int) -> Facts:
    """One situation as the voice gets it (``from_values``): sparse or full, any call, any plans, now and then rivals without a TLA."""
    (car, tla), (ahead, ahead_tla), (behind, behind_tla) = rng.sample(TLAS, 3)
    tla_of = {car: tla, ahead: ahead_tla, behind: behind_tla} if rng.random() > 0.15 else {car: tla}
    total = rng.choice((44, 52, 57, 58, 66, 70))
    lap = rng.randint(2, total - 1)
    action = rng.choice(ACTIONS)
    trigger = rng.choice(TRIGGERS)
    trigger = trigger.format(lap=lap + rng.randint(2, 15)) if trigger else None
    call = Call(1000.0 + k, car, action, None if action in ("STAY_OUT", "NO_CALL") else rng.choice(COMPOUNDS), rng.choice((None, 0.8)),
                random_reasons(rng), random_plan(rng, "A", lap, total) if rng.random() < 0.9 else None,
                random_plan(rng, "B", lap, total, trigger) if rng.random() < 0.7 else None)
    maybe = lambda *x: rng.choice((None, *x))  # noqa: E731
    v = {"t": call.t, "lap": lap, "total_laps": total, "position": rng.randint(1, 20), "compound": rng.choice(COMPOUNDS),
         "tyre_age": rng.randint(0, 40), "pit_stops": rng.randint(0, 3), "rules__sc_phase": maybe("none", "sc", "vsc", "sc_ending", "red"),
         "pitstop__loss_now": maybe(round(rng.uniform(18, 30), 1)), "pitstop__rejoin_if_box_now": maybe(rng.randint(1, 20)),
         "tyre__cliff_risk": maybe(rng.random()), "tyre__deg_s_per_lap": maybe(rng.uniform(0.01, 0.15)),
         "rivals__undercut_threat": maybe(rng.random()), "rules__penalty_s_pending": rng.choice((0, 0, 5, 10)),
         "rules__drive_through_pending": rng.random() < 0.1, "weather__rain_prob_10min": maybe(rng.random()),
         "weather__crossover": maybe("none", "to_inters"), "rules__must_stop": rng.random() < 0.3}
    for key, rival in (("ahead", ahead), ("behind", behind)):
        if rng.random() < 0.8:
            v[f"rivals__{key}"] = rival
            v[f"rivals__gap_{key}"] = maybe(round(rng.uniform(0.2, 40), 1), round(rng.uniform(0.2, 4), 2), float(rng.randint(1, 9)))
    return from_values(call, v, tla_of, tuple(rng.randint(0, 5) for _ in range(6)))


def tasks(f: Facts):
    """Everything the voice writes for one set of facts: (task, target) pairs."""
    out = [("radio", None), ("brief", None)]
    out += [("ask_gap", n.car) for n in (f.ahead, f.behind) if n] + [("ask_gap", "99")]
    out += [(f"ask_{i}", None) for i in INTENTS if i != "gap"]
    return out + [("free", q) for q in FREE_QUESTIONS]


def spread(n: int, seed: int = 2026):
    """(facts, task, target, model input, template text) for ``n`` random situations and every task."""
    rng = random.Random(seed)
    for k in range(n):
        f = random_facts(rng, k)
        for task, target in tasks(f):
            yield f, task, target, f.text(task, target), composer.compose(f, task, target)


def test_template_outputs_pass_the_guard_across_a_spread_of_fact_sets():
    seen = dict.fromkeys(("gap", "stop lap", "tyre"), 0)
    for f, task, target, inp, out in spread(100):
        assert guard.check(task, inp, out, f.action) == [], (task, target, inp, out)
        seen["gap"] += bool(re.search(r"[A-Z]{3}.*\d\.\d|\d\.\d seconds to [A-Z]{3}", out) and re.search(r"ahead|behind", out))
        seen["stop lap"] += bool(re.search(r"(?:stops|stopping|box|pits|stop|then) on lap \d+", out))
        seen["tyre"] += bool(re.search(r"\b(?:for|onto|take|taking|plan on) \w+s\b|\w+s (?:going on|ready|next)", out))
    assert all(n > 100 for n in seen.values()), seen  # the spread does reach the relational checks


def test_template_outputs_pass_the_guard_when_the_race_length_is_unknown():
    """Some live feeds never say how many laps the race has: the input keeps the lap, which the brief still says."""
    for f, task, target, _, _ in spread(30, seed=7):
        f = dataclasses.replace(f, total=None)
        inp, out = f.text(task, target), composer.compose(f, task, target)
        assert guard.check(task, inp, out, f.action) == [], (task, target, inp, out)


def side_flipped(out):
    return re.sub(r"\b(ahead|behind)\b", lambda m: "behind" if m.group(1) == "ahead" else "ahead", out)


def test_a_template_text_made_wrong_is_rejected():
    """The mistakes the guard exists for, made in the template's own sentences across the spread. Every mutant keeps
    to words the input holds, so the old guard passed it."""
    made = dict.fromkeys(("direction", "driver", "lap", "compound"), 0)

    def attempt(kind, task, inp, mutant):
        made[kind] += 1
        errs = guard.fact_errors(inp, mutant)
        assert errs and not vocabulary_errors(errs), (kind, task, inp, mutant, errs)

    for f, task, _, inp, out in spread(300):
        said = [n for n in (f.ahead, f.behind) if n and n.tla.isalpha() and n.gap and re.search(rf"\b{n.tla}\b", out) and n.gap in out]
        if said and re.search(r"seconds (?:ahead|behind)|,? (?:he is )?(?:ahead|behind)\b", out):
            attempt("direction", task, inp, side_flipped(out))  # "VER is 1.2 seconds behind", with VER ahead
        if len(said) == 2:  # the brief names both: VER's gap on NOR and NOR's on VER
            attempt("driver", task, inp, out.replace(said[0].tla, "§").replace(said[1].tla, said[0].tla).replace("§", said[1].tla))
        if task == "ask_gap" and len(said) == 1 and f.ahead and f.behind:  # the answer about the other car's name
            other = f.behind if said[0] is f.ahead else f.ahead
            if other.tla.isalpha():
                attempt("driver", task, inp, out.replace(said[0].tla, other.tla))
        if task == "ask_pit_window" and f.plan_a and f.plan_a.stops:
            lap = str(f.plan_a.stops[0][0])
            right = {lap, str(f.lap), *(str(s[0]) for p in (f.plan_a, f.plan_b) if p for s in p.stops)}
            wrong = [str(x) for x in (f.total, f.left, f.age, f.pos, f.rejoin, f.stops) if x is not None and str(x) not in right]
            if wrong and re.search(rf"\blap {lap}\b", out):  # the laps left, the tyre age, ... for the stop lap
                attempt("lap", task, inp, re.sub(rf"\blap {lap}\b", f"lap {wrong[0]}", out, count=1, flags=re.I))
        if f.fit and task in ("radio", "brief") and PLURAL[f.fit] in out:
            others = [c for c in COMPOUNDS if c != f.fit and c.lower() in inp.lower()]  # the tyres it is on, or another plan's
            if others:
                attempt("compound", task, inp, out.replace(PLURAL[f.fit], PLURAL[others[0]]))
    assert all(n >= 40 for n in made.values()), made  # enough of each kind to mean something
