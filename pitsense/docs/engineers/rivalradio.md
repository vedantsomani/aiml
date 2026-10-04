# Rival radio: what the other teams tell their drivers

Code: `pitwall/engineers/rivalradio.py` (`RivalRadio`, `classify`), evaluation `bench/rivalradio.py`
(`python -m pitsense rivalradio-eval`, `rivalradio-leak`), tests `tests/test_rivalradio.py`.
`in_bench = True` (it is leak-checked and its keys are bench columns), but it needs `feeds=True` logs:
on the normal benchmark (timing only) every key keeps its sentinel, so the core models and leaderboard are
unchanged (checked: Hungary 2025 rebuilt, 14 new constant columns, all others identical; a boosted model fit with
and without constant columns gives identical predictions).

## Model

`classify(text)` returns `{intent: (score, quote)}` per transcript (phrase/regex model; transcript known 15 s after
the message, so the value at `t` never uses later audio). Intents: `box` ("box box", "box this lap", "we're going to box
next lap"), `extend` ("stay out", "extend", "go long", "keep extending"), `planchange` ("Plan B/C", "Strategy C", "box
opposite", two-stop), `tyres_gone` ("tyres are gone", "no grip", "graining", "cliff", "deg"), `push`, `save` ("lift and
coast", "fuel save", "manage the tyres"), `cover` ("cover", "undercut", "they've pitted"), `weather` ("rain", "drops", "inters"),
`problem` (the mechanic table, `mechanic_radio.classify`).

* Negation voids a hit ("tyres are not gone", "can't stay out"); a negated box becomes `extend` ("don't box"), a negated push becomes `save`.
* A question ("should we box?", "why didn't we box?", "do you want to box this lap?") counts 0.35 x; "if / in case / would box" counts 0.25.
* "red box" and "box is ..." are ignored; "box to overtake" (frequent in Whisper output, meaning unclear) is 0.35.
* The newest box/extend clip decides ("stay out" after "box box" cancels the intent); a pit stop after the clip spends it.

## Keys (car scope)

| key | meaning | sentinel |
|---|---|---|
| `rr_box_intent` | 0-1, newest box clip x decay (half-life 120 s, zero after 15 min or after the car pits) | 0 |
| `rr_extend`, `rr_push`, `rr_save`, `rr_tyres_gone`, `rr_planchange`, `rr_cover`, `rr_weather`, `rr_problem` | strongest clip of that intent x decay (half-lives 240-900 s); push and save: newest wins | 0 |
| `rr_n` | transcripts known so far for the car (coverage) | 0 |
| `rr_age_s` | seconds since the last classified (>= 0.4) clip | -1 |
| `rr_last_t`, `rr_last_quote`, `rr_last_intent` | that clip's time (not a feature), text, intent | -1, "", "" |

`features` = every numeric key except `rr_last_t`. Alerts (`rival_box`, `rival_tyres`, `rival_plan`, `rival_extend`) go to
the focus team (`ctx.team`; the focus cars' own radio is skipped), e.g. `CCC's engineer: 'Box box, box this lap.'; he's 1.2 s ahead of you (lap 23)`.
The gap comes from the `rivals` engineer when it is on the wall, else "P3, 1 place behind you".

## Radio coverage (honest)

Radio exists for 2025-26 only: 24 of 24 races in 2025, 15 of 24 in 2026 (4 of those 15 have no clips at all, so 11 are usable;
Australia, China, Japan, Miami 2026 have none). 773 clips in 2025, 335 in 2026, median 30 per race for 20 cars, so
most cars say nothing in a given race. Of 1408 green-flag stops, only 84 (6 %) have any clip from that car in the 2 laps before them,
and only 41 clips of 1108 mention "box" at all (Whisper also garbles some). The ceiling on recall is therefore a few percent.

## Evaluation (`rivalradio-eval`; theta chosen on 2025, 2026 scored once)

Signal = `rr_box_intent >= theta` at the moment a transcript becomes known. Hit = the car's next green stop is at the end of the
lap in progress or the next one. Base rate (random car at a lap end stops within 2 laps): 6.0 % (2025), 6.5 % (2026).

| split | theta | signals | hits | precision | recall (stops) | lead median (s) [p25-p75] |
|---|---|---|---|---|---|---|
| 2025 (tune), 841 stops | 0.4 | 12 | 3 | 0.25 | 0.004 | 69 [63-74] |
| 2026 (test, once), 567 stops | 0.4 | 15 | 2 | 0.13 | 0.004 | 62 [53-71] |

Theta 0.4-0.8 on 2025 all give 2-3 hits (F1 is flat; 0.4 has the most). The few hits are real (about 1 minute of warning, i.e. the
transcript lands about a lap before the stop shows in timing), but the precision on 2026 (2 of 15) is not distinguishable from
chance given the numbers, and recall is 0.4 %. `extend` signals (>= 0.6): car did not stop within 2 laps in 16 of 18 (2025) and 2 of 3 (2026).
The classifier is no better than the data: most "box" clips are about the other car, a plan, a past stop, or Whisper noise.

### pit_within_1/3 ablation (`TASK_MODELS` entries `rr_gbm_with` / `rr_gbm_without`; core models untouched)

Rows rebuilt with the radio feed on (`data/bench/v0.1-rr`, 35 radio races, 39 126 rows; the car had a clip known in 13 057 of them, but
`rr_box_intent > 0` in only 151). Same settings as `gbm_hazard`, expanding window, trained only on earlier radio races.

| test year | task | log loss without rr | with rr | races better / total |
|---|---|---|---|---|
| 2025 (tune, >= 8 earlier races) | pit_within_1 / pit_within_3 | 0.1166 / 0.2558 | 0.1166 / 0.2558 | 0 / 16 |
| 2026 (scored once) | pit_within_1 / pit_within_3 | 0.1254 / 0.2887 | 0.1254 / 0.2887 | 0 / 11, 1 / 11 |

No change: with `min_samples_leaf=200` the trees never split on columns that are non-zero in a few hundred of 39 000 rows. Looser
leaves would overfit those rows (rows with a box intent stop within 3 laps 4.8 % of the time vs 9.2 % base, n = 21; extend 24 %, tyres_gone 19 %, n = 25).
Verdict: rival radio is an alert source for a human, not a model feature, until far more clips are transcribed.

## Run

```
python -m pitsense rivalradio-eval --tune-year 2025 --year 2026 --jobs 3   # records to data/scratch/rivalradio, rows to data/bench/v0.1-rr
python -m pitsense rivalradio-leak --year 2026 --race hungary --cuts 3       # corrupted-future test with the radio feed loaded
```

`pitsense leakcheck` loads timing only (no feeds), so it passes trivially for this engineer; `rivalradio-leak` is the one that
exercises it (Hungary 2026, 2 cuts, truncate and scramble: 0 mismatches). Ask the lead to give `leakcheck` a `--feeds` flag.
