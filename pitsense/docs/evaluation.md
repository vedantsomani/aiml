# Evaluation methodology

Code: `src/pitsense/bench/callscore.py` (fair call scoring), `bench/decision.py` (decision value),
`bench/provenance.py` (stamps), `bench/report_all.py` (`pitsense results`), `bench/readme_sync.py` (README block).
Tests: `tests/test_decision.py` (synthetic races and fields). Results: `reports/results.md` / `results.json`.

Three questions are kept apart, because mixing them is what made the old numbers hard to read:

1. **Prediction:** do the calls match what the team did? (fair call scoring)
2. **Value:** would the call have been worth making? (decision value, a simulator estimate)
3. **Trust:** how good is the simulator that answers 2? (calibration)

## Protocol

* **Test year 2026, scored once.** Every tuned component was tuned or trained without it; `pitsense results`
  runs a hold-out audit (automatic checks for the benchmark split, the models bundle cutoff and the voice corpus
  split; declared tuning years for hand-tuned settings, with the doc that records them).
* **As-of.** The pit wall is replayed event by event (`RaceState.apply`), with history and the models bundle
  trained on races that ended before the first 2026 race. The finished race is read only to label and to define
  the "team" option; nothing from it reaches an engineer.
* **Cars.** The grid top 4 of each race (a pre-race fact, so no selection on the result). Four focus cars keep the
  strategy engineer in full mode (288 futures, Plan B / BOX_IF_SC), as for a team's two cars. The older
  `strategy_calls` benchmark uses the top-5 *finishers*, which puts the engineer in its light mode (96 futures, no
  Plan B), so it almost never calls BOX_IF_SC; its legacy numbers are still reported for continuity.
* **Moments.** Calls every second lap of each focus car (laps 4, 6, ...), as in the strategy benchmark. Scores for
  prediction use calls as logged (on change); decision value uses every moment, with the call in force.
* **Deterministic.** Futures are seeded from the race (`seed_for`), bootstrap seeds are fixed, results are cached
  per race and settings hash in `data/scratch/decision`, so re-scoring costs nothing.

## Fair call scoring (`callscore`)

From the call's lap L (the lap the car is about to drive, i.e. its in-lap if it boxes now):

| Action | Right when | Note |
|---|---|---|
| BOX | a real stop in L..L+2 | the legacy +-2 rule also credited stops *before* the call |
| PREPARE_BOX | a real stop in L..L+3 | "box within 2 laps", one lap of slack |
| STAY_OUT | no real stop in L..L+2 | |
| BOX_IF_SC | triggered: the car drove an SC/VSC lap in L..L+4 (Plan B's window); then right with a stop within 2 laps of the first such lap | untriggered calls are counted, not scored; "held" = the car stayed out |

Red-flag stops are excluded. Recall is split into green-flag stops and stops on SC/VSC laps. `shadow_score`
(the runtime's per-race score) no longer counts BOX_IF_SC as a box; it reports it under `box_if_sc` with the
trigger rule. Uncertainty: per-race bootstrap (races resampled with replacement, ratio of sums, 2000 draws), 90 %
intervals. Splits: circuit, weather (wet = inters or wets used), race phase (thirds of the distance), safety car in
the race, track status at the call.

## Decision value (`decision`): a simulator estimate

At each moment the strategy engineer's analysis (ranked plans) is taken as-of, and a second, independent set of
288 futures (seed tag `dv`) scores these options on the same futures:

| Option | Plan |
|---|---|
| now | best plan (simulator utility) whose first stop is this lap |
| soon / later | best plan stopping in 1-2 laps / in 3+ laps or never |
| stay | best plan not stopping this lap |
| sc | `stay`, but in futures with an SC/VSC starting within 5 laps the first stop moves to that lap (tyre and second stop as Plan B builds them) |
| team | the team's real remaining in-laps and tyres, as a fixed plan (skipped when it fitted a wet tyre) |

Our call maps to an option (BOX -> now, PREPARE_BOX -> soon, STAY_OUT -> stay, BOX_IF_SC -> sc) and an
alternative (BOX <-> stay, PREPARE_BOX -> now, BOX_IF_SC -> stay). Per option: expected finishing position, expected
points, P(lose >= 2 places versus the position at the moment). Reported: regret (ours minus the best of now / soon /
later / sc / team, in places and points), ours versus the alternative, ours versus the team, P(ours ahead of the
team's plan in a future).

Why fresh futures: plans are chosen on the engineer's own futures; scoring them there would give the chosen plan a
winner's-curse bonus over the fixed team plan. Limits, stated next to every number: all plans are open-loop (they do
not react to the simulated race, while the team's real choices did, which handicaps `team` a little), and the
simulator's own simplifications apply (docs/engineers/strategy.md: one SC and one VSC per future, no blue flags,
rivals that do not react). **They are estimates of value under the model, not measured gains.**

## Simulator calibration

For the plan the team really drove (`team`), the simulated finishing distribution is compared with where the car
really finished: RPS against the current-position forecast, MAE and bias of the expected finish, 80 % interval
coverage, the PIT histogram (flat when calibrated), points bias, and reliability of P(lose 2+ places). The
benchmark's `strategy_finish` task (RPS at 25 / 50 / 75 % distance) is reported beside it.

## One versioned report

`pitsense results [--run strategy decision voice quali] [--readme]` writes `reports/results.json` and
`results.md`. Each component file carries a stamp: data version (race count, manifest and history hashes), config
(strategy `SETTINGS` and simulator `PARAMS`, with a hash), training cutoff, git commit (with a dirty flag) and
run time. Every headline number in results.json repeats the stamp of its source; the report warns when components
come from different data or configs. The README's results block (`<!-- results:begin -->` ... `end`) is
regenerated from results.md by `bench/readme_sync.py`; no other README text quotes a number.

Reproduce (about an hour at `--jobs 3`):

```bash
pitsense bench run --tasks pit_within_1 pit_within_3 position_after_stop next_lap_time lap_time_5 tyre_cliff_3 \
    fresh_tyre_pace pit_loss undercut_5 rain_10min sc_within_2 laps_to_stop
pitsense voice eval --out data/scratch/voice_eval.json
pitsense results --run strategy decision quali voice --voice-json data/scratch/voice_eval.json --readme
```
