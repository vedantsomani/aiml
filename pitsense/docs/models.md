# Models: cross-race models on the pit wall

The benchmark refits every model for every test race on the races before it. The pit wall
cannot refit mid-race, so `pitsense train` fits the chosen model of every row task **once**
on all races that ended before a cutoff and saves a bundle (`data/models/<version>_<cutoff>.pkl`).
The `models` engineer serves that bundle live from rows built exactly like the benchmark's.

```
pitsense train                              # every benchmark race
pitsense train --before 2026 hungary        # races that ended before that race started
pitsense train --before 2026-07-26T13:00Z   # or before a UTC time
```

## Bundle (`pitsense.modelstore`)

`TrainedBundle`: `train_end_utc`, `cutoff_utc`, `trained_on` (race ids), `feature_version`,
`git_commit`, `models` (task to fitted model), `metrics` (task to holdout scores).

- **Training window.** A race counts as finished at `start_utc + 4 h` (the evaluator's
  `RACE_SPAN`). A bundle trained with `cutoff = race start` is the same model the benchmark
  scores that race with.
- **Guard.** `load_bundle(path, race_start_utc)` and `Context.for_race(models=...)` raise
  `LeakageError` unless `train_end_utc < race start`.
- **Choice of model** (fixed table `BEST_MODEL`, picked on 2025 with `--test-year 2025`, so 2026 is
  untouched): `gbm_hazard` for `pit_within_1/3`, `pitstop_gbm` for `position_after_stop`,
  `next_lap_gbm`, `lap5_gbm`, `cliff_gbm`, `fresh_gbm`, and the rule `pitstop_loss` for `pit_loss`.
  Others fall back to the task's last model.
- **Stored metrics** are holdout scores (refit without the last 3 training races, scored on them);
  the saved model is fit on all. Deterministic: models use fixed seeds; the same data gives
  identical predictions.
- Pickle is code: load only bundles you trained yourself. Bundles and reports are not committed.

## Engineer `models` (per car, at the latest lap end)

| Key | Row | Model | Meaning |
|---|---|---|---|
| `pit_prob_1`, `pit_prob_3` | `lap_end` at the car's latest completed lap | `gbm_hazard` | probability of a stop within 1 / 3 laps (0-1) |
| `rejoin_pred` | `pit_entry` for a stop on the next lap | `pitstop_gbm` | position the car would rejoin in |

Rows come from `bench.features.base_row` (shared with `FeatureBuilder`) plus the values of every
bench engineer (tyre, pitstop, rivals, rules, weather). With no bundle, or when no decision is due
(car in the pits, last lap, red flag), all three are None.

## Model cards

Common to all: trained on 2025-26 races before the cutoff only (`ctx.models` checked against the
race start); inputs are `feature_columns()` of the benchmark (base features plus declared engineer
features); one row per car per lap end (or pit entry); 2026 scores are the leaderboard
(`pitsense bench run`), each race scored by models trained on earlier races only.

**pit_within_1 / pit_within_3 (`gbm_hazard`).** Histogram gradient boosting, 150 small trees.
Input: tyre age, stint, gaps, pace trend, track status, rivals' stops, engineer values. 2026:

| task | n | log loss | brier | AUC | base-rate log loss |
|---|---|---|---|---|---|
| pit_within_1 | 16,177 | 0.1181 | 0.0281 | 0.799 | 0.1375 |
| pit_within_3 | 16,177 | 0.2689 | 0.0754 | 0.762 | 0.3102 |

Calibration (predicted / observed): within 1 lap, 0.010/0.009, 0.032/0.043, 0.070/0.064,
0.132/0.146, 0.235/0.298, 0.357/0.455 (bins up to 0.02, 0.05, 0.1, 0.2, 0.3, 0.5): good below
0.2, **under-confident above 0.2**. Within 3 laps: 0.017/0.039, 0.032/0.031, 0.071/0.097, 0.143/0.137,
0.247/0.245, 0.358/0.423, 0.523/0.692: well calibrated in the middle, under-confident at the
extremes. Probabilities are not recalibrated (isotonic was tried by the benchmark authors and lost).

**position_after_stop (`pitstop_gbm`).** Correction to the pit-stop engineer's rejoin rule. 2026
(n=533): MAE 0.584, exact 60.8%, within one place 89.7%. The rule it corrects, `pitstop_rule`, is
slightly better in 2026 (MAE 0.567, exact 62.9%) though the GBM won on 2025 (MAE 1.078 vs 1.093); the
choice follows the protocol and is a candidate for revisiting. Both beat `gap_minus_pitloss`
(MAE 0.99) and `no_change` (1.97).

**Other bundle tasks** (stored, not yet served live), 2026 MAE or log loss: `next_lap_gbm` 0.670 s
(vs last clean lap 1.498), `lap5_gbm` 0.872 s, `cliff_gbm` log loss 0.1268 (base rate 0.1489),
`fresh_gbm` 0.823 s, `pitstop_loss` 3.38 s.

## Failure modes

- **Short training window.** Only 2025-26 are in the benchmark (34-39 races). Rare conditions (red
  flags, wet races, safety-car restarts) have few examples; probabilities there are least reliable.
- **Under-confidence at high probability**, see calibration. Treat >0.3 as "likely", not as 30%.
- **Pit-wall-only inputs.** Models see only what the benchmark's rows contain. A team's private
  plans (fuel, a planned undercut) are invisible, so the prediction is the field's tendency.
- **Stale rows.** The row is built from the car's latest completed lap, so mid-lap events are seen
  at the next lap end except where the engineers read live state (track status, gaps).
- **`rejoin_pred` while a car is in the pits** describes a stop on the next lap, not the current one.
- **Feature drift.** A bundle is tied to `feature_version`; `latest_bundle_for` ignores bundles with
  another version. Retrain when features change.
- **Missing data** (feed gaps) arrives as NaN, which the boosting models handle, but predictions from
  rows with many NaNs are weaker.

## Checks

- Parity: replaying a race, the engineer's rows equal the benchmark's in every shared column
  (synthetic test, and real races: Italian and Hungarian 2026, 119k and 162k cells, zero differences).
- Equivalence: a bundle with `cutoff = race start` matches the benchmark's predictions on that
  race exactly (max difference 0.0, three tasks, two races), live and from the stored rows.
- Guard: a bundle trained after the race started is refused (`tests/test_models.py`).
- Latency (Hungary 2026, mid-race, 21 cars, this machine): about 15 ms per car, 240 ms for every car's
  `models` values with all dependency engineers computed fresh, 240 ms for a full snapshot.
