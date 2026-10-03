# Tyre & performance engineer (`tyre`)

Code: `src/pitsense/pitwall/engineers/tyre/` (`engineer.py` state and caching, `model.py` maths),
benchmark: `src/pitsense/bench/tyre.py`, tests: `tests/test_tyre*.py`.

## Keys

Per car (`tyre__<key>`):

| Key | Meaning | Unit |
|---|---|---|
| `pace_s` | expected next clean lap on the current tyres, fuel included | s |
| `deg_s_per_lap` | lap-time increase per extra lap on the current set, fuel removed | s/lap |
| `fresh_soft_s`, `fresh_medium_s`, `fresh_hard_s` | expected first flying lap on a new set fitted at the end of the next lap (out-lap in between) | s |
| `cliff_risk` | probability of a sudden loss >= 1 s/lap on two consecutive laps within 3 laps (free air) | 0-1 |
| `pace_sd_s` | uncertainty of `pace_s` | s |
| `pace_trend_s` | pace without the short-term deviation (traffic, push lap) | s |
| `deg_sd` | uncertainty of `deg_s_per_lap` | s/lap |
| `stint_pace`, `stint_clean_laps`, `last_clean_s` | median / count / last of clean laps on the current set | s, count, s |
| `resid_last_s`, `resid_trend_s` | last lap vs the model; recent residuals vs earlier ones | s |
| `traffic` | last lap started within 1 s of the car ahead | bool |

Race: `fuel_s_per_lap` (lap time gained per lap from fuel burn plus track evolution, s/lap),
`fuel_sd`, `field_deg_<compound>`, `off_soft_s`, `off_hard_s` (compound offset vs medium),
`field_laps`. Keys are None until there are enough clean laps (the field fit falls back to
2025 priors with fewer than 40 clean laps). Declared benchmark `features`: `pace_s`,
`deg_s_per_lap`, `fuel_s_per_lap`, `cliff_risk`, `pace_sd_s`, `resid_trend_s`, `fresh_*_s`.

## Method

Clean lap = timed, not lap 1, not an in/out-lap, green throughout, dry compound, and within
1.07x of the field's 5th-fastest clean lap so far (some neutralised laps are published under
green). A lap started within 1 s of the car ahead counts half.

* Lap model: `t = base_car + off_compound + deg_compound * age - fuel * lap + noise`.
  Within a stint age and lap move together, so each stint gives only the net slope
  `deg - fuel`. Field fit, refit every 10 laps: (1) robust net slope per compound within
  stints, shrunk to a prior; (2) across stops (age resets, fuel load does not) stint levels
  give `fuel` and the compound offsets. `deg = net + fuel`.
* Car pace: a two-part Kalman filter, a slow random-walk level plus a fast decaying
  deviation. `pace_s` leans on the fast part; longer horizons fall back to the slow level.
  The car's own wear slope on the current set is shrunk toward the field's.
* `fresh_*`: the car's level plus the compound offset at age 2, fuel for the stop lap.
* `cliff_risk`: logistic on age / typical compound life, race fraction, free air, pace
  residuals and the set's net slope. Coefficients (`CLIFF_B`) fitted on 2025 labels only.
* All values are as-of-now; no cross-race state, deterministic.

## Benchmark tasks (labels in `bench/tyre.py`, read the finished race only)

`next_lap_time` (time of L+1), `lap_time_5` (time of L+5, no stop between),
`tyre_cliff_3` (cliff within 3 laps, NaN when unobservable), `fresh_tyre_pace` (first flying
lap after a stop, given the fitted compound). Labels ignore laps slower than 1.25x the race's
median clean lap (neutralised laps published under green; Dutch/Italian 2026 had ~170-200 s
"green" laps). Learned models (`*_gbm`) are residual boosters on top of the engineer's values,
trained only on earlier races. Settings tuned on 2025 (`--test-year 2025 --min-train 3`),
2026 scored once afterwards.

## Results, 2026 (races scored by models trained on earlier races)

next_lap_time (s):

| model | n | MAE | RMSE | bias |
|---|---|---|---|---|
| next_lap_gbm | 13678 | 0.670 | 4.25 | 0.14 |
| tyre_pace (`pace_s`) | 13678 | 0.718 | 4.46 | 0.24 |
| last_clean_lap | 13678 | 1.498 | 9.54 | 1.05 |
| last_lap | 13678 | 1.659 | 8.99 | 1.23 |
| stint_median | 13678 | 1.729 | 9.86 | 1.16 |

lap_time_5 (s):

| model | n | MAE | RMSE |
|---|---|---|---|
| lap5_gbm | 9127 | 0.872 | 4.83 |
| tyre_pace | 9127 | 0.918 | 5.00 |
| tyre_extrap (`pace + 4(deg - fuel)`) | 9127 | 0.931 | 5.00 |
| last_clean_lap | 9127 | 1.803 | 11.06 |
| last_lap | 9127 | 1.871 | 10.05 |
| stint_median | 9127 | 2.042 | 11.45 |

tyre_cliff_3 (base rate 3.3%):

| model | n | log loss | brier | AUC | avg precision | brier skill |
|---|---|---|---|---|---|---|
| cliff_gbm | 8929 | 0.1268 | 0.0305 | 0.800 | 0.149 | 0.058 |
| tyre_cliff_risk | 8929 | 0.1384 | 0.0317 | 0.718 | 0.084 | 0.020 |
| base_rate | 8929 | 0.1489 | 0.0324 | 0.407 | 0.027 | 0 |

fresh_tyre_pace (s):

| model | n | MAE | RMSE | bias |
|---|---|---|---|---|
| fresh_gbm | 291 | 0.823 | 1.31 | -0.17 |
| tyre_fresh (`fresh_*_s`) | 291 | 0.836 | 1.29 | 0.06 |
| stint_best | 291 | 1.005 | 2.04 | 0.42 |

## Core leaderboard (2026), before vs after declaring tyre features

| task | model | before | after |
|---|---|---|---|
| pit_within_1 | gbm_hazard | log loss 0.1182, AUC 0.798 | 0.1179, 0.800 |
| pit_within_3 | gbm_hazard | log loss 0.2703, AUC 0.759 | 0.2690, 0.762 |
| position_after_stop | rejoin_gbm | exact 0.5385, MAE 0.739 | 0.5422, 0.743 |

Baselines in those tasks are unchanged. The gains are small; position_after_stop MAE is slightly worse.

## Caveats

* RMSE is dominated by a few laps after red-flag restarts; compare MAE.
* `cliff_risk` as a fixed formula is weakly calibrated-skilled (Brier skill 0.02); the learned
  `cliff_gbm` is better but is a benchmark model, not an engineer value.
