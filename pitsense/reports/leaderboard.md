# PitSense-Bench leaderboard

Test races: 15 (2026), each scored with models trained only on races that finished before it started. Decision points: 43,123 from 39 races. Feature version v0.1.

## pit_within_1

| model | n | positives | log_loss | brier | auc | avg_precision | brier_skill |
|---|---|---|---|---|---|---|---|
| gbm_hazard | 16177 | 498 | 0.1182 | 0.0281 | 0.7979 | 0.1374 | 0.0567 |
| tyre_age_logit | 16177 | 498 | 0.1311 | 0.0294 | 0.6989 | 0.0713 | 0.0142 |
| base_rate | 16177 | 498 | 0.1375 | 0.0298 | 0.4831 | 0.0287 | 0.0000 |

Calibration of `gbm_hazard` (predicted vs observed rate):

| bin | n | predicted | observed |
|---|---|---|---|
| (-0.001, 0.02] | 10047 | 0.0100 | 0.0086 |
| (0.02, 0.05] | 3143 | 0.0321 | 0.0385 |
| (0.05, 0.1] | 1835 | 0.0729 | 0.0692 |
| (0.1, 0.2] | 1047 | 0.1323 | 0.1280 |
| (0.2, 0.3] | 95 | 0.2369 | 0.2737 |
| (0.3, 0.5] | 10 | 0.3408 | 0.4000 |

## pit_within_3

| model | n | positives | log_loss | brier | auc | avg_precision | brier_skill |
|---|---|---|---|---|---|---|---|
| gbm_hazard | 16177 | 1507 | 0.2703 | 0.0758 | 0.7592 | 0.2923 | 0.1031 |
| tyre_age_logit | 16177 | 1507 | 0.2936 | 0.0810 | 0.6967 | 0.1966 | 0.0419 |
| base_rate | 16177 | 1507 | 0.3102 | 0.0845 | 0.4830 | 0.0872 | 0.0000 |

Calibration of `gbm_hazard` (predicted vs observed rate):

| bin | n | predicted | observed |
|---|---|---|---|
| (-0.001, 0.02] | 2108 | 0.0174 | 0.0323 |
| (0.02, 0.05] | 6697 | 0.0314 | 0.0336 |
| (0.05, 0.1] | 2804 | 0.0704 | 0.0938 |
| (0.1, 0.2] | 2321 | 0.1443 | 0.1357 |
| (0.2, 0.3] | 1333 | 0.2482 | 0.2078 |
| (0.3, 0.5] | 889 | 0.3605 | 0.3870 |
| (0.5, 1.0] | 25 | 0.5178 | 0.6000 |

## position_after_stop

| model | n | exact | within_1 | mae |
|---|---|---|---|---|
| rejoin_gbm | 533 | 0.5385 | 0.8630 | 0.7392 |
| gap_minus_pitloss | 533 | 0.4991 | 0.7917 | 0.9925 |
| no_change | 533 | 0.3265 | 0.5197 | 1.9662 |
