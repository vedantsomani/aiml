# PitSense-Bench leaderboard

Test races: 15 (2026), each scored with models trained only on races that finished before it started. Decision points: 43,123 from 39 races. Feature version v0.1.

## pit_within_1

| model | n | positives | log_loss | brier | auc | avg_precision | brier_skill |
|---|---|---|---|---|---|---|---|
| gbm_hazard | 16177 | 498 | 0.1181 | 0.0281 | 0.7990 | 0.1375 | 0.0568 |
| tyre_age_logit | 16177 | 498 | 0.1311 | 0.0294 | 0.6990 | 0.0714 | 0.0143 |
| base_rate | 16177 | 498 | 0.1375 | 0.0298 | 0.4831 | 0.0287 | 0.0000 |

Calibration of `gbm_hazard` (predicted vs observed rate):

| bin | n | predicted | observed |
|---|---|---|---|
| (-0.001, 0.02] | 10070 | 0.0100 | 0.0087 |
| (0.02, 0.05] | 3120 | 0.0321 | 0.0378 |
| (0.05, 0.1] | 1814 | 0.0724 | 0.0728 |
| (0.1, 0.2] | 1067 | 0.1318 | 0.1218 |
| (0.2, 0.3] | 95 | 0.2367 | 0.2632 |
| (0.3, 0.5] | 11 | 0.3299 | 0.4545 |

## pit_within_3

| model | n | positives | log_loss | brier | auc | avg_precision | brier_skill |
|---|---|---|---|---|---|---|---|
| gbm_hazard | 16177 | 1507 | 0.2706 | 0.0760 | 0.7594 | 0.2868 | 0.1010 |
| tyre_age_logit | 16177 | 1507 | 0.2934 | 0.0810 | 0.6969 | 0.1973 | 0.0425 |
| base_rate | 16177 | 1507 | 0.3102 | 0.0845 | 0.4830 | 0.0872 | 0.0000 |

Calibration of `gbm_hazard` (predicted vs observed rate):

| bin | n | predicted | observed |
|---|---|---|---|
| (-0.001, 0.02] | 2136 | 0.0173 | 0.0314 |
| (0.02, 0.05] | 6722 | 0.0313 | 0.0324 |
| (0.05, 0.1] | 2776 | 0.0706 | 0.0987 |
| (0.1, 0.2] | 2286 | 0.1442 | 0.1365 |
| (0.2, 0.3] | 1367 | 0.2485 | 0.2165 |
| (0.3, 0.5] | 866 | 0.3626 | 0.3776 |
| (0.5, 1.0] | 24 | 0.5188 | 0.5417 |

## position_after_stop

| model | n | exact | within_1 | mae |
|---|---|---|---|---|
| rejoin_gbm | 533 | 0.5385 | 0.8630 | 0.7392 |
| gap_minus_pitloss | 533 | 0.4991 | 0.7917 | 0.9925 |
| no_change | 533 | 0.3265 | 0.5197 | 1.9662 |
