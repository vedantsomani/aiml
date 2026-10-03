# Rival strategist (`rivals`)

Code: `src/pitsense/pitwall/engineers/rivals.py`, benchmark: `src/pitsense/bench/rivals.py`,
tests: `tests/test_rivals.py`. `requires = ("tyre", "pitstop")`.

## Keys (`rivals__<key>`, per car)

| Key | Meaning | Unit |
|---|---|---|
| `ahead`, `behind`, `gap_ahead`, `gap_behind` | neighbours in the running order and the gaps to them | car no., s |
| `pit_prob_1`, `pit_prob_3`, `pit_prob_5` | chance the car stops within 1 / 3 / 5 laps | 0-1 |
| `undercut_threat` | chance the car behind stops first (within 5 laps) and ends ahead of this car | 0-1 |
| `undercut_chance` | the same for this car over the car in front | 0-1 |
| `uc_margin_threat`, `uc_margin_chance` | fresh-tyre gain over the defender's cover delay minus the gap (positive = the attacker ends ahead) | s |
| `uc_first_threat`, `uc_first_chance` | chance the attacker stops first within 5 laps | 0-1 |
| `in_pit_window` | on a set at least 0.7x the typical stint (and more than 3 laps to go) | bool |
| `typical_stint_laps` | median finished stint on this compound, this circuit (all circuits if few) | laps |
| `team_cover_rate` | how often the team's cars reacted within 2 laps to a rival ahead stopping | 0-1 |
| `prior_pit_1`, `prior_pit_3`, `lp_k` | history-only stop probability (and its logit) | 0-1 |
| `sc`, `vsc`, `ahead_pit`, `behind_pit`, `mate_pit`, `cover`, `cliff`, `must`, `stuck`, `threat`, `stops`, `rem`, `ratio` | the in-race signals behind `pit_prob_k` | see below |

Alerts: `undercut_threat` (warn at >= 35 %, critical at >= 60 %), level-triggered, for the car being threatened.

## Method

**Pre-race knowledge.** `summarize_race` stores every finished stint by compound (`[laps, ended_with_stop]`;
a stint still running at the flag is censored) and, per team, how often the car within two places behind a
rival that stopped on green (not within 8 laps of the flag) stopped within 2 laps (`[opportunities, covers]`).
`__init__` reads these from `ctx.past_races` only.

**`pit_prob_k`.** Layer 1 is a survival hazard: per-lap stop hazards `h_j` (stints that ended on lap j / stints
that reached it) at this circuit, shrunk to all circuits, shrunk to a default stint life, compounded over k laps
given the laps already run. Layer 2 is a logistic correction `PIT_B[k]` on the prior's logit plus signals:
SC / VSC, a rival within two places stopped in the last 2 laps (ahead / behind), a teammate stopped, tyre
`cliff_risk`, laps left (`rem`, `late`), age over the typical stint (`ratio`), must-stop, stuck behind (<1.2 s),
threatened (<1.5 s behind), stops made, and `cover` = team cover rate x ahead stopped. Coefficients are fitted
on 2025 labels only, with the history prior computed leave-one-race-out inside 2025 (so its strength matches a
season of history, not the thin early-2025 history).

**Undercut.** If B (behind A) stops it pays `loss_now` but A covers ~2 laps later (paying the same), so B ends
ahead if the fresh-tyre gain over those 2 laps beats the gap:
`margin = sum_{j=1..2}(pace_A + deg_A*j - fresh_B) - gap`, `fresh_B` = mean of B's `fresh_*_s`. Probability =
P(B stops first within 5 laps) x sigmoid(a + b*margin + c*A's team cover rate); P(first) comes from the two cars'
`pit_prob_5` (both stopping: split by their relative odds), calibrated on 2025 (`FIRST_B`, `UC_B`).

## Benchmark

* `undercut_5` (`y_uc`): the car directly behind stops within 5 laps, before this car, and is ahead of it once both
  have stopped (positions 3 laps after the later stop; if this car never stops, 3 laps after the chaser's). Rows:
  green lap ends with a car behind. NaN if positions can't be compared. Also `y_uc_chance` (this car over the one ahead).
* `pit_within_k` gets extra models: `rivals_pit` (the engineer's own `pit_prob_k`), `rivals_prior` (history only),
  `rivals_logit` / `rivals_logit_noteam` (logistic on prior + signals, trained on earlier races, with / without team habit),
  `rivals_gbm` / `rivals_gbm_noteam` (gbm_hazard's inputs + rivals keys, with / without team habit) and `rivals_blend`
  (logit average of `rivals_pit` and `rivals_gbm`).

### Results, 2026 (each race scored by models trained on earlier races only)

pit_within_1 (log loss, lower is better; n = 16177, 498 stops):

| model | log loss | AUC |
|---|---|---|
| rivals_gbm | 0.1181 | 0.807 |
| gbm_hazard (core reference) | 0.1181 | 0.798 |
| rivals_gbm_noteam | 0.1185 | 0.805 |
| rivals_blend | 0.1228 | 0.779 |
| rivals_logit | 0.1297 | 0.713 |
| rivals_pit (engineer value) | 0.1305 | 0.718 |
| tyre_age_logit | 0.1311 | 0.699 |
| base_rate | 0.1375 | 0.483 |

pit_within_3 (n = 16177, 1507 stops):

| model | log loss | AUC |
|---|---|---|
| gbm_hazard (core reference) | 0.2685 | 0.763 |
| rivals_gbm | 0.2734 | 0.760 |
| rivals_gbm_noteam | 0.2736 | 0.761 |
| rivals_blend | 0.2757 | 0.760 |
| rivals_logit | 0.2825 | 0.736 |
| rivals_pit (engineer value) | 0.2838 | 0.740 |
| base_rate | 0.3102 | 0.483 |

undercut_5 (n = 13932, 163 positives, base rate 1.2 %):

| model | log loss | Brier skill | AUC |
|---|---|---|---|
| undercut_gbm | 0.0564 | 0.013 | 0.818 |
| rivals_threat (engineer value) | 0.0569 | 0.026 | 0.813 |
| gap_logit (gap and tyre age only) | 0.0614 | -0.001 | 0.768 |
| base_rate | 0.0661 | 0 | 0.539 |

**Honest summary.** The engineer's `undercut_threat` is a real signal (AUC 0.81, log loss 14 % under the base rate,
well ahead of a gap-and-age model). For stop timing, the learned models only *tie* gbm_hazard at k = 1 and trail it
at k = 3 (0.2734 vs 0.2685): the rivals keys add nothing the tyre and race features don't already hold. The engineer's own
`pit_prob_k` is calibrated on 2025 and generalises worse than the boosted model trained on 2026's earlier races.
The history prior alone is no better than the base rate (stint lengths changed between seasons; one year of
history is thin: more seasons should help).

### Team-habit ablation

| task | with team habit | without | change |
|---|---|---|---|
| pit_within_1, logistic | 0.1297 | 0.1298 | -0.0001 |
| pit_within_3, logistic | 0.2825 | 0.2825 | 0 |
| pit_within_1, boosted | 0.1181 | 0.1185 | -0.0004 |
| pit_within_3, boosted | 0.2734 | 0.2736 | -0.0002 |
| undercut_5, success layer (refit on 2025) | 0.0578 | 0.0579 | -0.0001 |

Team habits add essentially nothing yet. Only ~600 cover opportunities across 2025-26 (about 50 per team), so team
rates are mostly the field average after shrinkage. Re-test after 2018-2024 are merged.

### Notes

* No keys are declared as benchmark `features`: on 2025 none of them improved `gbm_hazard` (k = 3 got worse), so the
  core leaderboard is unchanged.
* Tuning used 2025 only (`--test-year 2025 --min-train 10`); 2026 was scored after tuning, but 2026 numbers were seen
  several times during development (the final design choices, fewer declared features and gbm_hazard-sized boosters,
  were made after the 2025 comparison `data/scratch/decl.py`).
* Leak check (`pitsense leakcheck --year 2026 --cuts 2`): hungary and australian PASS. The leak check builds its
  context without race history, so the history path (`ctx.past_races`) is protected by the HistoryStore cutoff, not by it.
