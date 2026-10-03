# Pit-stop engineer

Role: pit-stop analyst. What does a stop cost right now, and where would the car come out?
Code: `src/pitsense/pitloss.py`, `pitwall/engineers/pitstop.py`; benchmark `bench/pitstop.py`.

## Values

Race level (`pitstop__<key>`):

| Key | Meaning | Unit |
|---|---|---|
| `loss_green`, `loss_sc`, `loss_vsc` | expected stop loss under each status: prior updated by this race's measured stops | s |
| `loss_now` | loss under the status in force (red flag: 0; whole field driving through the pit lane: stationary time + 3 s) | s |
| `loss_now_sd` | spread of one stop around `loss_now` | s |
| `loss_condition` | `green`, `sc`, `sc_ending`, `vsc`, `vsc_ending`, `pass_through`, `red` | str |
| `pass_through` | race control has the field driving through the pit lane behind the SC | bool |
| `phase_age_s` | time since this green/SC/VSC phase began | s |
| `loss_prior_green`, `loss_prior_source`, `loss_drift_s`, `lane_time_s` | the pre-race prior and where it came from (`circuit+drift`, `circuit`, `global`, `default`) | s / str |
| `stationary_field_s`, `stops_measured`, `cars_in_pit_lane` | field stationary time; measured stops so far; cars in the lane now | s, count |

Car level:

| Key | Meaning | Unit |
|---|---|---|
| `rejoin_if_box_now` | expected position after a stop now (mode of the distribution of cars that get by) | position |
| `rejoin_delta` | places lost: `rejoin_if_box_now - position` | places |
| `cars_within_loss` | cars behind within one pit loss | count |
| `cars_within_stopping` | of those, expected to stop in the same window | count |
| `passers_expected` | expected cars that get by | count |
| `loss_if_box_now` | `loss_now` + the team's stationary time vs the field + penalty to serve | s |
| `stationary_s` | the team's expected stationary time (past races + this one) | s |
| `margin_s` | gap behind us beyond the loss to the first car that stays behind | s |
| `rejoin_gap_ahead_s`, `rejoin_gap_behind_s` | gaps to the cars we would come out between | s |
| `penalty_s` | time penalty served at the stop (from `rules`) | s |

All as of now: `state`, `memory`, `ctx.past_races`, and the `rules`/`rivals` values.

## How it works

**Measuring a stop** (`measure_stop`). Green throughout: v0.1's lap-number method
(in-lap + out-lap minus the field median for the same laps, minus twice the car's pace offset).
When the stop touches an SC/VSC the lap-number method is biased: cars reach a lap number at different
times, and a status change happens at one moment for everyone. So each reference car's two laps start at its
line crossing nearest to the moment the stopper began its in-lap, only cars within 15 s on the road count,
and the medians of cars ahead and behind are averaged. Every input must be published by out-lap end + 30 s
(`available_at`), so the value never depends on when it is computed (tested). Status in the pit lane
(`lane_condition`) decides the class: green / sc / vsc / mixed (status changed in the lane; kept out of priors) / red (skipped).

**Prior** (`loss_prior`): the circuit's last green loss + a season drift (how far recent races landed from
their own circuits' last visit), else the median of recent races. SC/VSC are a share of green, blended with
this circuit's own SC/VSC stops, then with this race's stops (`current_losses`).

**Rejoin** (`expected_rejoin`): each car behind gets by if it does not stop in the same window
and its gap is under the loss (normal, `loss_now_sd`). Co-stop chance comes from counts in past races by
status, already-stopped, tyre age and SC age (`CoStopRates`), or the rivals' `pit_prob_1` when present. The
answer is the mode of the resulting (Poisson-binomial) distribution; mode beat median and mean on 2025 exact
match, and the mean count is well calibrated (2025: 4.25 expected vs 4.34 actual passers under green).
When the whole field drives through the pit lane nobody gets by while a car is stationary,
and nearly every pit entry there is a drive-through.

## Benchmark (tuned on 2025, scored once on 2026)

`position_after_stop`, 2026, 533 pit entries (models in `registry.TASK_MODELS`):

| Model | exact | within 1 | MAE |
|---|---|---|---|
| `pitstop_rule` (engineer value; pass-through: no change) | **0.629** | **0.901** | **0.567** |
| `pitstop_gbm` (rule + correction from engineer values) | 0.608 | 0.897 | 0.584 |
| `rejoin_gbm` (previous best) | 0.539 | 0.863 | 0.739 |
| `gap_minus_pitloss` | 0.499 | 0.792 | 0.993 |
| `no_change` | 0.327 | 0.520 | 1.966 |

Exact match per race, `pitstop_rule` vs `rejoin_gbm`:

| Race | rule | rejoin_gbm | Race | rule | rejoin_gbm |
|---|---|---|---|---|---|
| Australia | .53 | .41 | Austria | .66 | .63 |
| China | .32 | .37 | Britain | .75 | .61 |
| Japan | .59 | .62 | Belgium | .68 | .46 |
| Miami | .68 | .50 | Hungary | .71 | .64 |
| Canada | .44 | .35 | Netherlands | .64 | .45 |
| Monaco | .54 | .54 | Italy | .67 | .67 |
| Barcelona | .65 | .60 | Spain | .60 | .64 |
| Azerbaijan | **.86** | .50 | | | |

Azerbaijan (gap rule .22, no change .86): the lap-36 pass-through (all 16 entries) and the lap-30/31
mass stop under SC are now handled; the rule scores .86. Worse than `rejoin_gbm` in China, Japan and Spain (by 3-5 points, within noise at 19-29 stops per race).

`pit_loss` (regression, 2026, 418 stops with a typical measured loss), predicted at pit entry:

| Split | n | `v01_prior_shrink` MAE | `pitstop_loss` MAE | `pitstop_loss_gbm` MAE |
|---|---|---|---|---|
| all | 418 | 4.08 | **3.38** | 3.39 |
| green | 293 | 3.03 | **2.77** | **2.77** |
| sc | 39 | 6.76 | 5.87 | **5.76** |
| vsc | 75 | 6.61 | **4.07** | 4.26 |

Target: `y_pit_loss`, the time-aligned measure on the finished race (stops outside the plausible range, 5-45 s green and 0-40 s SC/VSC, are dropped; "mixed" stops stay in "all").
The floor is high: a race-median oracle scores about 2.1 (2025) to 2.4 (2026) MAE on green stops, since slow stops cannot be seen at entry.
SC loss varies from 2 s to 25 s between stops and is the main open problem.

Core leaderboard (`pit_within_*`, `position_after_stop` base models) is unchanged: v0.1 `measure_stops`,
the base features and history priors were not touched. `bench build` time 25 s -> 33 s.

## Limits

- SC/VSC loss is noisy; lap times under SC are too sparse live to model the field's slowdown (tried: fewer than 10 stops had 3 whole SC laps).
- `penalty_s_pending` comes from `rules` (0 until that engineer fills it).
- Pass-through is detected from race-control messages; if the wording changes, `pass_through` reads False.
- Pit-entry rows under pass-through are mostly drive-throughs; the benchmark model assumes so, the engineer's `rejoin_if_box_now` still answers "if we really stopped".
