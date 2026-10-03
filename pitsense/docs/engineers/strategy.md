# Strategy engineer and head of strategy (`strategy`, `head`)

Code: `src/pitsense/pitwall/engineers/strategy/` (`priors.py` what past races teach, `sim.py` futures and
tyre/lap model, `run.py` field simulation and plan evaluation, `analysis.py` field building, candidate plans,
ranking, Plan B, `engineer.py` values and caching), `pitwall/engineers/head.py` (the calls), benchmark
`src/pitsense/bench/strategy.py`, tests `tests/test_strategy.py`.

## What it does

For the focus cars (`ctx.team.focus`; every running car when no team is set, in a lighter mode above 4 cars)
it simulates the rest of the race lap by lap and ranks stop plans on the same simulated futures.

**The futures** (288 per decision, seeded from the race, not from the clock). Every running car gets a
sampled strategy: its first stop from the rivals' `pit_prob_1/3/5` (the models' `pit_prob_1/3` replace them
when present), later stops and compounds from stint lengths in past races at this circuit (`summarize_race`
stores them, shrunk to all circuits). Lap time is the tyre engineer's `pace_s`, `deg_s_per_lap`,
`fuel_s_per_lap`, `fresh_*_s` and field degradation, plus a tyre cliff (typical stint x 1.7, scaled up by
`cliff_risk` for the current set), a per-car pace uncertainty and lap noise. A stop costs the pitstop
engineer's `loss_green/sc/vsc` (by the status on the in-lap) with `loss_now_sd`, the team's stationary offset,
and 6 s of rejoin/traffic cost; penalties are added at the flag. A car arriving before a slower one must pass
(chance by pace advantage per circuit, learned from adjacent cars within 1 s) or is held; safety cars and VSCs
come with a per-lap hazard by circuit (circuit history shrunk to all circuits), bunch the field behind the SC,
make stops cheap, and pull cars due to stop into the pits. A neutralisation already in force
(`rules__sc_phase`) is simulated as such. Retirements are random (small).

**Plans.** For the focus car: no stop, 1 stop (every lap offset 0-58 in a grid x 3 compounds), 2 stops
(grid of first stops and gaps x 9 compound pairs), a few 3-stop plans. Legality: two dry compounds
(`must_stop`), `reg_min_stops`, stint limit (`stint_laps_left`, `reg_max_stint_laps`), stints no longer than
1.35x typical tyre life. Screening on 96 futures, then the best, the best for each of the next 8 stop laps
and each stop count are re-run on all 288. The focus car is played against the simulated opponents'
trajectories, so a plan costs a sorted search per lap rather than a field simulation (about 0.1 s per car for
about 250 plans). Utility = smoothed finishing position (each rival counts by `sigmoid(margin/4 s)`) + 0.001 per
second of race time + a small penalty for stopping far from the typical timing (`lam_prior` 0.02 places per
lap); expected finishing position and points are reported on the exact positions.

**Plan B.** 160 futures with a safety car (70 %) or VSC starting in the next 0-4 laps: plan A unchanged versus
stopping on that lap for each compound (plus a second stop when the stint is too long). Plan B is the best of
those with the trigger "SC/VSC within 5 laps: box that lap for X"; `strategy__sc_gain` is the places gained.
When a neutralisation is already out, Plan B is the best alternative plan.

**Head of strategy** (`decide`). Plan A = best plan, unless last lap's target stop lap is within `tol_keep`
(0.05) of the best utility (hysteresis). First stop this lap: `BOX` (`PREPARE_BOX` if the pit lane is closed);
within 2 laps: `PREPARE_BOX`; none due but a safety car would be worth >= 0.6 places: `BOX_IF_SC`; otherwise
`STAY_OUT`. `NO_CALL` with the reason when no plan can be made: before lap 2, wet race (`race_dry` false or
`weather__wet_running`) since only dry tyres are modelled. Confidence is the paired-future probability that
the chosen timing beats the best alternative (0.5-0.97), scaled down 20 % for rain risk / `rain_now` /
crossover and 15 % before lap 8. Reasons, most important first: safety car phase and its cheaper stop, closed
pit lane, must-stop, penalty, tyre cliff, undercut threat/chance, the numbers behind the plan, rejoin position,
expected finish, rain. Weather keys are read with `.get()`; missing values change nothing.

Refresh and determinism: a car's plan is recomputed when it completes a lap, the track status or pit lane
changes, or its `must_stop` / penalty changes; otherwise the cached plan stands. The random numbers are drawn for
every driver slot and every race lap from a seed of (race, kind), then cut to the laps still to run, so lap after
lap the same futures are used and calls move on evidence, not noise. The head's hysteresis is the one place a call
depends on the call before it (live: the previous lap's target).

## Keys

Per focus car (`strategy__<key>`): `plan_a`, `plan_b` (text "L33 MEDIUM, L54 SOFT"), `plan_a_pos`,
`plan_a_pts`, `plan_b_pos`, `next_stop_lap`, `now_cost` (places lost by stopping this lap vs plan A), `sc_gain`,
`ok`, `why`, `sim_ms`. Race: `sc_prob_5`, `vsc_prob_5`, `strategy_history_races`. Non-focus cars: no keys.

## Benchmark (tuned on 2025, 2026 scored once)

Tuned on 2025 (one pass over the grid of knobs): level / lap noise, cliff multiple and slope, retirement rate
(forecast RPS); stop cost and `lam_prior` (calls). Command: `pitsense bench run --tasks strategy_finish
strategy_nextstop strategy_calls --test-year 2026 --min-train 1`; the first run replays every race (about 10
minutes at `--jobs 3`), results cached in `data/scratch`.

**Finishing position at 25 / 50 / 75 % distance**, finishers only (RPS over positions, lower is better):

| 2026 | n | RPS | MAE | MAE@25 | MAE@50 | MAE@75 |
|---|---|---|---|---|---|---|
| sim | 771 | **0.0523** | 1.703 | **2.114** | 1.703 | 1.292 |
| current position | 771 | 0.0720 | **1.656** | 2.257 | **1.642** | **1.070** |
| pace extrapolation | 771 | 0.0769 | 1.768 | 2.241 | 1.864 | 1.198 |

(2025, for reference: RPS 0.0633 / 0.0846 / 0.0914; MAE 2.08 / 1.95 / 2.10.) The simulation is clearly the
better probabilistic forecast (RPS -27 % vs current position) but its expected position is not closer than the
current position on average (MAE), only at 25 % distance: the expectation of a wide distribution sits between
positions, while the field rarely changes in the last quarter. P(top 10) is calibrated on 2026 only at the
level of the first bin so far (see the report; the first bin is 0.4 % predicted vs 2.3 % observed).

**Next stop** (laps from the anchor to the real in-lap, cars that stop later):

| 2026 | n | MAE laps | within 2 laps | bias |
|---|---|---|---|---|
| sim (median of simulated first stops) | 439 | **6.45** | 0.260 | 0.00 |
| rivals hazard (cumulative `pit_prob_1/3/5`, then typical stint) | 439 | 7.09 | **0.287** | +0.44 |

Better on average error, worse on the share within 2 laps. The 2025 bias was -2.9 laps (simulated stops too
early); not corrected.

**Calls vs reality** (top-5 finishers' real stops, calls on every second lap, +-2 laps, red-flag stops
excluded; BOX / PREPARE_BOX are box calls):

| | races | box calls | real stops | precision | recall | BOX with no stop in L..L+2 | STAY_OUT right |
|---|---|---|---|---|---|---|---|
| 2025 (tuning) | 24 | 245 | 218 | 0.339 | 0.317 | 0.75 | 0.909 |
| 2026 | 15 | 147 | 128 | 0.306 | 0.305 | 0.876 | 0.930 |

`pitsense shadow-score` on the replay call log of the Hungarian GP (top 5, k = 2): 17 box calls, precision
0.353 (BOX 4/11, PREPARE_BOX 2/6), STAY_OUT accuracy 0.933; recall is 0.106 there because the log covers 5 cars
and the CLI counts every car's stops (use `--cars`).

Honest reading: the head is weak at timing a stop to within 2 laps. About a third of box calls land
and about a third of the real stops are covered. It is right that a stop is not due (STAY_OUT 93 %), but it
calls BOX on the first lap the plan says now and the plan's best lap moves with every lap's new information.
Real stop timing depends on things the engineers do not see (tyre temperature, the team's own targets, traffic
planning).

## Speed

Two focus cars on one snapshot: 0.6 s (288 futures, about 250 plans each) on this machine; per car 0.2 s once
the field is built, with the second lap's cache hit costing nothing. Calls over a whole race at every second lap
for five cars: about 0.19 s per call. A snapshot with no focus car and 20 running cars uses the lighter mode
(96 futures) and is about 6 s per lap.

## Leak test

`pitsense strategy-leakcheck --year 2026 --race hungary --cuts 3` builds the calls (plans, reasons, confidence)
at random cuts twice, from the full log and from `log.until(t)`: identical (Hungary, Monaco; also in
`tests/test_strategy.py` on the synthetic race). `pitsense leakcheck` does not cover `strategy` (slow engineers
are not in the benchmark rows).

## Open issues

* Timing accuracy of box calls (above). Ideas: model the team's real decision process (track position first),
  use the models bundle's `pit_prob_1` in the focus car's own utility, correct the early bias seen in 2025.
* Wet and mixed races produce `NO_CALL`: no intermediate/wet tyre model and no crossover plans.
* One safety car and one VSC at most per simulated future; red flags are treated like a safety car.
* Lapped cars and blue flags are ignored; a car in the pit lane now is treated as on its old tyres.
* The calls refresh per lap, so a mid-lap change in gaps is not seen until the next lap or status change.
