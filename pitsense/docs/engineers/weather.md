# Weather engineer

Rain in the next minutes, and when slicks or intermediates become the faster tyre. Code:
`src/pitsense/pitwall/engineers/weather.py`; measurements in `src/pitsense/bench/weather.py`
(`python -m pitsense weather-report`, task `rain_10min`). Requires `rules` (for the FIA "risk of rain").

## Values (race scope, `weather__<key>`)

| key | meaning |
|---|---|
| `rainfall`, `track_temp`, `air_temp`, `humidity` | latest feed readings. `Rainfall` is a 0/1 flag sampled about once a minute |
| `wet_running` | any running car on intermediates or wets |
| `rain_prob_10min` | probability the rain flag is up at some moment in the next 10 min (0-1; raining now counts) |
| `crossover` | `none`, `to_inters` (inters have become faster than slicks) or `to_slicks` (slicks faster than inters) |
| `crossover_f` | the same as 0 / 1 / -1 |
| `inters_vs_slicks_s` | wet-tyre lap time minus slick lap time on the shared laps, s (negative: inters faster). None while `crossover` is `none`; `inters_vs_slicks_f` is the same with 0.0 |
| `crossover_since` | session time the current call started (None if none) |
| `rain_now` | rain flag as 0 / 1 |
| `rain_minutes`, `minutes_since_rain` | length of the current rain run; time since the flag dropped (-1 = no rain yet) |
| `track_temp_trend`, `air_temp_trend`, `humidity_trend` | change over the last 10 min |
| `rc_rain_risk_f` | the FIA figure from `rules` (-1 before the first message) |
| `prior_rain_rate` | share of earlier races at this circuit with rain (shrunk to the global rate) |
| `wet_cars`, `dry_cars` | running cars on wet / dry tyres |

Alerts (level-triggered): `rain_onset_likely` (dry now, `rain_prob_10min` >= 0.35), `crossover_reached`
(`since` = when the call started). No benchmark `features` are declared: adding `rain_prob_10min`,
`rain_now`, `crossover_f`, `inters_vs_slicks_f` moved the core models slightly the wrong way on 2026
(pit_within_3 log loss 0.2584 to 0.2588, rejoin_gbm exact 0.527 to 0.516), so the columns are in every
row but the core models don't read them. Every value is a scalar and never NaN (sentinels where needed).

## How it works

**Nowcast.** A logistic regression over: flag now, length of the current rain run, recency of the last
rain, rain seen so far, 10-min trends of humidity / track / air temperature, humidity, and the circuit prior.
Training data are samples (every 2 min of the race) that `summarize_race` stores for each race, so
`ctx.past_races` supplies them and only races that ended before this one started are used. The circuit
prior inside each training race also uses only the races before that one. Before any history exists, a fixed
persistence rule is used. `summarize_race` reads the finished race's `WeatherData` stream file (an engineer
cannot read the event log, and the state keeps only the latest reading).

*The FIA "risk of rain" is not a model input.* Fed in as published it made held-out log loss worse on both
2025 (the selection year; sample log loss 0.123 with it vs 0.105 without) and 2026 (0.131 vs 0.102), and
taken at face value it scores below the base rate (table below). It is exposed as `rc_rain_risk_f` only.
Settings (C = 1, prior kept, rc dropped) were chosen on 2025, then scored on 2026.

**Crossover.** Over the last 4 laps, clean laps only (green throughout, not an in- or out-lap, not lap 1),
lap by lap: median lap time of cars on wet tyres minus that of cars on slicks, for laps where both groups
ran. The mean of the last two such laps (or one lap, if it differs by 1.5 s or more) must be at least
0.5 s: negative gives `to_inters`, positive `to_slicks`. The evidence comes from the earliest switchers,
since the others are still on the old tyre. With no mixed laps the previous call is held for 8 laps, then
drops to `none`. Cars are not corrected for car quality (no per-car baseline exists when a race starts wet).

## Measurements

Races 2018-2026 built (188). "Wet race" = the rain flag was up at some lap end, or wet tyres were used:
35 of 188 (2026: 3 of 15, Canadian and Italian with wet tyres but no flag, Dutch with the flag; 2025: 4;
2024: 4; overall 11 flag only, 9 tyres only, 15 both). Base rate of the label: 5.9% of lap-end rows. Each
test race is scored with a nowcast built only from earlier races. The wet-race subset is picked by what
happened in the race, which is how it is meant to be read: "given the weekend turned wet".

### 1. rain_10min (lap-end rows; log loss, lower is better; Brier skill against the base rate)

2026 (15 races, 3 wet):

| rows | model | log loss | Brier | Brier skill |
|---|---|---|---|---|
| all races (n=16323, 727 positive) | **nowcast** | **0.079** | 0.0244 | +0.43 |
| | gbm on weather columns | 0.124 | 0.0331 | +0.23 |
| | persistence | 0.161 | 0.0346 | +0.20 |
| | FIA risk taken as published | 0.184 | 0.0568 | -0.32 |
| | base rate | 0.186 | 0.0429 | 0 |
| wet races only (n=3440, 710 pos) | **nowcast** | **0.325** | 0.1106 | +0.41 |
| | gbm | 0.529 | 0.1495 | +0.20 |
| | FIA risk | 0.544 | 0.2001 | -0.07 |
| | base rate | 0.640 | 0.1863 | 0 |
| | persistence | 0.677 | 0.1580 | +0.15 |
| onset: dry now, wet races (n=3292, 562 pos) | **nowcast** | **0.337** | 0.1154 | +0.25 |
| | base rate | 0.540 | 0.1547 | 0 |

Only 3 wet 2026 races, and the Dutch one has nearly all the rain. Race bootstrap of nowcast minus base-rate
log loss on wet races: median -0.31, 90% interval [-0.57, -0.05] (3 races). More wet races, models retrained
per race:

| test year | races (wet) | all: nowcast / base rate | wet races: nowcast / persistence / base rate | onset in wet races: nowcast / base rate |
|---|---|---|---|---|
| 2026 | 15 (3) | 0.079 / 0.186 | 0.325 / 0.677 / 0.640 | 0.337 / 0.540 |
| 2025 | 24 (4) | 0.073 / 0.153 | 0.474 / 0.572 / 0.710 | 0.523 / 0.483 |
| 2024 | 24 (4) | 0.065 / 0.268 | 0.321 / 0.350 / 1.222 | 0.474 / 0.414 |

The 2025 wet-race interval against the base rate is [-0.56, +0.05] (includes zero). **Onset in wet races
(dry now) is the weak spot: the nowcast beats the base rate only in 2026; in 2025 and 2024 it is worse.**
Beating plain persistence in wet races is also thin in 2024 (0.321 vs 0.350; the gbm, 0.290, is best there).
AUC for the constant-per-race baselines means nothing and is left out here.

Calibration of the nowcast, 2026, all rows (predicted vs observed): up to 0.02: 0.008 vs 0.001 (14801 rows);
0.05-0.1: 0.078 vs 0.52 (246); 0.1-0.2: 0.144 vs 0.48 (483); 0.2-0.3: 0.237 vs 0.24 (230); 0.3-0.5: 0.38 vs
0.93 (160); above 0.5: 0.94 vs 1.00 (148). It is well calibrated at the extremes and under-confident in
between; the middle bins are the Dutch race's onset, so this rests on one race. 2025 had the opposite
problem (0.2-0.3 predicted, nothing observed). Full tables: `weather-report --test-years 2026 2025 2024`.

### 2. Crossover timing, wet races 2018-2026 (leader's laps)

Switch episodes: at least 3 cars changing between slicks and wet tyres the same way, consecutive switches no
more than 8 laps apart, red-flag stops excluded: 25 episodes in 13 races. Several are lap-1 changes or
changes under a neutralisation, not weather crossovers. `paid off`: first lap from which the first three
switchers' elapsed time since their stop is below the median of the other cars on two laps running (stop
included). `call`: start of the engineer's call in that direction if it was up around the switching;
`opposite`: the opposite call in that window.

Summary: 13 of 25 episodes were called; 5 of those 13 before the median switcher and most at the same lap
(median switch minus call = 0); the call came before `paid off` in 7 of the 7 episodes that have both. The 12
uncalled episodes mostly switch within one or two laps, often under a neutralisation, so no mixed clean laps
exist to compare. The engineer follows the field rather than leading it.

| race | dir | cars | first switch | call | opposite | paid off | median switch |
|---|---|---|---|---|---|---|---|
| 2018 German | to_inters | 9 | 43 | | 46 | | 50 |
| 2018 German | to_slicks | 7 | 46 | 46 | | | 55 |
| 2019 German | to_inters | 12 | 28 | 24 | 25 | 30 | 28 |
| 2019 German | to_slicks | 12 | 21 | 25 | 24 | 28 | 26 |
| 2019 German | to_slicks | 15 | 44 | | | 47 | 46 |
| 2020 Hungarian | to_slicks | 18 | 1 | | | | 3 |
| 2021 Emilia Romagna | to_slicks | 19 | 20 | 27 | 25 | 34 | 27 |
| 2021 Hungarian | to_slicks | 12 | 3 | | | 13 | 3 |
| 2021 Russian | to_inters | 18 | 46 | 49 | | 50 | 49 |
| 2022 Emilia Romagna | to_slicks | 18 | 16 | 18 | | 19 | 18 |
| 2022 Monaco | to_slicks | 16 | 17 | 21 | | | 21 |
| 2022 Singapore | to_slicks | 14 | 33 | 35 | 23 | | 34 |
| 2023 Monaco | to_inters | 20 | 51 | | 54 | | 54 |
| 2023 Dutch | to_inters | 15 | 1 | 3 | 9 | | 2 |
| 2023 Dutch | to_inters | 18 | 59 | | | 64 | 60 |
| 2023 Dutch | to_slicks | 15 | 9 | 9 | 3 | 12 | 10 |
| 2024 Canadian | to_slicks | 18 | 40 | | 30 | 56 | 44 |
| 2024 British | to_inters | 20 | 19 | | 21 | | 26 |
| 2024 British | to_slicks | 18 | 37 | 21 | | | 38 |
| 2025 Australian | to_inters | 15 | 44 | 46 | | 47 | 44 |
| 2025 Australian | to_slicks | 15 | 33 | | | 37 | 33 |
| 2025 British | to_inters | 6 | 9 | | 8 | 12 | 10 |
| 2025 British | to_slicks | 15 | 37 | | 40 | | 41 |
| 2025 Belgian | to_slicks | 20 | 11 | | | 13 | 12 |
| 2026 Canadian | to_slicks | 7 | 1 | 2 | | | 2 |

The 2024 British to_slicks "call" at lap 21 is an earlier call of the same direction that was still inside the
window, 16 laps before the first switcher; treat it as a timing miss. Calls in a direction with no switch
episode: 2021 Emilia Romagna, 2022 Singapore, 2023 Monaco, 2024 Canada. Canada 2024 is the useful case: cars
that fitted slicks early (laps 40-45) were slower than the inters for several laps (the engineer says
`to_inters`), and the switch only paid off at lap 56.

### 3. Leakcheck (`--cuts 2`, truncate and scramble)

2026 Dutch (rain flag), 2026 Miami (rain flag, 3 samples), 2026 Canadian (wet tyres): all PASS, 0 mismatches.
A stronger private check with the `Context` history attached (so the fitted nowcast is used, p up to 0.97) on
Dutch 2026, cuts at 35 / 60 / 85% of the race, truncate and scramble: 0 mismatches. The stock `leakcheck`
builds rows without history, so it does not exercise the fitted model.

## Known limits

- Few wet races (35 in 9 seasons, 3 in 2026). Onset timing in wet races is the weakest part, and the intervals are wide.
- `Rainfall` is a coarse flag that can stay up while the track dries (2019 German: up for the whole race).
- Crossover can only be seen once cars on both tyres complete clean green laps; mass switches under SC or red flag give no call.
- `summarize_race` needs the finished race's `WeatherData.jsonStream` (archive); a live recording without it adds no samples.
- `RaceState` keeps only the latest weather reading. A weather series in the state (data engineer) would let
  `summarize_race` work from `final` alone.
