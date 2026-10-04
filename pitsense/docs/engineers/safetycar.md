# Safety-car engineer

Warns "SC / VSC likely in the next 1-2 laps" from what has been published so far. Code:
`src/pitsense/pitwall/engineers/safetycar.py` (engineer, `Tracker`, models), labels / task / report in
`src/pitsense/bench/safetycar.py` (`python -m pitsense sc-report`). Tests: `tests/test_safetycar.py`.

## Signals (all as of `state.t`)

| signal | how it is read |
|---|---|
| yellow / double-yellow sectors | race-control flags per sector; opened by the flag, closed by "CLEAR IN TRACK SECTOR n" / "TRACK CLEAR", treated as cleared after 7 min with no message |
| track status 2 duration | `state.track_status_since` |
| "car stopped", off track / spun, recovery vehicle / marshals / medical car / debris, incidents noted or under investigation | message text (the wording changes between seasons; "car stopped" and "off track and continued" only exist in 2018-24 feeds, 2025-26 mostly send marshals / recovery vehicle) |
| big lap-time loss | a car's published lap time against the median of its last 3-5 clean laps (no in / out laps) |
| lap 1 chaos | yellow / incident messages tagged lap 1, active through lap 4 |
| restart | seconds since the last SC / VSC ended |
| wet running | share of recent laps on intermediates / wets |
| circuit history | SC and VSC starts per lap at this circuit in `ctx.past_races`, shrunk (120 pseudo-laps) to all races |
| telemetry / position feeds | when loaded: cars at < 8 km/h for 6 s (not in the pit), cars off track. Exposed, named in `sc_reason`, but do **not** move the probability (below) |

## Model

Two L2 logistic models (SC, VSC; 18 scaled signals, L2 = 5) are fitted at the start of each race on the lap-end
samples that `summarize_race` stored for every earlier race (`ctx.past_races`, so only races that ended
before this one started). A sample is one leader lap end with no neutralisation on; label = an SC (status 4) /
VSC (status 6) starts before the leader completes two more laps. The circuit's log-odds shift
(circuit 2-lap probability against the all-circuit one, weight 0.25) is added. With fewer than 25 earlier
positives a fixed hand-set weighting is used (no history, e.g. the leak check). Settings (L2, circuit weight, shrink,
alert level) were chosen on 2025; the plateau is flat (L2 5-10, weight 0-0.5 differ by < 0.002 log loss).
Fitting is deterministic (Newton steps, no random draws).

## Values (race scope, `safetycar__<key>`)

| key | meaning |
|---|---|
| `sc_prob_2laps`, `vsc_prob_2laps` | probability an SC / a VSC starts within the next 2 leader laps, 0-1. 0 while an SC / VSC / red flag is on |
| `neutral_prob_2laps` | either; `neutral_prob_base` is the same with no live signal (circuit and race stage only) |
| `sc_reason` | text, e.g. `double yellow, car stopped`; `circuit history`, `none`, or `safety car out` / `VSC out` / `red flag` |
| `sc_warn` | alert condition: `neutral_prob_2laps >= 0.08` and at least 1.5x the base |
| `dy_sectors`, `y_sectors`, `yellow_age_s`, `stopped_msgs`, `incident_msgs`, `vehicle_msgs`, `loss_max_s`, `lap1_chaos` | the signals, as counts / seconds |
| `circuit_sc_rate`, `circuit_vsc_rate`, `history_races` | circuit prior |
| `feed_present`, `feed_stopped`, `feed_off_track` | feeds (0 without them) |

Alert `sc_likely` (level-triggered, `since` = start of the oldest active yellow or the latest stop / incident message).
No NaN anywhere (sentinels 0). `features` declares nothing: the probabilities did not help the core pit tasks.

## Benchmark task `sc_within_2`

Lap-end rows, one per race lap (the first car across the line), label `y_sc_within_2` = SC or VSC starts after the row and
before the leader has done two more laps; rows with an SC / VSC / red flag already on are left out. Models: `circuit_base_rate`
(reference: circuit share of positive laps in earlier races, shrunk), `base_rate`, `engineer` (the shipped probability),
`gbm_signals` (boosting on the same signals over earlier benchmark rows). Trained / fitted only on races that ended before
the test race. Selection year 2025, test year 2026.

### Results (lap-end task; 2025 = selection, 2026 = test)

Log loss / Brier / AUC (Brier skill is against `circuit_base_rate`; 46 positive laps of 1343 in 2025, 55 of 853 in 2026):

| model | 2025 log loss | Brier | AUC | 2026 log loss | Brier | AUC | 2026 Brier skill |
|---|---|---|---|---|---|---|---|
| circuit_base_rate (reference) | 0.1515 | 0.0334 | 0.533 | 0.2541 | 0.0612 | 0.543 | 0 |
| base_rate | 0.1494 | 0.0331 | 0.262 | 0.2491 | 0.0612 | 0.483 | 0.001 |
| **engineer** | 0.1389 | 0.0313 | 0.622 | 0.2349 | 0.0577 | 0.625 | 0.057 |
| gbm_signals | 0.1415 | 0.0319 | 0.641 | 0.2282 | 0.0553 | 0.610 | 0.097 |

Calibration of the engineer, 2026 (predicted / observed): 0.02-0.05: 0.029 / 0.057 (785 laps); 0.05-0.1: 0.060 / 0.035 (58);
0.1-0.2: 0.13 / 0.50 (2); 0.2-0.3: 0.26 / 1.0 (4); 0.3-0.5: 0.38 / 0.75 (4). Low bins are under-predicted in 2026 (more
VSCs: 38 VSC laps against 22 in 2025, AUC of the VSC probability 0.49 there); the few high bins are right. Per kind AUC
(2026): SC 0.65, VSC 0.49. Skill is real but modest: most deployments (collisions, breakdowns) come with no warning before the
lap line, and the lap-end snapshot misses signals that arrive later in the lap.

### Lead time and false alarms (alert threshold 0.08, chosen on 2025; "warned" = warning up before the deployment and within 2 laps)

The pit wall asks every update, not only at lap ends. The report replays each race and asks the engineer every 10 s
(`sc-report --continuous`; label = same rule, sample = a moment):

| | races | deployments | warned | median lead (s) | min lead (s) | false alarms / race | log loss | AUC |
|---|---|---|---|---|---|---|---|---|
| 2025 engineer alert | 24 | 31 | 23 (74 %) | 28.6 | 2.2 | 0.75 | 0.1271 | 0.704 |
| 2025 rule "status 2" | 24 | 31 | 23 (74 %) | 27.3 | 2.2 | 0.75 | | |
| 2026 engineer alert | 15 | 30 | 20 (67 %) | 20.5 | 6.8 | 1.53 | 0.1978 | 0.556 |
| 2026 rule "status 2" | 15 | 30 | 20 (67 %) | 20.5 | 6.8 | 1.53 | | |

A false alarm is a run of consecutive warnings none of which is followed by a deployment within 2 laps; a warning is up in
about 1-2.5 % of samples. At lap ends only (task rows): 2025 6 of 26 deployments warned (median lead 40 s, 0.21 false alarms
per race), 2026 7 of 28 (15 s, 0.13). Honest reading: the warning is essentially "a yellow flag is out": the engineer
matches the plain status-2 rule on warnings and adds probabilities (calibrated, with the circuit history and lap-1 / wet
context) rather than a better trigger. A quarter to a third of deployments give no warning at all; the lead is 20-30 s, enough
to prepare a call, not to run a new plan. The 2026 test number includes the earlier-tuned threshold only: the 2026 report was
first looked at once with an earlier build (feeds multiplier on, alert on the bare probability) and again after the final fix;
the model, L2 and circuit weight were not changed between them.

### Feeds (2025-26, telemetry and positions loaded)

Cars below 8 km/h for 6 s outside the pit: 3150 samples in 2025 (27 %: cars parked or crawling), positive share 0.044 against
0.029 overall, nothing for 2+ cars. Multiplying the odds by 3 raised 2025 log loss 0.127 to 0.133 and the warnings to 28 % of
samples, so the multiplier is 1.0: `feed_stopped` / `feed_off_track` are exposed and named in `sc_reason`, no more. With the
feeds loaded the 2025 numbers are unchanged (log loss 0.1263, 74 % warned) and 2026 is 0.2052 / 67 %. A usable stopped-car
detector needs the track map (stopped on the racing line, not in the run-off), which is open work.

## Effect on the strategy calls

Hook (strategy/, 3 small edits, `SETTINGS["use_sc_prob"]` turns it off): `analysis.near_rates(view)` reads
`safetycar__sc_prob_2laps` / `vsc_prob_2laps` through the view (None if the engineer is not on the wall) and
`sim.Draws._status` uses `1-(1-p)^(1/2)` as the per-lap start rate for the next 2 laps instead of the circuit rate
(later laps keep the circuit rate); `head.decide(..., sc_prob=)` lowers the BOX_IF_SC gain bar to 0.3 places while
`neutral_prob_2laps >= 0.08`.

`strategy_calls` replay (top-5 cars, every 2nd lap, +-2 laps, models on, `use_sc_prob` off then on):

| top 5 | box+prep calls | precision | BOX | PREPARE_BOX | recall | STAY_OUT right | races with a different call log |
|---|---|---|---|---|---|---|---|
| 2025 off | 200 | 0.495 | 30/59 | 69/141 | 0.349 | 0.967 | |
| 2025 on | 201 | 0.493 | 31/60 | 68/141 | 0.349 | 0.967 | 14 of 24 |
| 2026 off | 100 | 0.420 | 11/20 | 31/80 | 0.266 | 0.959 | |
| 2026 on | 97 | 0.423 | 10/18 | 31/79 | 0.258 | 0.951 | 7 of 15 |

Effect on calls: within noise (a handful of calls move in about half the races; precision +-0.003, recall 0 to -0.008). The
simulator's hazard only matters in the few laps around a warning, and a stop call needs the pit-probability model to agree.
Latency about 0.7-0.9 s per call (unchanged). **BOX_IF_SC is not exercised by this bench**: with five focus cars the strategy
runs in light mode (no Plan B scenario), and `gain_sc` tops out at 0.31 places against the 0.6 bar, so BOX_IF_SC never fires
in any replay (0 of 14360 evaluations, before or after). The warning-time bar of 0.3 therefore never fired either (160
evaluations had a warning, all with no Plan B). Fixing the bar needs a full-field or two-car replay and is left to the
strategy owner.

## Open issues

* Skill is modest: AUC 0.62 at lap ends, 0.70 (2025) / 0.56 (2026) live-like; about a third of deployments give no warning.
* 2026 has more VSCs than any year before; the VSC probability is under-predicted there (0.017 vs 0.045 mean).
* A stopped-car detector from telemetry / positions needs the track outline (stopped on the racing line vs in the run-off).
* Retirement flags (`Retired` / `Stopped` in timing) are not a signal: their time is not recorded in `RaceState`, so the
  training samples cannot be built as-of.
* The alert threshold (0.08) and the 1.5x-over-base rule come from a sweep on 2025 with 31 deployments; treat as soft.
* `leakcheck` has no history context, so it exercises the fallback weights (hungary / monaco / british 2026: PASS).

