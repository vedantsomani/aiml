# Weekend engineers: tyre sets and qualifying

Code: `weekend.py` (sessions, download, set book, `tyresets` tool), `pitwall/engineers/tyresets.py`,
`quali.py` (logic), `pitwall/engineers/quali.py`, `bench/quali.py`, `quali_model.json` (fitted constants).
Tests: `tests/test_weekend.py`.

## Fetching the weekend

```
pitsense fetch --year 2025 2026 --weekend      # + every other session of each race's meeting
```

Practice 1-3, Sprint Qualifying, Sprint and Qualifying of each meeting, with the default topics, and
`Position.z` for the two qualifying sessions. Race files are untouched. `weekend.before(ref)` lists the
sessions of the meeting that started before `ref`; those streams are all public before it starts.
2025-26 as downloaded: 30 + 21 qualifying sessions (6 + 5 sprint qualifying).

## Tyre-set planner (`tyresets`)

Per car, from the TimingAppData stints of every earlier session of the weekend:

| Key | Meaning |
|---|---|
| `new_soft_left`, `new_medium_left`, `new_hard_left` | new sets of that compound still unused (None if no earlier session is known) |
| `used_sets` | used sets still free to fit, `"S12 M20"`: compound letter + laps on the set |
| `sets_run` | sets run so far this weekend (including the race) |

How sets are told apart: stint `New="true"` is a new set; `New="false"` is matched to the car's set of the
same compound with the closest age at or below `StartLaps` (so a set run in FP1 and again in qualifying is
one set); no match means a used set of unknown origin (counted as used, never as a new set). Wet compounds
are not dry-set allocation and are skipped. During the race the stints fitted so far are taken off (a set
on the car is not "free").

**Allocation assumption (not in any feed).** 13 dry sets at a normal weekend, split 8 soft / 3 medium /
2 hard; 12 at a sprint weekend (7 / 3 / 2). The nominated mix differs by race and the feed never says
which sets a team handed back, so `new_X_left` is an upper bound on what the team really holds. Override
with `ctx.meta["tyre_allocation"]` (dict) and `ctx.meta["tyre_returned"]` (sets handed back per compound),
or `--returned S=1 M=0` in the tool.

Pre-race tool: `pitsense tyresets --year 2026 --race hungary` prints the table for the race grid.
Weekend sets reach the engineer through the session's `SessionRef` fields in `ctx.meta` (archive replay);
tests inject `ctx.meta["weekend_sets"]`. A recording without them gives None values, never a guess.
`in_bench = False`: it reads earlier sessions from disk, so it is not part of the per-row benchmark;
`pitsense leakcheck` still passes with it on the wall (Hungary and Monaco 2026, 0 mismatches).

## Qualifying engineer (`quali`)

Active when the TimingData feed carries `SessionPart` (Qualifying, Sprint Qualifying); silent otherwise.

* **Cut line.** `NoEntries` is [cars in part 1, 2, 3]; the cars that advance from part p are the first
  `NoEntries[p]` of part p (P15 in Q1 / P10 in Q2 with 20 cars, the feed says 22 -> 16 -> 10 in 2026). Cars flagged
  `KnockedOut` are out of later parts. Rank by the part's best time, untimed cars last.
* **Race values:** `part`, `time_left_s` (session clock, frozen under red flag), `cut_pos`, `cut_time_s`,
  `pred_cut_s` (predicted final cut), `bubble_s` (cut car to first car out), `n_on_track`, `traffic_ahead`
  (needs car positions and a circuit outline: cars in the first 40 % of the lap after the line), `running`, `sprint`.
* **Car values:** `eligible`, `best_s`, `rank`, `gap_to_cut_s` (>0 slower than the cut car), `in_zone`, `p_ko`
  (probability of ending the part beyond the cut), `send` (`SEND_NOW`, `WAIT`, `TOO_LATE`, `ON_TRACK`, `SAFE`),
  `laps_needed` (2 for a car in the pits: out-lap + push; 1 on its out-lap), `slack_s`.
* **Send now.** A car needs a run when it has no time, is in the drop zone or is slower than the predicted
  cut. The push lap may start before the clock reaches zero, so the window shuts when the time left is
  shorter than an out-lap, taken as 1.3 x the best lap so far (observed out-lap times include garage waiting).
  `SEND_NOW` when slack <= 25 s + 6 s per other car waiting to leave (+15 s with 8 or more cars on track);
  `TOO_LATE` when slack < 0. These thresholds are judgement, not fitted: the benchmark does not score them.
* **Alerts** (level-triggered, in the last 3:00 of a part, `since` = first time the window opened):
  `drop_zone` / `send_now`, e.g. "car 16 (ALB) in the drop zone (P17, +0.154s to the cut), 2:30 left, needs one run (out-lap + push)".
* **Predicted cut time.** `cut now + evolution`, the evolution a table of median improvements by seconds
  left (bins 30/60/120/240/480 s) and part, plus a separate cell when fewer cars than the cut have set a time
  (then "cut now" is the slowest time set). `quali_model.json` holds it and the knockout logit, both fitted on 2025.

## Pit wall

```
pitsense pitwall --year 2026 --race hungary --session qualifying     # or sprint-qualifying, sprint
```

`--session` takes `race`, `sprint`, `qualifying`, `sprint-qualifying`, `practice1-3`. For the two qualifying
sessions the runtime loads only the quali engineer (the race engineers have nothing to say in qualifying),
detected from the archive ref or the SessionInfo message (recordings). The dashboard gets a "Qualifying" panel
(cars by rank: best, gap to the cut, KO %, send call, laps needed; summary line with cut, prediction, bubble
and traffic) and the header shows `Q2  4:32` instead of the lap. Alerts and the radio panel work as for races.
Replayed Hungary 2026 qualifying: 153 snapshots, no errors, median snapshot 5.6 ms.

## Benchmark (`pitsense quali-bench`)

Moments: every 20 s of a Q1/Q2 (SQ1/SQ2) part while its clock runs, replayed with the pit-wall code
(`collect_events`). Labels come from the end of the part: the final cut time and whether the car ends beyond
the cut. 2025: 2085 cut moments, 37 195 car moments (30 sessions). **Tuned on 2025** (leave-one-session-out for the
table variant, 5 session folds for the logit's C), fitted on all of 2025, **2026 scored once**
(20 sessions, 1374 / 26 616 moments; `reports/quali_bench.json`).

Cut time, MAE in seconds (2026):

| Moments | `now` (baseline) | `now` + 2025 mean | evolution table |
|---|---|---|---|
| all | 3.131 | 3.248 | **2.712** |
| cut already defined (15+ timed cars) | 1.649 | 2.161 | **1.355** |
| under 1 min left | 0.219 | 1.593 | **0.159** |
| 1-4 min left | 0.500 | 1.478 | **0.346** |
| over 4 min left | 4.567 | 4.174 | **3.994** |

Knocked out, log loss (2026; lower is better):

| Moments | prior (share knocked out) | rank rule (drop zone now: 0.8 / 0.2) | logit |
|---|---|---|---|
| all | 0.621 | 0.493 | **0.418** |
| under 1 min left | 0.626 | 0.327 | **0.287** |
| 1-4 min left | 0.627 | 0.362 | **0.325** |
| over 4 min left | 0.618 | 0.565 | **0.471** |

The logit uses: gap of the car's best to the predicted cut (clipped to +-3 s, 3 s if untimed), untimed flag,
rank offset to the cut, time left, in pit, laps in the part, gap x time left.

Reading: the table roughly halves the late-session error, but most of the all-moment MAE is the first minutes,
when fewer than 15 cars have a time and the cut is a guess (about 4 s). The 2025 choice of median over mean
(1.54 vs 2.16 s leave-one-out) is why the tail of slow early laps does not drag the table.

## Open issues

* Returned sets and the nominated compound mix are assumptions (see above); for a real team the allocation should be entered by hand.
* Early-session cut prediction (before 15 cars have a time) is weak; a prior from the weekend's practice pace would help.
* Traffic is only a count from positions plus an outline; no model of pit-lane queues, which the feed does not publish. The send-now thresholds are unscored heuristics.
* Q3 has no cut: only best time, rank and the send call.
* Red-flagged parts: the clock freezes, so `time_left_s` holds; the labels use the final state of the part, whatever happened.
* `weekend.py` holds the sprint-weekend test as "a Sprint session exists in the meeting".
