# Rules engineer (sporting director)

Turns race control and the regulations into facts as of now. Code:
`src/pitsense/pitwall/engineers/rules/` (`parser.py`, `book.py`, `regs.py`),
measurements in `src/pitsense/bench/rules.py` (`python -m pitsense rules-report`).

## Values

Race (`rules__<key>`)

| key | meaning |
|---|---|
| `race_dry` | only dry compounds used so far |
| `sc_phase` | `none`, `sc`, `sc_ending`, `vsc`, `vsc_ending`, `red`. From the track status; `sc_ending` once "safety car in this lap" has been shown for the current deployment, `vsc_ending` from the VSC-ending message or status 7 |
| `sc_ending` | bool, `sc_phase` is `sc_ending` or `vsc_ending` |
| `pit_lane_open` | pit entry open (false between "pit lane entry closed" and "open") |
| `pit_exit_open` | last pit-exit message (None before any) |
| `n_sc`, `n_red` | SC + VSC deployments and red flags so far |
| `rc_rain_risk` | latest FIA "risk of rain" percentage (None before the first); `rain_risk_f` = same, -1 when none |
| `drs_enabled`, `low_grip`, `chequered` | latest DRS / grip / flag state |
| `rc_messages`, `rc_unknown` | messages seen, messages the parser did not recognise |
| `reg_min_stops`, `reg_max_stint_laps` | pre-race regulation facts for this race |

Car

| key | meaning |
|---|---|
| `must_stop` | a stop is still required: second dry compound (dry race), minimum stops (Monaco 2025), or the stint limit would be exceeded (Qatar 2025) |
| `stint_laps_left` / `stint_laps_left_f` | laps left on the set under a stint limit (None / 999 without one) |
| `penalty_s_pending` | seconds of time penalty announced and not yet served (s) |
| `drive_through_pending` | a drive-through or stop-go announced and not yet served |
| `penalties_issued`, `reprimands` | counts so far |
| `under_investigation`, `post_race_investigation` | stewards' investigation open now / to be investigated after the race |
| `incidents_noted` | incidents noted involving the car |
| `track_limits_deleted` | lap times deleted for track limits so far (net of reinstatements) |
| `black_white_flag` | has been shown a black-and-white flag |

Alerts (level-triggered): `penalty_pending`, `pit_lane_closed`, `sc_ending`, `red_flag`, `under_investigation`.

Benchmark inputs (`features`): `must_stop`, `penalty_s_pending`, `drive_through_pending`,
`stint_laps_left_f`, `sc_ending`, `pit_lane_open`.

## Parser

`parse(message, category, flag) -> RCEvent` on normalised text; unknown lines return
`kind="unknown"` (counted in `rc_unknown`), never an exception. Covers time penalties
(5/10 s and others), drive-through, stop-go, served, reprimands, investigations
(open, after the race, closed), noted incidents, track-limits deleted / reinstated,
black-and-white and blue flags, yellow sectors, SC / VSC deployed / ending / in this lap,
red flag and resume, pit entry and exit open / closed, DRS and overtake mode, rain risk,
grip, start procedure, chequered flag. Corrections ("CORRECTION:" lines) and trailing
incident time stamps are handled.

Penalty bookkeeping (`RCBook`): a penalty is pending from the announcement until a
matching "penalty served" (same type, seconds, reason, falling back to looser matches).
On 2025-26 races, 89 penalties were announced, 47 were served in the race and 42 stay
pending to the flag (added to race time afterwards).

## Regulation table (`regs.py`)

Pre-race facts only; never add a rule because of how a race went.

| race | rule | source |
|---|---|---|
| all 2025, 2026 | two different dry compounds in a dry race, void once wet tyres are used | FIA Sporting Regulations (tyre article) |
| Monaco 2025 | at least 2 stops (3 sets) | FIA WMSC / Pirelli press release (links in `regs.py`) |
| Qatar 2025 | at most 25 laps per set (57 laps, so 2 stops) | Pirelli / FIA bulletin via RaceFans (link in `regs.py`) |
| Monaco 2026 | two-stop rule dropped, default row | Motorsport Week, 28 Feb 2026 (link in `regs.py`) |

Seasons before 2025 only get the two-compound rule. Article numbers are not cited; the
tech lead should confirm against the FIA documents.

## Measurements (2025-26, 39 races)

Parser coverage: 4876 messages, 0 unknown (0.00 %). By type, see `rules-report` (largest:
blue flags 1515, track-limits deletions 595, sector clears 529, noted incidents 318,
double yellows 308, yellows 267, investigations 133 open / 147 closed / 75 after race,
penalties 89, served 47, SC deployed 28, in this lap 25, VSC deployed 32).

Lead time of the ending message over the track-status change (SC / VSC leaving):

| message | n | median s | p10 s | min s | max s |
|---|---|---|---|---|---|
| SAFETY CAR IN THIS LAP | 25 | 78.8 | 35.6 | 26.1 | 214.0 |
| VSC ENDING | 32 | 12.6 | 10.9 | 10.1 | 14.9 |

The SC-deployed message arrives with status 4 (median lag 0.0 s), so `sc_phase` loses nothing by using the status.

`must_stop` against what finishers did (lap-end rows of 2025-26):

| rows | n | share that stop later |
|---|---|---|
| must_stop = 1 | 14307 | 98.8 % |
| must_stop = 0, dry race | 21680 | 35.8 % |
| last 3 laps of finishers, dry race: must_stop still 1 | 557 | 0.5 % |
| Monaco 2025, must_stop = 1 (with regulation) | 842 | 98.0 % |
| Monaco 2025, two-compound rule only | 511 | 96.7 % |
| Qatar 2025, must_stop = 1 (with regulation) | 612 | 99.5 % |
| Qatar 2025, two-compound rule only | 492 | 99.4 % |

Core leaderboard, 2026 test year (before = base, after = rules features declared):

| task / metric | before | after |
|---|---|---|
| pit_within_1 gbm_hazard log loss / AP | 0.1181 / 0.1375 | 0.1179 / 0.1437 |
| pit_within_3 gbm_hazard log loss / AP | 0.2706 / 0.2868 | 0.2699 / 0.2935 |
| position_after_stop rejoin_gbm exact / within 1 / MAE | 0.5385 / 0.8630 / 0.739 | 0.5441 / 0.8574 / 0.743 |

The gains are inside noise (533 pit entries, 498 / 1507 positives). On the 2025 test
year (selection), no rules features: pit_within_1 log loss 0.1074, pit_within_3 0.2371;
lean set 0.1079 / 0.2383; all numeric keys 0.1083 / 0.2390. So the rules facts do not add
pit-timing skill by themselves; they are constraints for the strategy engineer.
All-NaN columns break the histogram boosting when a training set has no value, so model
inputs use sentinels (`stint_laps_left_f`, `rain_risk_f`).

Leakcheck: hungary 2026 and monaco 2026, 2 cuts, truncate and scramble: PASS.
