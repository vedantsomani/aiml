# Incidents: unscheduled stops

Code: `pitwall/engineers/incidents.py` (engineer), `bench/incidents.py` (labels, fit, scoring:
`python -m pitsense incidents-eval`), tests `tests/test_incidents.py`. `in_bench = False`; needs a `feeds=True` archive for
the telemetry inputs (0 without).

Keys per car: `damage_prob` (unplanned stop or retirement on the next 2 laps), `unscheduled_stop_prob_3` (within 3 laps, not explained
by tyre age), `incident_reason`, plus the evidence: `pace_drop`, `pace_drop_prev`, `neigh_drop` (s), `tel_slow`, `tel_power`,
`rc_collision`, `rc_incident`, `rc_debris`, `radio_dmg`. Sector times and the speed trap are not in the race state; telemetry
slowing / full-throttle speed loss stand in for them.

Head of strategy: BOX with reason `damage` for our cars when `damage_prob >= 0.15` (`head.DAMAGE_BOX`), overriding the plan.

## Method

Logistic models on the evidence, fitted on 2025 only (C and alert threshold from leave-one-race-out 2025); 2026 scored once.
Labels (read from the finished race): retirement; or a pit stop not under SC/VSC/red and not to or from wet tyres that is `early`
(stint <= 0.6 x the race median for the compound), `collapse` (2+ s loss against the field just before) or `incident` (race control
named the car in a collision/off-track message). `collapse`/`incident` overlap the engineer's inputs, so `early` and `retire` are the clean kinds.
Alert = `damage_prob >= 0.02`. An alert is true if an event follows within 3 laps of it; an event is caught if an alert was active 1-3 laps before.

## 2025 (leave-one-race-out), events {'retire': 27, 'early': 47, 'incident': 3, 'collapse': 2}

alerts 33, precision 0.09, events 79, recall 0.04, warning median 1.0 laps (mean 1.0), false alerts/car-race 0.065

- early:  0.09, events 47, recall 0.02, warning median 1.0 laps (mean 1.0)
- retire:  0.09, events 27, recall 0.07, warning median 1.0 laps (mean 1.0)
- collapse:  0.09, events 2, recall 0.00, warning median nan laps (mean nan)
- incident:  0.09, events 3, recall 0.00, warning median nan laps (mean nan)
- baseline pace_drop >= 3 s: alerts 50, precision 0.08, events 79, recall 0.05, warning median 1.0 laps (mean 2.0), false alerts/car-race 0.099
- baseline pace_drop >= 2 s: alerts 169, precision 0.04, events 79, recall 0.08, warning median 1.0 laps (mean 1.8), false alerts/car-race 0.352

## 2026 (scored once), events {'early': 39, 'retire': 51, 'collapse': 5, 'incident': 1}

alerts 51, precision 0.12, events 96, recall 0.06, warning median 2.0 laps (mean 2.0), false alerts/car-race 0.132

- early:  0.12, events 39, recall 0.08, warning median 1.0 laps (mean 1.7)
- retire:  0.12, events 51, recall 0.04, warning median 2.5 laps (mean 2.5)
- collapse:  0.12, events 5, recall 0.20, warning median 2.0 laps (mean 2.0)
- incident:  0.12, events 1, recall 0.00, warning median nan laps (mean nan)
- baseline pace_drop >= 3 s: alerts 74, precision 0.11, events 96, recall 0.08, warning median 1.5 laps (mean 1.8), false alerts/car-race 0.193
- baseline pace_drop >= 2 s: alerts 204, precision 0.07, events 96, recall 0.16, warning median 2.0 laps (mean 1.9), false alerts/car-race 0.553

## Unscheduled stops in 2026

| race | car | lap | kind | compound / stint | max damage_prob in 3 laps before | warned | laps of warning |
|---|---|---|---|---|---|---|---|
| Australian | COL | 9 | early | HARD / 9 laps | 0.01 | no | - |
| Australian | HAD | 11 | retire | retired | 0.00 | no | - |
| Australian | ALO | 13 | retire | retired | 0.01 | no | - |
| Australian | BOT | 16 | retire | retired | 0.01 | no | - |
| Australian | STR | 34 | retire | retired | 0.01 | no | - |
| Chinese | HAD | 1 | early | SOFT / 1 laps | 0.00 | no | - |
| Chinese | STR | 10 | retire | retired | 0.00 | no | - |
| Chinese | ALO | 32 | early | MEDIUM / 1 laps | 0.01 | no | - |
| Chinese | VER | 46 | retire | retired | 0.00 | no | - |
| Japanese | BEA | 21 | retire | retired | 0.01 | no | - |
| Japanese | STR | 30 | early | MEDIUM / 6 laps | 0.00 | no | - |
| Miami | HUL | 1 | early | MEDIUM / 1 laps | 0.00 | no | - |
| Miami | GAS | 5 | retire | retired | 0.00 | no | - |
| Miami | HAD | 5 | retire | retired | 0.00 | no | - |
| Miami | LAW | 7 | retire | retired | 0.00 | no | - |
| Miami | HUL | 8 | retire | retired | 0.01 | no | - |
| Miami | BOT | 21 | early | SOFT / 15 laps | 0.01 | no | - |
| Miami | BOT | 30 | early | MEDIUM / 9 laps | 0.01 | no | - |
| Miami | STR | 37 | collapse | SOFT / 16 laps | 0.01 | no | - |
| Canadian | BOT | 9 | early | SOFT / 6 laps | 0.00 | no | - |
| Canadian | ALB | 12 | retire | retired | 0.01 | no | - |
| Canadian | PIA | 12 | early | MEDIUM / 11 laps | 0.01 | no | - |
| Canadian | ALO | 24 | retire | retired | 0.00 | no | - |
| Canadian | RUS | 30 | retire | retired | 0.01 | no | - |
| Canadian | NOR | 39 | retire | retired | 0.01 | no | - |
| Canadian | PER | 40 | retire | retired | 0.01 | no | - |
| Canadian | HAD | 62 | early | SOFT / 10 laps | 0.00 | no | - |
| Monaco | BEA | 1 | early | MEDIUM / 1 laps | 0.00 | no | - |
| Monaco | ALO | 3 | early | MEDIUM / 3 laps | 0.00 | no | - |
| Monaco | STR | 4 | early | MEDIUM / 4 laps | 0.00 | no | - |
| Monaco | PER | 9 | early | MEDIUM / 5 laps | 0.00 | no | - |
| Monaco | OCO | 9 | early | MEDIUM / 9 laps | 0.01 | no | - |
| Monaco | HUL | 12 | early | MEDIUM / 12 laps | 0.01 | no | - |
| Monaco | BOT | 16 | retire | retired | 0.01 | no | - |
| Monaco | BEA | 28 | retire | retired | 0.01 | no | - |
| Monaco | NOR | 44 | retire | retired | 0.01 | no | - |
| Monaco | STR | 57 | retire | retired | 0.03 | yes | 3 |
| Monaco | LEC | 65 | retire | retired | 0.00 | no | - |
| Monaco | SAI | 71 | retire | retired | 0.00 | no | - |
| Barcelona | STR | 5 | early | HARD / 5 laps | 0.00 | no | - |
| Barcelona | BOT | 15 | early | MEDIUM / 1 laps | 0.01 | no | - |
| Barcelona | HUL | 30 | retire | retired | 0.01 | no | - |
| Barcelona | BOR | 33 | collapse | HARD / 18 laps | 0.01 | no | - |
| Barcelona | ALB | 34 | early | HARD / 5 laps | 0.00 | no | - |
| Barcelona | ALO | 38 | retire | retired | 0.01 | no | - |
| Barcelona | BEA | 61 | retire | retired | 0.00 | no | - |
| Barcelona | ANT | 62 | retire | retired | 0.00 | no | - |
| Barcelona | LEC | 63 | retire | retired | 0.01 | no | - |
| Austrian | BOT | 2 | early | MEDIUM / 2 laps | 0.00 | no | - |
| Austrian | PER | 4 | early | MEDIUM / 4 laps | 0.00 | no | - |
| Austrian | SAI | 24 | retire | retired | 0.02 | no | - |
| Austrian | STR | 46 | retire | retired | 0.00 | no | - |
| British | ALB | 1 | early | MEDIUM / 1 laps | 0.00 | no | - |
| British | PIA | 2 | early | MEDIUM / 2 laps | 0.00 | no | - |
| British | ALB | 14 | collapse | HARD / 13 laps | 0.02 | yes | 2 |
| British | ALB | 23 | early | MEDIUM / 9 laps | 0.01 | no | - |
| British | RUS | 34 | early | HARD / 11 laps | 0.01 | no | - |
| British | HUL | 37 | retire | retired | 0.00 | no | - |
| British | ANT | 41 | early | HARD / 6 laps | 0.01 | no | - |
| British | ANT | 43 | early | MEDIUM / 2 laps | 0.01 | no | - |
| British | ALB | 44 | retire | retired | 0.00 | no | - |
| British | VER | 47 | retire | retired | 0.00 | no | - |
| Belgian | PER | 13 | early | HARD / 1 laps | 0.00 | no | - |
| Belgian | STR | 26 | retire | retired | 0.00 | no | - |
| Hungarian | STR | 8 | early | SOFT / 8 laps | 0.01 | no | - |
| Hungarian | BOT | 14 | retire | retired | 0.01 | no | - |
| Hungarian | PER | 48 | early | SOFT / 2 laps | 0.01 | no | - |
| Hungarian | PIA | 56 | retire | retired | 0.00 | no | - |
| Dutch | BOT | 3 | retire | retired | 0.00 | no | - |
| Dutch | BEA | 3 | retire | retired | 0.00 | no | - |
| Dutch | LIN | 5 | early | MEDIUM / 3 laps | 0.00 | no | - |
| Dutch | OCO | 36 | collapse | HARD / 20 laps | 0.01 | no | - |
| Dutch | STR | 45 | early | HARD / 10 laps | 0.01 | no | - |
| Dutch | OCO | 53 | retire | retired | 0.00 | no | - |
| Dutch | ALB | 67 | retire | retired | 0.18 | yes | 2 |
| Italian | LAW | 12 | early | HARD / 9 laps | 0.01 | no | - |
| Italian | ALO | 24 | retire | retired | 0.01 | no | - |
| Italian | STR | 27 | retire | retired | 0.01 | no | - |
| Spanish | HAM | 6 | early | SOFT / 6 laps | 0.00 | no | - |
| Spanish | STR | 13 | retire | retired | 0.00 | no | - |
| Spanish | PER | 31 | early | HARD / 17 laps | 0.04 | yes | 3 |
| Spanish | SAI | 31 | early | MEDIUM / 8 laps | 0.01 | no | - |
| Spanish | SAI | 43 | early | HARD / 12 laps | 0.02 | yes | 1 |
| Spanish | BOT | 43 | early | HARD / 14 laps | 0.03 | yes | 1 |
| Spanish | BEA | 43 | collapse | MEDIUM / 10 laps | 0.02 | no | - |
| Azerbaijan | STR | 8 | retire | retired | 0.01 | no | - |
| Azerbaijan | ALO | 21 | retire | retired | 0.01 | no | - |
| Azerbaijan | ALB | 30 | retire | retired | 0.01 | no | - |
| Azerbaijan | NOR | 36 | retire | retired | 0.01 | no | - |
| Azerbaijan | GAS | 36 | retire | retired | 0.01 | no | - |
| Azerbaijan | COL | 37 | retire | retired | 0.00 | no | - |
| Bahrain | BOT | 8 | retire | retired | 0.01 | no | - |
| Bahrain | BOR | 15 | incident | SOFT / 6 laps | 0.01 | no | - |
| Bahrain | PER | 21 | early | HARD / 3 laps | 0.00 | no | - |
| Bahrain | ALB | 42 | retire | retired | 0.01 | no | - |
| Bahrain | RUS | 50 | retire | retired | 0.00 | no | - |

## Reading it

Honest result: the signal is weak. Most unplanned stops give no warning in lap times, race control or telemetry (the damage is done on
the in-lap itself, and race-control incident messages usually arrive after the stop). A 4 s collapse against the field is followed by a
stop about 1 time in 5, but only a few percent of events are preceded by one. Recall 6 %, precision 12 % on 2026, a median of 2 laps of
warning, about the same as the plain lap-time rule (`pace_drop >= 3 s`: precision 0.11, recall 0.08) with fewer false alerts per car-race (0.13 vs 0.19).
Use the alert as a watch flag; the BOX call is deliberately rare.
