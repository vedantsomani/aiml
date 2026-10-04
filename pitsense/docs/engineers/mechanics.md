# Mechanics: telemetry, radio, chief

Three engineers that flag a car problem early. Code: `pitwall/engineers/mechanic_telemetry.py`,
`mechanic_radio.py`, `mechanic_chief.py`; evaluation `bench/mechanics.py`
(`python -m pitsense mechanics-eval --tune-year 2025 --year 2026 --jobs 3`); tests `tests/test_mechanics.py`.
All three have `in_bench = False` and need `feeds=True` archives (silent otherwise).

## What the feed can show

RPM, speed, gear, throttle, brake (on/off), DRS at ~4 Hz, plus X/Y. No temperatures, pressures,
ERS state or brake pressure. A fault is visible only through what it does to motion. Honest
consequence: gross failures (car slowing or stopping, big power loss, neutral) are detectable;
degradation of brakes/gearbox before failure mostly is not.

## mechanic_telemetry

Online, every 4 s of state time (samples measured one ingest late, so pit-entry slowing is not judged).
Per car baseline = its own earlier laps (lags 150 s, frozen while flagged); the field comparison uses
a 150 m grid cell from Position. Common-mode shifts (rain, restart) are divided out; full-throttle
power is compared only in clear air (no car within 1.8 s). Ignored: pits, out-laps, lap 1, SC/VSC/red/yellow
(except stationary-on-track), after the flag, and any time the position feed disagrees with CarData speed.

| key | meaning |
|---|---|
| `power_loss` | full-throttle speed (and rpm) below own baseline, field-relative, 0-1 |
| `gearbox` | neutral while moving, skipped upshifts, rpm/gear mismatch, gear outside its speed range |
| `brake_issue` | less speed shed per second under braking than before |
| `slow_car` | standing still on track, speed far below the field in the cell, coasting down a straight, or pace collapse over a lap |
| `mech_risk`, `mech_issue`, `mech_since`, `mech_active` | worst score, its name (>= 0.5), earliest active alert |
| `<check>_since` | when that level-triggered alert became true |

Alert on `score >= 0.9` (0.6 for slow_car) held for 20/8/12/8 s; clears below 0.6 x threshold for 20-30 s.

## mechanic_radio

Phrase table with weights and negation / question / "fixed" handling (`classify(text)`). Issues: power, brakes,
gearbox, damage, puncture, vibration, leak_fire, electrical, retire, generic. Transcripts count only 15 s after the message.
Keys: `radio_issue`, `radio_risk` (decays, 15 min half-life), `radio_score`, `radio_quote`, `radio_t`, `radio_lap`, `radio_n`.
Alert at risk >= 0.6. Caveat: the phrase table was written reading 2025 **and 2026** transcripts, so the 2026 radio numbers are not clean.

## mechanic_chief

`risk = 1 - prod(1 - w e)` over telemetry (w 1.0), radio (0.8), race control "car stopped / slow" (0.9), lap-time drop (0.5).
Keys: `risk`, `issue`, `since`, `why`, `tel`, `radio`, `lap`, `rc`, `out`, `alert`. Alert at risk >= 0.9. Example message:
`Car 6 (HAD): car slow / stopping suspected (telemetry 1.0 (slow car))`. No race-control "stopped/slow" lines exist in 2025-26
timing data (only stewards' "driving unnecessarily slowly"), so that evidence is dormant.

## Evaluation

Events (2025: 51 over 465 car-races; 2026: 78 over 320): `retire_quiet` / `retire_incident` (car out with 2..total-3 laps; incident = SC/VSC/red/yellow/contact
message in the 2 min before it stopped), `slowdown` (clean lap >= 5 s slower than the previous 5 clean laps, not in traffic). Zero race-control stop events.
Hit = alert raised in [t0 - 3 laps, t_end]; "before" = strictly before the car stopped / slow lap began; "by flag" = before the timing feed
says so (lead measured against it). A false alarm is an alert episode matching no event. Thresholds were chosen on 2025 (best hits - 0.2 x false alarms,
false alarms <= 0.25 per car-race); 2026 scored with them. The baseline (clean lap >= X s slower; X = 4 s chosen) shares its quantity with the
slowdown event, so its "by flag" recall on slowdowns is 1.0 by construction: compare "before".

2026 (scored with 2025 thresholds):

| detector | events | recall before | recall by flag | median lead (s) | false alarms / car-race |
|---|---|---|---|---|---|
| tel_power | 78 | 0.01 | 0.04 | 119 | 0.01 |
| tel_gearbox | 78 | 0.00 | 0.01 | 46 | 0.01 |
| tel_brake | 78 | 0.00 | 0.00 | - | 0.00 |
| tel_slow | 78 | 0.04 | 0.15 | 61 | 0.10 |
| tel_any | 78 | 0.05 | 0.21 | 61 | 0.11 |
| radio | 78 | 0.01 | 0.04 | 76 | 0.03 |
| chief | 78 | 0.05 | 0.21 | 61 | 0.17 |
| lap-time baseline | 78 | 0.04 | 0.24 | 0 | 0.02 |

2025 (tuning year): tel_any 0.10 / 0.24, lead 42 s, FA 0.04; chief 0.10 / 0.24, FA 0.03; radio 0.06 / 0.12, lead 101 s; baseline 0.08 / 0.27, lead 0, FA 0.02.
By kind (2026, by flag): retirements tel_any 0.19 (quiet) / 0.27 (incident) vs baseline 0.03; baseline's 0.24 overall comes from slowdowns.

Reading it: telemetry catches roughly 20-30 % of retirements before the timing feed does (median about a minute earlier, mostly
a car slowing or stopping), 5-10x the lap-time baseline on retirements, at a few false alarms per race. Brake and gearbox checks have no demonstrated skill. Radio is sparse
(~30 clips per race, 2026 has four races with none) and its hits are rare but early.

### Protocol caveat

2026 was scored four times, not once. The first run (before any change) had tel_any 0.17 / 0.32 recall but 0.65 false alarms per car-race,
chief 1.13. Inspecting those false alarms found real faults in the code, fixed without retuning thresholds on 2026: a frozen/scrambled position feed
(2026 Hungary), a restart grid read as stopped cars, pit-entry slowing before the InPit flag, and single-corner windows. The table above is the final run; its 2026 numbers
are therefore optimistic on false alarms and the thresholds were re-chosen on 2025 after each change. Remaining false alarms concentrate in 2026 Hungary (poor positions) and Austria.

## Open issues

- Recall is low in absolute terms; most retirements are crashes or end in the pits with no precursor. A car retiring in the garage is invisible to telemetry.
- No tyre/engine temperatures: no overheating or cooling detection. brake_issue / gearbox untested on real faults.
- Position feed quality (2026 Hungary, Monaco) limits field comparison; the sanity check drops it, so those races get little coverage.
- Radio phrase table: noisy Whisper text, English only, written with sight of 2026 transcripts.
