# Pre-race briefing and post-race report

Code: `src/pitsense/reports/` (`prerace.py` the pre-race facts and plans, `briefing.py` the page, `postrace.py` grading and the
report page, `svg.py` charts and page shell). Tests: `tests/test_reports.py`.

```
pitsense briefing --year 2026 --race azerbaijan --team ferrari [--out file.html] [--sims 288]
pitsense report   --year 2026 --race azerbaijan --team ferrari [--log calls.jsonl] [--k 2] [--telemetry] [--out file.html]
```

Both write one self-contained HTML file (inline CSS and SVG, no scripts, no links, no CDN; dark on screen, light when printed) and a JSON
summary next to it. Defaults go to `reports/<year>-<race>-<team>-briefing|report.(html|json)`. `--team` takes a team name or car numbers.

## Briefing (as of lights out)

`prerace(log, team)` cuts the log at the first `SessionStatus: Started` (`log.until(start)`) and reads nothing else, so the same
page comes from the full log or the cut one (tested, also with the future scrambled). Contents:

* grid and starting tyres (TimingAppData before the start), gap to the best qualifying lap;
* tyre sets left (the `tyresets` engineer: allocation assumption as in docs/engineers/weekend.md, an upper bound);
* circuit, from `ctx.past_races` only (races ended before this one started): pit loss green/SC/VSC, safety-car and VSC chance (history here
  shrunk to all circuits), overtaking difficulty (per-lap pass chance at 0.3-0.6 s a lap advantage vs all circuits), typical stint lengths
  by tyre;
* weather: readings and trend before the start, the weather engineer's 10-minute rain chance, past races here with rain;
* Plan A/B/C for each team car, with expected finish, points, podium and points chance, and the safety-car play (`strategy` simulator);
* rival threats: for each rival the share of simulated races in which it finishes ahead of our car on Plan A (threats from behind,
  targets ahead), with quali pace difference and starting tyre;
* a short text per car written by the voice (`voice.api.Voice.write(..., "brief")`, which is what `brief()` calls, so the guard and the
  template fallback apply; it is the template when `PITSENSE_VOICE` is `off`/`template` or no model is trained).

**The simulator from the grid.** The live engineer needs two laps of timing, so the briefing builds the field itself and calls the same
simulation (`analysis.analyse_focus`, `sc_scenario`, 288 seeded futures): car pace = best qualifying lap x 1.06 (race lap / quali lap,
1.04 to 1.10 over 2026 races), compound offsets and fuel burn as the engineer's defaults, wear = 2.4 / typical stint at the circuit, 0.3 s per
grid slot, opponents' stops from circuit history. Plan A/B/C are the best plans with different stop counts or tyre sequences. These are
ranking aids, not forecasts. Without qualifying on disk the pace is a grid-order assumption (stated in the page).

## Report

The race is replayed through the live runtime at full speed (`PitWallRuntime`, models loaded if a bundle trained before the race exists), or a
calls log from `pitsense pitwall` is used with `--log`. Then:

* **Shadow score per call.** Same rule as `pitsense shadow-score`: a BOX / PREPARE_BOX / BOX_IF_SC call from lap L is right if the car's real
  stop is within +-k laps of L; STAY_OUT is right if there is no stop in L..L+k; red-flag stops and NO_CALL are not scored. Per action rates,
  stop recall, whether the fitted tyre matched, and a per-call timeline over the car's real stints. The summary equals `shadow_score` (tested).
* The team's actual stops against the pre-race Plan A (from the briefing code) and the last Plan A the wall had at the first stop.
* Position over the race (other cars faint, safety-car laps shaded, stops marked), tyre stints.
* Pit loss per stop (the benchmark's measure, vs the field; the pit-lane time when the loss cannot be measured, e.g. most safety-car stops),
  stationary time when the feed has it.
* Mechanic alerts with the time the evidence first showed and when the wall fired (team cars first, then the field).
* Driver radio highlights: real transcripts only (Whisper files next to the mp3s), ranked by strategy and car-problem words.

## Real check: 2026 Azerbaijan, Ferrari (#16 Leclerc, #44 Hamilton)

* Briefing: about 32 KB HTML + 23 KB JSON, 3 s. Grid P2 (soft) and P6 (soft); Plan A for both is S-H-H with stops on laps 11 and 32,
  expected finishes P2.1 and P3.5; B and C within 0.1 places; SC chance 56 %, VSC 43 %, overtaking "easy" (2.6x average), pit loss 20.8 s.
* Report: about 37 KB HTML + 44 KB JSON, 25 s. Real race: both cars started on softs, one safety car (laps 31-38), both pitted laps 31 and 36 for
  mediums (LEC P2 to P4, HAM P6 to P6). Pre-race Plan A was far from the real race (a safety car changed everything). Shadow score for Ferrari
  (+-2): box calls 5/7 right, stay-out 6/6, stops covered 2 of 4; the BOX_IF_SC calls on laps 19 and 30 were wrong (the SC came at 31).
  One mechanic alert in the field was for other teams (HAD, RUS), none for Ferrari. One Ferrari radio clip with a transcript.

## Open issues

* In the environment this was built in, scikit-learn's compiled modules were blocked by an Application Control policy, so no model bundle
  loaded and the weather nowcast was skipped when it failed; the replay ran without the models engineer. Re-run on a normal machine for the
  real calls. Some other tests (`test_models`, `test_rivals`, ...) fail the same way on the base commit.
* Plan A from the grid is a single simulation of a race that has not shown any pace yet; the pace factor is a constant.
* Pit-loss measurement fails for most safety-car stops (lane time shown instead). Telemetry mechanic alerts need `--telemetry`.
* Starting tyre is whatever TimingAppData holds at the start; it can still be a placeholder for pit-lane starters.
