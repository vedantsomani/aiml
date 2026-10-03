# PitSense data card

What the benchmark is built from, how well the reconstruction matches an independent
one, and every way the older feed differs from 2025–26.

## Source and coverage

- **Source.** F1's live-timing archive (`livetiming.formula1.com/static/`). For each session,
  PitSense downloads the raw `Topic.jsonStream` files: one timestamped message per line,
  the same messages the live feed sends. It never reads the `Topic.json` end-of-session
  snapshots.
- **2022 comes from a mirror.** The archive answers HTTP 403 for every 2022 file, so
  `fetch` falls back to FastF1's mirror (`livetiming-mirror.fastf1.dev`), which FastF1 also
  uses. On 2021, where both serve the files, the mirror's copies are byte-identical to
  the archive's (checked on LapCount, TimingAppData and the 6.5 MB TimingData of one
  race). Every file taken from a mirror is listed in its folder's `SOURCES.json`.
- **Coverage.** Every Grand Prix race from 2018 to the 2026 Azerbaijan GP: **188 races**.
  No Grand Prix is left out (see [Excluded](#excluded-races-and-sessions)).
- **Terms.** The timing data stays under F1's terms. The repository ships none of it
  (`data/` is gitignored), and the unit tests use synthetic snippets only.

## Seasons

| Season | Races | Laps rebuilt | with lap time | Decision points | lap end | pit entry | Pit entries¹ | Red-flag races | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2018 | 21 | 22,280 | 21,871 | 21,948 | 21,386 | 562 | 563 | 0 | Pirelli compound names; first stint records after the start (rounds 1–17) |
| 2019 | 21 | 23,643 | 23,225 | 23,287 | 22,610 | 677 | 677 | 0 | |
| 2020 | 17 | 18,400 | 17,974 | 18,053 | 17,500 | 553 | 554 | 3 | reshuffled calendar; Austria and Styria, Silverstone ×2, Bahrain and Sakhir (outer track) |
| 2021 | 22 | 23,728 | 23,301 | 23,245 | 22,524 | 721 | 722 | 6 | Belgium: 3 laps behind the safety car |
| 2022 | 22 | 23,529 | 23,100 | 23,116 | 22,335 | 781 | 782 | 3 | from the mirror |
| 2023 | 22 | 24,386 | 23,949 | 23,910 | 23,063 | 847 | 848 | 4 | |
| 2024 | 24 | 26,574 | 26,109 | 26,102 | 25,304 | 798 | 798 | 3 | rounds 1–9 missing from the index |
| 2025 | 24 | 26,657 | 26,192 | 26,233 | 25,392 | 841 | 841 | 1 | |
| 2026 | 15 | 17,199 | 16,879 | 16,890 | 16,323 | 567 | 567 | 3 | to the Azerbaijan GP |
| **Total** | **188** | **206,396** | **202,600** | **202,784** | **196,437** | **6,347** | **6,352** | **23** | |

¹ Pit-lane entries seen on the timing feed, excluding the 444 entries under a red flag (free
tyre changes, not strategy stops). Lap 1 never has a lap time live. Decision points are the
rows of the benchmark (one per lap end and per pit entry, as built by `bench build`).

## Validation against FastF1

`pitsense validate --year Y` rebuilds every race message by message and compares each
lap with FastF1's post-processed table, joined on driver and lap number. Full per-race
output is in `reports/validation_<year>.txt`.

Tyre age has a second reference: the feed's own laps-on-this-set counter (`TotalLaps`
of the stint record in use when the lap ended). FastF1 counts a set's laps from when its
stint record *arrived*. So it falls behind whenever records arrive late, which is most of
2018 (see the quirks below). A race passes the tyre-age check if it matches either
reference on at least 97% of laps.

| Season | Races | Laps compared | Lap time | Position | Compound | Tyre age (FastF1) | Tyre age (feed counter) | In-lap | Out-lap | CHECK |
|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | 20 of 21¹ | 21,354 | 99.038% | 98.009% | 99.335% | 68.482% | 99.730% | 99.948% | 99.944% | 3 |
| 2019 | 21 | 23,643 | 100.000% | 99.645% | 99.793% | 98.608% | 99.581% | 100.000% | 99.992% | 4 |
| 2020 | 17 | 18,321 | 95.098% | 96.384% | 98.859% | 95.708% | 99.609% | 99.640% | 99.656% | 3 |
| 2021 | 22 | 23,728 | 100.000% | 99.507% | 99.865% | 98.972% | 99.445% | 100.000% | 99.992% | 4 |
| 2022 | 22 | 23,529 | 100.000% | 99.902% | 99.415% | 98.404% | 99.040% | 100.000% | 99.915% | 2 |
| 2023 | 22 | 24,386 | 100.000% | 99.430% | 99.848% | 99.722% | 99.614% | 100.000% | 100.000% | 1 |
| 2024 | 24 | 26,574 | 100.000% | 99.797% | 99.436% | 99.548% | 98.822% | 100.000% | 100.000% | 1 |
| 2025 | 24 | 26,657 | 99.996% | 99.722% | 99.917% | 96.722% | 97.490% | 100.000% | 100.000% | 4 |
| 2026 | 15 | 17,199 | 100.000% | 99.046% | 99.744% | 99.063% | 98.500% | 100.000% | 100.000% | 3 |

¹ FastF1 3.8.3 can't load 2018 Monza (see the quirks).

- **One race per season drags 2018 and 2020 down:** the two feed outages, where FastF1 compares
  the wrong laps. Without 2018 Bahrain, 2018 is 100.000% on lap time, 99.411% on position,
  100.000% on in-laps and 99.990% on out-laps. Without 2020 Austria, 2020 is 100.000%,
  99.780%, 100.000% and 100.000%.
- **2018 tyre age** matches FastF1 on only 68% of laps but the feed's own counter on 99.7%:
  see the late stint records under the quirks.
- **2025–26 is unchanged by this work.** Lap records, pit events and final driver states of
  all 39 races are byte-identical to the original reducer's (46,131 records). The only
  exception is `compounds_used`, which feeds `must_stop`, in Miami 2025. Every per-race
  number in `reports/validation_2025.txt` and `_2026.txt` equals the original
  `reports/validation.txt`, except compound in two races. Belgium 2025 went from 93.5% to
  100% and Hungary 2026 from 98.3% to 100%, because FastF1's missing compounds no longer
  count as mismatches. Pooled over the 39 races: lap time 99.998%, position 99.457% and
  tyre age 97.64%, as before; compound 99.66% → 99.85%. The same seven races are CHECK.
- **The feed's counter isn't perfect either.** In Belgium 2025 it trails us by 7–18 laps for
  15 cars (66% agreement), while FastF1 agrees with us on 94.5% of laps. A race therefore
  passes only when one reference agrees closely.

### Races marked CHECK, and why

None of these is a reconstruction error, except the two tyre limitations at the end of
[Excluded](#excluded-races-and-sessions) (2022 Monaco and Austria).

| Race | Below threshold | Reason |
|---|---|---|
| 2018 Bahrain | time 79.4%, pos 69.4%, in/out 98.9/99.0% | Feed outage (lap counter 12→16). FastF1 renumbers laps and loses one for five cars, so its laps are compared with the wrong ones of ours. Ours: 57 laps for the winner, the race distance. |
| 2018 Azerbaijan | pos 94.9% | Order at the line around safety-car restarts (laps 8 and 49). |
| 2018 France | cmp 93.9%, pos 98.9% | A 0-lap set appended after the flag for Sirotkin, which FastF1 spreads over his last 51 laps. |
| 2019 Australia, Azerbaijan, Mexico, Brazil | pos 98.2–98.8% | Order at the line. |
| 2020 Austria | time 15.4%, pos 41.1%, cmp 94.3%, in/out 93.8/94.1% | Feed outage (400 s). FastF1 loses five laps per car. Tyre age matches the feed's own counter on 99.8% of laps. |
| 2020 Italy, Tuscany | pos 98.5%, 98.3% | Red-flag restarts; order at the line. |
| 2021 Imola | pos 94.0%; age 86.1% (feed 99.9%) | Cars running side by side; the red-flag formation lap counts toward tyre age for us but not for FastF1. |
| 2021 Monaco, Styria | pos 98.7%, 98.6% | Order at the line. |
| 2021 Belgium | age 65.0% (feed 69.0%) over 60 laps | Three laps behind the safety car; no lap times; FastF1 reports compound `UNKNOWN`; formation laps under the red-flag procedure. |
| 2022 Monaco | cmp 94.9%, age 84.5% (feed 81.9%), out 98.3% | Stint records rewritten in place (car 6), wet start. Known limitation. |
| 2022 Austria | age 85.8% (feed 97.0% of 498 laps) | Stale `TyresNotChanged` flag on a real change to another used set (Russell lap 40, and two more cars). Known limitation. |
| 2023 Netherlands | pos 92.8% | FastF1 ranks Russell two places lower over laps 21–35 than the order in which the cars crossed the line. Ours is the tower at the crossing and follows that order. |
| 2024 Netherlands | pos 98.7% | Order at the line. |

"Order at the line": we record the tower at the moment a car crosses the line; FastF1
re-orders positions after the session. This was the known cause in 2025–26 too.

2025–26 has the same seven CHECK races as the original report: Australia, Bahrain, Miami and
Belgium 2025; Japan, Monaco and the Netherlands 2026. The reasons are in the README. Monaco
and the Netherlands 2026 now pass on tyre age (feed counter 98.8% and 98.7%) but not on
position.

## Feed quirks and how they're handled

Found while validating 2018–2024. The 2025–26 quirks are in the README.

**Season index (`Index.json`)**

- **2018 has no Australian GP and 2024 starts at round 10**, although their streams are
  published. `archive.INDEX_GAPS` lists the 12 missing sessions (10 races, 2 sprints). Each
  session's own `SessionInfo` stream, published before the session, supplies the fields an
  index entry would have. Round numbers then match the real calendar.
- **2022 is served only by the mirror** (see [Source](#source-and-coverage)).
- **Type "Race" isn't always a Grand Prix.** 2021's sprints were called "Sprint Qualifying",
  and the 2021 index lists a 2022 FOM track test under the Saudi GP, with a path starting
  `../uat/` that would have been written outside `data/raw`. `races()` selects Grands Prix
  by name, and sessions filed under another season are skipped. 2023 Qatar lists two
  duplicate sessions without data; they have no path and are skipped.
- **Circuit key 149 is used for two circuits:** Mugello (2020 Tuscan GP) and Jeddah (from
  2021). Pit-loss priors come from the last visit to the same circuit, so Jeddah's first
  race would have borrowed Mugello's. Mugello gets its own key (−149, which isn't an
  official key). Lusail/Losail is one circuit spelled two ways.

**Tyres**

- **2018 uses Pirelli's compound names**: see [Compound mapping](#compound-mapping-2018).
- **2018 rounds 1–17 publish the first stint records 2.5–5.5 minutes after the start,**
  after lap 1 for nearly every car. From the US GP (round 18) they arrive about 6 minutes
  before the start. Lap 1 of those races therefore has no compound or tyre age (`UNKNOWN`
  in decision rows); that is what was known then. FastF1 starts counting a set's laps when
  its record arrives, so its `TyreLife` is one lap short for the whole first stint. That
  explains the 2018 tyre-age gap with FastF1; the feed's own counter matches us.
- **Several stint records can arrive in one message**: 2018's late first records come after
  any lap-1 stops, and Miami 2025 resent every car's stints after an outage. Before
  the fix, only the last record counted as "used", so `must_stop` stayed 1 for cars that
  had already used two compounds. Every published set now counts (`compounds_used`).
- **Records appended after the flag** (2018 France: a 0-lap set for Sirotkin, four minutes
  after the chequered flag) don't touch any lap. FastF1 spreads that set back over the
  race, which explains France's compound difference.
- **Records rewritten in place** (2022 Monaco, wet start and red flag): the feed relabelled
  existing records (record 2 became HARD at Latifi's lap-19 stop, record 3 MEDIUM after
  the red flag) instead of appending, while shifting a spurious record out. The reducer
  follows the last real record, so it tracks Latifi's tyres wrongly from lap 19 (about 45
  rows). This is one car in one race, so it is a known limitation rather than a new
  heuristic.
- **Red-flag restarts from the pit lane** (2021 Imola): the formation lap behind the safety
  car is pit lane to pit lane. The feed's restart record (and FastF1) leaves it out of the
  set's lap count; we count it, because the tyres drove it. Tyre age is then one higher
  until the next stop.

**Lap times and lap counts**

- **2018 sometimes sends `LastLapTime` about 1.5 s before `NumberOfLaps`.** That value is
  the new lap's time, but the reducer used to write it into the previous lap if that one
  was still inferred (and once gave lap 1 the time of lap 2). A value now completes the
  pending lap only if less than half of that lap has passed since it ended. This affected
  21 values in 2018; 2025–26 never send a lap time apart from its lap count.
- **Feed outages.** In 2018 Bahrain (the lap counter jumps from 12 to 16) and 2020 Austria
  (400 s, laps 9–13), `NumberOfLaps` jumps when the feed comes back. The laps in the
  gap get no lap time, and their decision rows are written together when the data
  returns: that is what a live system would have seen. Our lap counts match the race
  distance (71 for the 2020 Austrian winner). FastF1 renumbers laps one after another and
  loses the laps in the gap, so from there on its laps are compared with the wrong ones of
  ours. That causes the low lap-time and position matches of those two races.
- **Retirements in 2018 are often `Stopped` without `Retired`** (e.g. Ricciardo and Hartley,
  Monza). The reducer already treats `Stopped` as out of the race.

**Pit stops**

- **`PitStopSeries` starts at the 2024 US GP.** Before that the reducer uses
  `PitLaneTimeCollection` for official pit-lane times. The fallback works: in 2018–2023
  every record comes from it, the median lane time per season is 23.5–24.1 s, and
  89.7–100% of records per season fall on an in- or out-lap of our pit entries. The rest are pit-lane passages under a red
  flag, which pit entries deliberately leave out. `PitLaneTimeCollection` isn't published
  for 2021 Belgium (no stops) or the first six 2023 races (archive 403, mirror 404).
  These records only feed the state hash: pit entries come from `TimingData`, and pit
  loss from lap times.
- **2024 São Paulo and Qatar** have fewer official records than pit entries (13 for 19, and
  28 for 61). As above, no feature uses them.

**Validation-only (FastF1)**

- FastF1 writes a missing compound as the strings `"nan"` or `"None"`. `validate` used to
  count those as mismatches.
- FastF1 3.8.3 can't load 2018 Monza (an `IndexError` in its stint correction), so that race
  has no FastF1 comparison. Another race failed once while jolpica rate-limited us and
  loaded on retry.

## Compound mapping (2018)

2018 is the only season whose feed uses Pirelli's own compound names (HYPERSOFT,
ULTRASOFT, SUPERSOFT, SOFT, MEDIUM, HARD, SUPERHARD). From 2019 the feed says SOFT, MEDIUM
and HARD for the softest, middle and hardest of the three dry compounds nominated for that
weekend, and features and models expect those relative names. A 2018 "SOFT" was the
hardest of the three at seven races, the middle one at nine and the softest at
Silverstone, so the names can't be used as they are.

The reducer maps each 2018 name to its rank among that weekend's three nominations
(`config.NOMINATIONS`, `state.relative_compound`). Pirelli announced the nominations
weeks before each race, so the mapping uses no information from the race. Deriving
it from the tyres actually used would leak. No topic in the feed lists the nominations:
SessionInfo, TimingAppData, TyreStintSeries and CurrentTyres were checked, and they only
show the tyres in use.

| Round | Grand Prix | SOFT | MEDIUM | HARD |
|---|---|---|---|---|
| 1 | Australian | Ultrasoft | Supersoft | Soft |
| 2 | Bahrain | Supersoft | Soft | Medium |
| 3 | Chinese | Ultrasoft | Soft | Medium |
| 4 | Azerbaijan | Ultrasoft | Supersoft | Soft |
| 5 | Spanish | Supersoft | Soft | Medium |
| 6 | Monaco | Hypersoft | Ultrasoft | Supersoft |
| 7 | Canadian | Hypersoft | Ultrasoft | Supersoft |
| 8 | French | Ultrasoft | Supersoft | Soft |
| 9 | Austrian | Ultrasoft | Supersoft | Soft |
| 10 | British | Soft | Medium | Hard |
| 11 | German | Ultrasoft | Soft | Medium |
| 12 | Hungarian | Ultrasoft | Soft | Medium |
| 13 | Belgian | Supersoft | Soft | Medium |
| 14 | Italian | Supersoft | Soft | Medium |
| 15 | Singapore | Hypersoft | Ultrasoft | Soft |
| 16 | Russian | Hypersoft | Ultrasoft | Soft |
| 17 | Japanese | Supersoft | Soft | Medium |
| 18 | United States | Ultrasoft | Supersoft | Soft |
| 19 | Mexican | Hypersoft | Ultrasoft | Supersoft |
| 20 | Brazilian | Supersoft | Soft | Medium |
| 21 | Abu Dhabi | Hypersoft | Ultrasoft | Supersoft |

Sources: RaceFans, "Pirelli announces final F1 tyre selections of 2018" (23 Aug 2018),
for all races. Russia is from Autosport, "Hypersoft F1 tyres part of Pirelli's selection
for Russian GP" ("Pirelli ditches supersofts again"). Singapore and Germany were also
checked against their Wikipedia race articles.

Transcription check (not used for the mapping): every dry compound used in each 2018 race
is one of its three nominations, and in every race all three were used. This caught one
transcription error (Russia had been entered as hypersoft/ultrasoft/supersoft).
`validate` maps FastF1's 2018 names the same way, because FastF1 keeps the raw names.

## Excluded races and sessions

**No Grand Prix from 2018 to 2026 is excluded.** All 188 races are in the benchmark.

Sessions that are not Grands Prix stay out:

- Sprints: the 2021 "Sprint Qualifying" races and every later "Sprint". `fetch --sprints`
  still downloads them.
- Pre-season tests, and the 2022 FOM "High Speed Track Test" filed in the 2021 index.
- Practice and qualifying.

Races kept despite oddities:

| Race | Oddity | Why it stays |
|---|---|---|
| 2021 Belgium | Three laps behind the safety car, then the race was stopped. | Its 40 decision points (laps 1–2) are real "don't stop" decisions; the labels are correct. |
| 2018 Bahrain, 2020 Austria | Feed outages (lap counter 12→16; 400 s). | Laps in the gap have no time, and their rows are written together when the feed returns, as a live system would have seen them. Lap counts match the race distance. |
| 2022 Monaco | The feed rewrote stint records in place. | Only car 6's tyres are wrong from lap 19 (about 45 rows); everything else is consistent. |
| 2018 Monza | FastF1 can't load it. | Our side is consistent: every car's lap count, stint records and pit entries add up. No FastF1 reference only. |
| 2018 rounds 1–17 | No compound or tyre age on lap 1 (326 rows `UNKNOWN`). | That is what the feed had published by then. |

Known limitations, not fixed:

- An uncorrected `TyresNotChanged` flag on a real change to another used set of the same
  compound is treated as "no change". Seen for Russell, 2022 Austria lap 40: his age
  stays on the lap-11 set. The flag is stale and same-compound in every season, but usually
  correctly (red-flag restarts continue the same set), so a fix needs more than one rule.
- Tyre age counts the formation lap of a red-flag restart from the pit lane; the feed's own
  counter and FastF1 don't (2021 Imola: +1 until the next stop).
- Positions are the tower at the moment a car crosses the line, as before. FastF1
  post-processes them, which explains every position-only CHECK.

## Reproduce

```bash
pitsense fetch --year 2018 2019 2020 2021 2022 2023 2024        # ~1 GB, resumable
pitsense validate --year 2018 --jobs 3 > reports/validation_2018.txt
pitsense bench build --year 2018 2019 2020 2021 2022 2023 2024 2025 2026 --jobs 3
pitsense bench run
pitsense leakcheck --year 2020 --race austrian --cuts 3
```
