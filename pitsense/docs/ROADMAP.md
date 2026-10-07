# Roadmap: from replay-good to race-day-real

What stands between PitSense and a tool a real F1 strategy team would trust, ordered by impact.
Each item names the gap, why it matters on a real pit wall, and how we would measure it.

## 1. Data a real team has and we don't (biggest gap)

| Gap | Why it matters | Status |
|---|---|---|
| **Tyre sets left** (new/used, per compound) | Real plans are limited by sets in the garage | Done: `engineers/tyresets.py` builds each car's inventory from practice and qualifying; plans only use sets that exist |
| **Practice long runs** | Teams know each compound's degradation before the race | Done: `practice.py` fits deg on long runs as the race-start prior |
| **Fuel / energy and car damage** | Teams box for damage and manage energy; we only see lap times | Partly: the mechanics engineers flag power loss and slow cars; no unscheduled-stop hazard in the simulator yet |
| **Pit-crew speed per team** | Stationary time varies by 0.5-1 s between teams | Open: per-team stationary-time priors from history |

## 2. Simulator realism

Done: several SC / VSC per race and red flags with free tyre changes (learnt from history), dirty air, passing
difficulty per circuit, pit loss by track status, a risk setting (`--risk expected|protect|aggressive`), the
teammate double stack under green flag (`head.double_stack`), and wet races on their own simulator.

Open, by value:
- **Rivals that react, fully:** the car directly ahead now covers our undercut with its team's learnt cover rate
  (`analysis.rival_cover`); the rest of the field still follows its own sampled plans. Full reactions need opponent
  trajectories per candidate plan (today they are shared by every plan).
- **Teammates fully planned together:** both cars' plans optimised jointly (stop order, split strategies), beyond
  the double-stack rule.
- **Wet v2:** crossover laps per circuit, a drying-line model, and the field (positions) in the wet simulator.

Measure: simulated vs actual position change per stop, on 2026 races the model has never seen.

## 3. Decision quality ("did the call help?")

Done: decision value (simulator counterfactual, `bench/decision.py`); fair call scoring per action
(`bench/callscore.py`); calibrated confidence (`pitsense calibrate`: per action, fitted only on earlier races;
calls carry `confidence` and `confidence_raw`).

Open:
- The known failure modes from the live Bahrain test: calls 2-7 laps early, compound flip-flops, and
  neutralisations nobody could predict, which should become instant reactions.
- Publish per-call decision value for every 2026 race in the reports.

## 4. Race-day operations

Done: the operator role (accept / reject each call with a reason, `POST /api/ack`, graded by `shadow-score`);
an audit log (every call change with what it changed from and why, every alert, every operator answer); slow work
off the feed path (voice in its own thread, questions on a copy); replays decide on the live cadence at any speed;
health alarms; a token from `PITSENSE_TOKEN`.

Open:
- **Feed failover:** the recorder reconnects after drop-outs and a watchdog restarts it if it dies (backoff 5-60 s,
  `recorder_restarts` in /api/health); still open: a second, independent live source.
- **Latency budget:** a call within 1 s of the lap finishing (measure p95 on a live recording).
- **Learning from the operator:** use rejected calls and their reasons to tune the head (today they are only graded).
- **Multi-screen setup:** strategist, race engineer and mechanics views.

## 5. Frontend

Done: replay lab (pause, speed, lap jumps, bookmarks), alarm banner, strategy comparison tab, gap chart, stint
chart, what-if box (dry and wet tyres), call history, accept / reject and "why it changed" in the call bar.

Next, by value:
1. **Pace and degradation chart** per stint for our cars.
2. **Keyboard shortcuts** and a big-type pit-wall mode for distance reading.
3. **Phone layout polish:** the strategy tab on small screens.

(Done: the risk selector in the header.)
