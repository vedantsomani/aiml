"""Pit-stop engineer: what a stop costs right now, and where the car would come out.

Loss. Past races (``ctx.past_races``, summarised by :meth:`summarize_race`) give a
prior per track status: the circuit's last green loss plus the season's drift, and
SC/VSC as a share of green blended with the circuit's own SC/VSC stops. This race's
stops update it as soon as each is measured (``pitloss.measure_stop``: by lap number
under green, over the same stretch of time when a safety car or VSC is involved).

Rejoin. A car boxing now loses ``loss_if_box_now`` (loss now + its team's stationary
time vs the field + any penalty served at the stop). Each car behind within that
gets by unless it stops in the same window: cars in the pit lane do; under SC/VSC
most cars that haven't stopped yet do (rates counted in past races by tyre age and
how long the SC has been out); the rival analyst's ``pit_prob_1`` replaces those
rates when it is available. When race control sends the whole field through the
pit lane, nobody gains or loses places by driving through.

Everything is read from ``state`` / ``self.memory`` / ``self.ctx`` / ``view`` as of now.
"""

from __future__ import annotations

import math
from statistics import median

from ...pitloss import (
    ALREADY_S,
    SETTLE_S,
    STATIONARY_RANGE,
    CoStopRates,
    LossPrior,
    StopLoss,
    condition_now,
    current_losses,
    expected_rejoin,
    gaps_behind,
    loss_prior,
    loss_sd,
    measure_stop,
    pass_through,
    race_summary,
)
from ..engineer import Engineer

PASS_THROUGH_EXTRA_S = 3.0  # a real stop while the field drives through: stationary time + slowing into the box
HISTORY_STOPS_PER_TEAM = 30  # stationary times remembered per team from past races


def _number(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


class PitStopEngineer(Engineer):
    name = "pitstop"
    # rules: penalty_s_pending. Not rivals: rivals reads our loss (undercut), so co-stop rates
    # from past races stand in for the cars behind stopping too (the rejoin was measured that way).
    requires = ("rules",)
    features = ()

    @classmethod
    def summarize_race(cls, final, meta):
        return race_summary(final)

    def __init__(self, ctx, memory):
        super().__init__(ctx, memory)
        self._prior: LossPrior | None = None
        self._rates: CoStopRates | None = None
        self._stat_history: dict[str, list[float]] = {}
        self._stat_field_history: list[float] = []
        self._measured: dict[int, StopLoss] = {}  # pit event index -> measured loss
        self._pending: list[int] = []
        self._n_pits = 0
        self._n_pit_stops = 0
        self._stat_race: dict[str, list[float]] = {}
        self._phase_key = None
        self._phase: tuple[float, bool, bool] = (0.0, False, False)  # (start, pass-through, sc ending)

    # ------------------------------------------------------------------ pre-race knowledge
    def _learn(self) -> None:
        past = [(s.circuit_key, s.extra["pitstop"]) for s in self.ctx.past_races
                if isinstance(s.extra.get("pitstop"), dict)]
        self._prior = loss_prior(past, self.ctx.meta.get("circuit_key"))
        self._rates = CoStopRates.merged(s.get("costop", {}) for _, s in past)
        for _, s in past:
            for team, times in (s.get("stationary") or {}).items():
                self._stat_history.setdefault(team, []).extend(times)
                self._stat_field_history.extend(times)
        self._stat_history = {k: v[-HISTORY_STOPS_PER_TEAM:] for k, v in self._stat_history.items()}
        self._stat_field_history = self._stat_field_history[-10 * HISTORY_STOPS_PER_TEAM:]

    @property
    def prior(self) -> LossPrior:
        if self._prior is None:
            self._learn()
        return self._prior

    @property
    def rates(self) -> CoStopRates:
        if self._rates is None:
            self._learn()
        return self._rates

    # ------------------------------------------------------------------ following the race
    def observe(self, state):
        n = len(state.pit_events)
        if n > self._n_pits:
            self._pending.extend(range(self._n_pits, n))
            self._n_pits = n
        if self._pending:
            waiting = []
            for i in self._pending:
                p = state.pit_events[i]
                if p.under_red:
                    continue
                out = self.memory.index.by_driver.get(p.driver, {}).get(p.out_lap) if p.out_lap else None
                if out is None or state.t < out.t_end + SETTLE_S:
                    waiting.append(i)
                    continue
                m = measure_stop(self.memory.index, p, state.status_log)
                if m is not None:
                    self._measured[i] = m
            self._pending = waiting
        if len(state.pit_stops) > self._n_pit_stops:
            for x in state.pit_stops[self._n_pit_stops:]:
                d = state.drivers.get(x.driver)
                if d is not None and x.stop_time is not None and STATIONARY_RANGE[0] < x.stop_time < STATIONARY_RANGE[1]:
                    self._stat_race.setdefault(d.team, []).append(x.stop_time)
            self._n_pit_stops = len(state.pit_stops)

    def _phase_now(self, state) -> tuple[float, bool, bool]:
        """(when the current green/SC/VSC phase began, field driving through the pit lane, SC in this lap)."""
        key = (len(state.status_log), len(state.rc))
        if key != self._phase_key:
            start, prev = 0.0, None
            for ts, code in state.status_log:
                c = condition_now(code) or "red"
                if c != prev:
                    start, prev = ts, c
            status = state.track_status
            through = pass_through(state.rc, status, start)
            ending = status == "4" and any(m.t >= start and "SAFETY CAR IN THIS LAP" in m.message.upper()
                                           for m in state.rc)
            self._phase, self._phase_key = (start, through, ending), key
        return self._phase

    def _stationary(self, team: str) -> tuple[float | None, float | None]:
        """(this team's expected stationary time, the field's), from past races and this race so far."""
        field = self._stat_field_history + [x for v in self._stat_race.values() for x in v]
        if not field:
            return None, None
        mine = self._stat_history.get(team, []) + self._stat_race.get(team, [])
        field_s = median(field[-10 * HISTORY_STOPS_PER_TEAM:])
        team_s = median(mine[-HISTORY_STOPS_PER_TEAM:]) if len(mine) >= 3 else field_s
        return team_s, field_s

    # ------------------------------------------------------------------ values
    def race(self, state, view):
        prior = self.prior
        samples = {"green": [], "sc": [], "vsc": []}
        for m in self._measured.values():
            if m.typical and m.condition in samples:
                samples[m.condition].append(m.loss)
        est = current_losses(prior, samples)
        start, through, sc_ending = self._phase_now(state)
        status = state.track_status
        _, field_stat = self._stationary("")
        if status == "5":
            cond, now, sd = "red", 0.0, 0.0  # tyre changes under a red flag are free
        elif through:
            cond, now, sd = "pass_through", (field_stat or 2.5) + PASS_THROUGH_EXTRA_S, 1.5
        elif status == "4":
            cond = "sc_ending" if sc_ending else "sc"
            now = (est["sc"] + est["green"]) / 2 if sc_ending else est["sc"]
            sd = loss_sd(prior, "sc")
        elif status in ("6", "7"):
            cond = "vsc_ending" if status == "7" else "vsc"
            now = (est["vsc"] + est["green"]) / 2 if status == "7" else est["vsc"]
            sd = loss_sd(prior, "vsc")
        else:
            cond, now, sd = "green", est["green"], loss_sd(prior, "green", len(samples["green"]))
        return {
            "loss_green": round(est["green"], 3),
            "loss_sc": round(est["sc"], 3),
            "loss_vsc": round(est["vsc"], 3),
            "loss_now": round(now, 3),
            "loss_now_sd": round(sd, 3),
            "loss_condition": cond,
            "pass_through": through,
            "phase_age_s": round(state.t - start, 3),
            "loss_prior_green": prior.green,
            "loss_prior_source": prior.source,
            "loss_drift_s": prior.drift,
            "lane_time_s": prior.lane_time,
            "stationary_field_s": None if field_stat is None else round(field_stat, 3),
            "stops_measured": sum(len(v) for v in samples.values()),
            "cars_in_pit_lane": sum(1 for d in state.drivers.values() if d.running and d.in_pit),
        }

    def car(self, state, number, view):
        order = [x for x in state.running_order() if x.running]
        i = next((k for k, x in enumerate(order) if x.number == number), None)
        if i is None or order[i].position is None:
            return {}
        me = order[i]
        race = view.race(self.name)
        team_stat, field_stat = self._stationary(me.team)
        stat_adj = (team_stat - field_stat) if team_stat is not None and race["loss_condition"] != "red" else 0.0
        penalty = _number(view.car("rules", number).get("penalty_s_pending")) or 0.0
        loss = max(0.0, race["loss_now"] + stat_adj + penalty)
        sd = max(race["loss_now_sd"], 0.5)

        gaps = gaps_behind(order, i)
        cond = condition_now(state.track_status) or "green"
        start, through, _ = self._phase_now(state)
        recent = {}
        for p in state.pit_events:
            if p.driver != number and not p.under_red and p.in_t >= max(start, state.t - ALREADY_S):
                recent[p.driver] = p.in_t
        p_stop = []
        for k, g in enumerate(gaps):
            if g > loss + 4 * sd:
                break
            j = order[i + 1 + k]
            if through:
                p = 0.0  # it drives through the lane but doesn't stop: it gets by while we're stationary
            elif j.in_pit:
                p = 1.0
            else:
                p = self.rates.rate(cond, j.number in recent, j.tyre_age, state.t - start)
            p_stop.append(p)
        r = expected_rejoin(me.position, gaps[:len(p_stop)], loss, sd, p_stop)
        beyond = [g for g in gaps if g >= loss]
        return {
            "rejoin_if_box_now": r.position,
            "rejoin_delta": r.position - me.position,
            "cars_within_loss": r.within,
            "cars_within_stopping": r.within_stopping,
            "passers_expected": r.passers,
            "margin_s": round(beyond[0] - loss, 3) if beyond else None,
            "rejoin_gap_ahead_s": r.gap_ahead,
            "rejoin_gap_behind_s": r.gap_behind,
            "loss_if_box_now": round(loss, 3),
            "stationary_s": None if team_stat is None else round(team_stat, 3),
            "penalty_s": penalty,
        }
