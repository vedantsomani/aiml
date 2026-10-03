"""Race-control bookkeeping: what the messages seen so far add up to.

``RCBook.add(t, event)`` consumes parsed messages in feed order. A penalty is
pending from the moment it is announced until a matching "penalty served"
message; time penalties that never get one are added to the race time after the
flag, so they stay pending to the end.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .parser import RCEvent


class RCBook:
    def __init__(self) -> None:
        self.kinds: Counter = Counter()
        self.unknown = 0
        self.total = 0
        self.pending: dict[str, list[list]] = defaultdict(list)  # car -> [[pen, seconds, reason, t], ...]
        self.issued: Counter = Counter()
        self.served: Counter = Counter()
        self.last_penalty: tuple | None = None  # (t, car, pen, seconds, reason)
        self.open_inv: Counter = Counter()  # car -> investigations open (during the race)
        self.after_inv: Counter = Counter()  # car -> to be investigated after the race
        self.noted: Counter = Counter()
        self.tl_deleted: Counter = Counter()
        self.bw: set[str] = set()
        self.reprimands: Counter = Counter()
        self.sc_deployed_t: float | None = None
        self.sc_in_t: float | None = None
        self.vsc_deployed_t: float | None = None
        self.vsc_ending_t: float | None = None
        self.n_sc = 0
        self.n_vsc = 0
        self.red_t: float | None = None
        self.resume_t: float | None = None
        self.n_red = 0
        self.pit_entry_open = True
        self.pit_entry_since: float | None = None
        self.pit_exit_open: bool | None = None
        self.drs_enabled: bool | None = None
        self.overtake_enabled: bool | None = None
        self.rain_risk: float | None = None
        self.grip_low = False
        self.chequered_t: float | None = None

    def add(self, t: float, e: RCEvent) -> None:
        self.total += 1
        self.kinds[e.kind] += 1
        k = e.kind
        if k == "unknown":
            self.unknown += 1
        elif k == "penalty" and e.car:
            self.pending[e.car].append([e.pen, e.seconds, e.reason, t])
            self.issued[e.car] += 1
            self.last_penalty = (t, e.car, e.pen, e.seconds, e.reason)
            for c in e.cars:  # a decision closes one open investigation of the car
                if self.open_inv[c] > 0:
                    self.open_inv[c] -= 1
        elif k == "penalty_served" and e.car:
            self._serve(e)
        elif k == "investigation":
            for c in e.cars:
                self.open_inv[c] += 1
        elif k == "investigation_after":
            for c in e.cars:
                self.after_inv[c] += 1
        elif k == "investigation_closed":
            for c in e.cars:
                if self.open_inv[c] > 0:
                    self.open_inv[c] -= 1
        elif k == "noted":
            for c in e.cars:
                self.noted[c] += 1
        elif k == "track_limits_deleted" and e.car:
            self.tl_deleted[e.car] += 1
        elif k == "track_limits_reinstated" and e.car and self.tl_deleted[e.car] > 0:
            self.tl_deleted[e.car] -= 1
        elif k == "bw_flag" and e.car:
            self.bw.add(e.car)
        elif k in ("reprimand", "warning") and e.car:
            self.reprimands[e.car] += 1
        elif k == "sc_deployed":
            self.sc_deployed_t, self.sc_in_t = t, None
            self.n_sc += 1
        elif k == "sc_in_this_lap":
            self.sc_in_t = t
        elif k == "vsc_deployed":
            self.vsc_deployed_t, self.vsc_ending_t = t, None
            self.n_vsc += 1
        elif k == "vsc_ending":
            self.vsc_ending_t = t
        elif k == "red_flag":
            self.red_t, self.n_red = t, self.n_red + 1
        elif k == "race_resume":
            self.resume_t = t
        elif k == "pit_entry_closed":
            self.pit_entry_open, self.pit_entry_since = False, t
        elif k == "pit_entry_open":
            self.pit_entry_open, self.pit_entry_since = True, t
        elif k == "pit_exit_open":
            self.pit_exit_open = True
        elif k == "pit_exit_closed":
            self.pit_exit_open = False
        elif k == "drs_enabled":
            self.drs_enabled = True
        elif k == "drs_disabled":
            self.drs_enabled = False
        elif k == "overtake_enabled":
            self.overtake_enabled = True
        elif k == "overtake_disabled":
            self.overtake_enabled = False
        elif k == "rain_risk":
            self.rain_risk = e.value
        elif k == "grip_low":
            self.grip_low = True
        elif k == "grip_normal":
            self.grip_low = False
        elif k == "chequered":
            self.chequered_t = t

    def _serve(self, e: RCEvent) -> None:
        lst = self.pending[e.car]
        for pred in (lambda p: p[0] == e.pen and p[1] == e.seconds and p[2] == e.reason,
                     lambda p: p[0] == e.pen and p[1] == e.seconds,
                     lambda p: p[0] == e.pen):
            for i, p in enumerate(lst):
                if pred(p):
                    del lst[i]
                    self.served[e.car] += 1
                    return

    # ------------------------------------------------------------------ reads
    def time_pending_s(self, car: str) -> float:
        return float(sum(p[1] or 0 for p in self.pending.get(car, ()) if p[0] == "time"))

    def drive_pending(self, car: str) -> bool:
        return any(p[0] in ("drive_through", "stop_go") for p in self.pending.get(car, ()))
