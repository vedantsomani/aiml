"""Test that reconnect snapshots don't create duplicate laps or pit events.

Reconnect snapshots send the entire state back; we must not duplicate lap records or pit events.
"""



from pitsense.events import Event, EventLog
from pitsense.state import replay


def test_reconnect_snapshot_does_not_duplicate_laps(race_log):
    """Replay with a mid-race snapshot: no duplicate laps."""
    log = race_log

    # Find a point in the middle of the race
    mid_point = len(log.events) // 2
    mid_event = log.events[mid_point]

    # Collect events up to midpoint, then inject a full snapshot (simulating reconnect)
    events = list(log.events[:mid_point])

    # Inject snapshots for key topics to simulate reconnect
    snapshot_topics = [
        "SessionStatus", "DriverList", "LapCount", "TrackStatus", "TimingAppData",
        "TyreStintSeries", "PitStopSeries"
    ]
    for topic in snapshot_topics:
        if topic in log.topics():
            # Find the current state of this topic
            for e in log.events[:mid_point]:
                if e.topic == topic:
                    # Create a snapshot event (which is a full state copy, not a delta)
                    events.append(Event(mid_event.t, topic, e.data, len(events), "snapshot"))

    # Continue with the rest of events
    events.extend(log.events[mid_point:])
    log_with_reconnect = EventLog(events, log.meta)

    state = replay(log_with_reconnect)

    # Verify no duplicate laps
    lap_keys = {}
    for lap in state.laps:
        key = (lap.driver, lap.lap)
        assert key not in lap_keys, f"Duplicate lap: {lap.driver} lap {lap.lap}"
        lap_keys[key] = lap

    # Compare with original
    original_state = replay(log)
    assert len(state.laps) == len(original_state.laps), \
        f"Replay with snapshot has {len(state.laps)} laps, original has {len(original_state.laps)}"


def test_reconnect_snapshot_does_not_duplicate_pit_events(race_log):
    """Replay with a mid-race snapshot: no duplicate pit events."""
    log = race_log

    # Find a point in the middle of the race
    mid_point = len(log.events) // 2
    mid_event = log.events[mid_point]

    # Collect events up to midpoint, then inject a full snapshot (simulating reconnect)
    events = list(log.events[:mid_point])

    # Inject snapshots for key topics to simulate reconnect
    snapshot_topics = [
        "SessionStatus", "DriverList", "LapCount", "TrackStatus", "TimingAppData",
        "TyreStintSeries", "PitStopSeries"
    ]
    for topic in snapshot_topics:
        if topic in log.topics():
            # Find the current state of this topic
            for e in log.events[:mid_point]:
                if e.topic == topic:
                    # Create a snapshot event (which is a full state copy, not a delta)
                    events.append(Event(mid_event.t, topic, e.data, len(events), "snapshot"))

    # Continue with the rest of events
    events.extend(log.events[mid_point:])
    log_with_reconnect = EventLog(events, log.meta)

    state = replay(log_with_reconnect)

    # Verify no duplicate pit events
    pit_event_keys = {}
    for pit in state.pit_events:
        key = (pit.driver, pit.in_lap)
        assert key not in pit_event_keys, f"Duplicate pit event: {pit.driver} in lap {pit.in_lap}"
        pit_event_keys[key] = pit

    # Compare with original
    original_state = replay(log)
    assert len(state.pit_events) == len(original_state.pit_events), \
        f"Replay with snapshot has {len(state.pit_events)} pit events, original has {len(original_state.pit_events)}"
