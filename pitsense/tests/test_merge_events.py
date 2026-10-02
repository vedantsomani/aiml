import gzip
import json

from pitsense.events import Event, EventLog, load_archive_session, parse_stream_line
from pitsense.merge import deep_merge


def test_dict_merge_and_list_index_updates():
    state = {"Lines": {"1": {"Sectors": [{"Value": ""}, {"Value": ""}], "Position": "3"}}}
    deep_merge(state, {"Lines": {"1": {"Sectors": {"1": {"Value": "31.7"}}, "Position": "2"}}})
    assert state["Lines"]["1"]["Sectors"][1]["Value"] == "31.7"
    assert state["Lines"]["1"]["Position"] == "2"


def test_list_update_replaces_and_deleted_removes():
    state = {"PitTimes": {"18": {"Duration": "21.7"}, "44": {"Duration": "22.0"}}, "Stints": [{"a": 1}]}
    deep_merge(state, {"PitTimes": {"_deleted": ["18"]}, "Stints": [{"b": 2}]})
    assert "18" not in state["PitTimes"] and "44" in state["PitTimes"]
    assert state["Stints"] == [{"b": 2}]


def test_merge_never_mutates_the_update():
    update = {"Lines": {"1": {"Stints": [{"Compound": "SOFT"}]}}}
    snapshot = json.dumps(update, sort_keys=True)
    state = deep_merge({}, update)
    state["Lines"]["1"]["Stints"][0]["Compound"] = "HARD"
    assert json.dumps(update, sort_keys=True) == snapshot


def test_parse_stream_line_handles_bom():
    t, data = parse_stream_line('\ufeff01:02:03.456{"CurrentLap":2}')
    assert t == 3723.456 and data == {"CurrentLap": 2}
    assert parse_stream_line("   ") is None


def test_until_is_a_strict_prefix(race_log: EventLog):
    cut = race_log.events[len(race_log) // 2].t
    past = race_log.until(cut)
    assert all(e.t <= cut for e in past)
    assert len(past) < len(race_log)
    assert past.events == race_log.events[: len(past)]


def test_jsonl_roundtrip(tmp_path, race_log: EventLog):
    path = tmp_path / "log.jsonl.gz"
    race_log.save_jsonl(path)
    again = EventLog.load_jsonl(path)
    assert [(e.t, e.topic, e.data) for e in again] == [(round(e.t, 3), e.topic, e.data) for e in race_log]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert json.loads(f.readline())["meta"] == race_log.meta


def test_archive_loader_orders_topics(tmp_path):
    (tmp_path / "LapCount.jsonStream").write_text('\ufeff00:00:01.000{"CurrentLap":1}\n00:00:05.000{"CurrentLap":2}\n', encoding="utf-8")
    (tmp_path / "TimingData.jsonStream").write_text('00:00:05.000{"Lines":{}}\n00:00:02.000{"Lines":{}}\n', encoding="utf-8")
    log = load_archive_session(tmp_path)
    assert [e.t for e in log] == [1.0, 2.0, 5.0, 5.0]
    # same timestamp: LapCount (priority 6) before TimingData (8)
    assert [e.topic for e in log][-2:] == ["LapCount", "TimingData"]
    assert isinstance(log.events[0], Event)
