import pandas as pd

from pitsense.bench.dataset import load_bench, write_manifest
from pitsense.bench.features import FEATURE_VERSION


def test_manifest_keeps_stale_races_out(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSENSE_DATA", str(tmp_path))
    folder = tmp_path / "bench" / FEATURE_VERSION
    folder.mkdir(parents=True)
    for slug in ("2025-01-a-race", "2026-01-b-race"):
        pd.DataFrame({"race_id": [slug], "start_utc": ["2026-01-01T00:00:00+00:00"], "t": [1.0], "driver": ["1"]}
                     ).to_parquet(folder / f"{slug}.parquet")
    assert set(load_bench().race_id) == {"2025-01-a-race", "2026-01-b-race"}  # no manifest: older builds
    write_manifest(["2026-01-b-race"])  # e.g. `bench build --year 2026` after a 2025+2026 build
    assert set(load_bench().race_id) == {"2026-01-b-race"}
