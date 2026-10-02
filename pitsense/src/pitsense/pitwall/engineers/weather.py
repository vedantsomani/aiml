"""Weather engineer: rain risk, and when slicks or intermediates become the faster tyre.

STUB (owner: weather engineer). Contract keys (docs/ENGINEERING.md) are all
present; the stub passes on the feed's readings and whether anyone is on wet
tyres. To build: a rain-onset nowcast from the weather trend and race control,
and crossover detection from the lap times of cars on different tyres.
"""

from __future__ import annotations

from ..engineer import Engineer

_WET = ("INTERMEDIATE", "WET")


class WeatherEngineer(Engineer):
    name = "weather"

    def race(self, state, view):
        w = state.weather
        return {
            "rainfall": w.get("Rainfall"),
            "track_temp": w.get("TrackTemp"),
            "air_temp": w.get("AirTemp"),
            "humidity": w.get("Humidity"),
            "wet_running": any(d.compound in _WET for d in state.drivers.values() if d.running),
            "rain_prob_10min": None,
            "crossover": None,
        }
