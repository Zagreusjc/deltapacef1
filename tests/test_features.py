"""Tests for feature engineering (fuel correction + tyre age)."""

from __future__ import annotations

import pandas as pd

from src.pipeline.features import FeatureEngineer


def test_fuel_corrected_lap_time_decreases_as_fuel_burns():
    laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 10,
            "LapNumber": range(1, 11),
            "LapTimeSeconds": [90.0 + i * 0.01 for i in range(10)],
            "Stint": [1] * 10,
            "Compound": ["MEDIUM"] * 10,
            "TyreLife": range(1, 11),
        }
    )
    out = FeatureEngineer().transform(laps, weather=None)

    assert "FuelCorrectedLapTime" in out.columns
    assert "FuelMassKg" in out.columns
    # Fuel mass should drop each lap.
    assert out["FuelMassKg"].iloc[0] > out["FuelMassKg"].iloc[-1]
    # Fuel effect should shrink over the race.
    assert out["FuelEffectSeconds"].iloc[0] > out["FuelEffectSeconds"].iloc[-1]


def test_tyre_age_from_tyre_life():
    laps = pd.DataFrame(
        {
            "Driver": ["LEC"],
            "LapNumber": [5],
            "LapTimeSeconds": [92.0],
            "Stint": [2],
            "Compound": ["HARD"],
            "TyreLife": [7],
        }
    )
    out = FeatureEngineer().transform(laps)
    assert out["TyreAge"].iloc[0] == 7
