"""Tests for feature engineering (fuel correction + tyre age)."""

from __future__ import annotations

import numpy as np
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


def test_merge_weather_tolerates_nat_time_on_laps():
    """Laps with NaT Time must not crash merge_asof; they keep NaN weather metrics."""
    laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 3,
            "LapNumber": [1, 2, 3],
            "LapTimeSeconds": [90.0, 90.5, 91.0],
            "Stint": [1] * 3,
            "Compound": ["MEDIUM"] * 3,
            "TyreLife": [1, 2, 3],
            "Time": [
                pd.Timedelta(seconds=100),
                pd.NaT,
                pd.Timedelta(seconds=300),
            ],
        }
    )
    weather = pd.DataFrame(
        {
            "Time": [
                pd.Timedelta(seconds=50),
                pd.NaT,
                pd.Timedelta(seconds=250),
            ],
            "AirTemp": [25.0, np.nan, 27.0],
            "TrackTemp": [30.0, np.nan, 34.0],
        }
    )
    out = FeatureEngineer().transform(laps, weather=weather)

    assert len(out) == 3
    assert out["LapNumber"].tolist() == [1, 2, 3]
    # Middle lap had no timestamp — weather metrics must be NaN.
    assert pd.isna(out.loc[out["LapNumber"] == 2, "TrackTemp"].iloc[0])
    assert pd.isna(out.loc[out["LapNumber"] == 2, "AirTemp"].iloc[0])
    # Other laps received a weather join.
    assert pd.notna(out.loc[out["LapNumber"] == 1, "TrackTemp"].iloc[0])
    assert pd.notna(out.loc[out["LapNumber"] == 3, "TrackTemp"].iloc[0])
