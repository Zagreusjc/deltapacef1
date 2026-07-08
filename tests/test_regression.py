"""Tests for tyre degradation regression on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MLSettings
from src.models.regression import TireDegradationModel


def _synthetic_enriched(slope: float = 0.05, n_laps: int = 20) -> pd.DataFrame:
    """Build laps where fuel-corrected time grows linearly with tyre age."""
    ages = np.arange(1, n_laps + 1, dtype=float)
    base = 90.0
    times = base + slope * ages + np.random.default_rng(42).normal(0, 0.02, n_laps)

    return pd.DataFrame(
        {
            "Driver": ["VER"] * n_laps,
            "FullName": ["Max Verstappen"] * n_laps,
            "TeamName": ["Red Bull Racing"] * n_laps,
            "TeamColor": ["3671C6"] * n_laps,
            "Compound": ["MEDIUM"] * n_laps,
            "Stint": [1] * n_laps,
            "TyreAge": ages,
            "FuelCorrectedLapTime": times,
            "IsAccurate": [True] * n_laps,
        }
    )


def test_regression_recovers_known_slope():
    settings = MLSettings(min_stint_laps=5, min_samples_per_group=8)
    model = TireDegradationModel(settings=settings)
    enriched = _synthetic_enriched(slope=0.05, n_laps=20)

    result = model.fit(enriched)
    assert len(result) == 1
    assert result.iloc[0]["Driver"] == "VER"
    assert abs(result.iloc[0]["DegradationPerLap"] - 0.05) < 0.02
    assert result.iloc[0]["R2Score"] > 0.9
