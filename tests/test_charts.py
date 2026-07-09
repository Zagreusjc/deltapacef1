"""Tests for chart generation (offline, empty-frame guards)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.report.charts import generate_all_charts, plot_degradation_bars


def test_empty_degradation_returns_no_paths(tmp_path: Path):
    assert plot_degradation_bars(pd.DataFrame(), tmp_path) == []


def test_generate_all_charts_with_minimal_data(tmp_path: Path):
    degradation = pd.DataFrame(
        {
            "Driver": ["VER"],
            "FullName": ["Max Verstappen"],
            "TeamName": ["Red Bull Racing"],
            "TeamColor": ["3671C6"],
            "Compound": ["MEDIUM"],
            "DegradationPerLap": [0.04],
        }
    )
    team_summary = pd.DataFrame(
        {
            "TeamName": ["Red Bull Racing"],
            "MedianPaceSeconds": [89.0],
            "TeamColor": ["3671C6"],
            "PaceRank": [1],
        }
    )
    enriched = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER"],
            "FullName": ["Max Verstappen"] * 3,
            "TeamColor": ["3671C6"] * 3,
            "Compound": ["MEDIUM"] * 3,
            "TyreAge": [1, 2, 3],
            "FuelCorrectedLapTime": [89.0, 89.1, 89.2],
        }
    )
    paths = generate_all_charts(enriched, degradation, team_summary, tmp_path)
    assert len(paths) >= 2
    for p in paths:
        assert p.exists()
