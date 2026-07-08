"""Tests for driver/team identity enrichment."""

from __future__ import annotations

import pandas as pd

from src.pipeline.identity import IdentityEnricher


def test_enrich_laps_joins_full_name():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER"],
            "LapNumber": [1, 2],
            "FuelCorrectedLapTime": [89.5, 89.7],
            "Team": ["Red Bull Racing"] * 2,
        }
    )
    results = pd.DataFrame(
        {
            "Driver": ["VER"],
            "FullName": ["Max Verstappen"],
            "TeamName": ["Red Bull Racing"],
            "TeamColor": ["3671C6"],
            "GridPosition": [1],
            "Position": [1],
            "Points": [25],
        }
    )
    out = IdentityEnricher().enrich_laps(laps, results)
    assert out["FullName"].iloc[0] == "Max Verstappen"
    assert out["TeamColor"].iloc[0] == "3671C6"


def test_build_team_summary_combines_pace_and_degradation():
    enriched = pd.DataFrame(
        {
            "Driver": ["VER", "PER", "LEC", "SAI"],
            "FullName": ["Max Verstappen", "Sergio Perez", "Charles Leclerc", "Carlos Sainz"],
            "TeamName": ["Red Bull Racing", "Red Bull Racing", "Ferrari", "Ferrari"],
            "TeamColor": ["3671C6"] * 2 + ["DC0000"] * 2,
            "FuelCorrectedLapTime": [89.0, 89.5, 89.2, 89.8],
        }
    )
    degradation = pd.DataFrame(
        {
            "Driver": ["VER", "PER", "LEC", "SAI"],
            "FullName": enriched["FullName"],
            "TeamName": enriched["TeamName"],
            "TeamColor": enriched["TeamColor"],
            "Compound": ["MEDIUM"] * 4,
            "DegradationPerLap": [0.03, 0.05, 0.04, 0.06],
        }
    )
    summary = IdentityEnricher().build_team_summary(enriched, degradation)
    assert "PaceRank" in summary.columns
    assert len(summary) >= 2
