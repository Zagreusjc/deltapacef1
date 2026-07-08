"""Tests for offline mock report generation."""

from __future__ import annotations

import pandas as pd

from src.config import BedrockSettings, RaceContext
from src.pipeline.analysis import AnalysisResult
from src.report.writer import ReportWriter


def _minimal_result() -> AnalysisResult:
    race = RaceContext(2023, "Monza", "R")
    degradation = pd.DataFrame(
        {
            "Driver": ["VER", "LEC"],
            "FullName": ["Max Verstappen", "Charles Leclerc"],
            "TeamName": ["Red Bull Racing", "Ferrari"],
            "TeamColor": ["3671C6", "DC0000"],
            "Compound": ["MEDIUM", "MEDIUM"],
            "DegradationPerLap": [0.03, 0.06],
            "BasePaceSeconds": [89.0, 89.5],
            "R2Score": [0.95, 0.90],
            "SampleCount": [20, 18],
            "StintCount": [1, 1],
        }
    )
    team_summary = pd.DataFrame(
        {
            "TeamName": ["Red Bull Racing", "Ferrari"],
            "MedianPaceSeconds": [89.2, 89.6],
            "PaceRank": [1, 2],
            "MedianDegradationPerLap": [0.03, 0.06],
            "DegradationRank": [1, 2],
            "TeamColor": ["3671C6", "DC0000"],
        }
    )
    enriched = pd.DataFrame(
        {
            "Driver": ["VER", "LEC"],
            "FullName": ["Max Verstappen", "Charles Leclerc"],
            "TeamName": ["Red Bull Racing", "Ferrari"],
            "TeamColor": ["3671C6", "DC0000"],
            "FuelCorrectedLapTime": [89.0, 89.5],
            "TyreAge": [5, 6],
            "Compound": ["MEDIUM", "MEDIUM"],
        }
    )
    return AnalysisResult(
        race=race,
        laps=pd.DataFrame(),
        features=pd.DataFrame(),
        enriched=enriched,
        degradation=degradation,
        team_summary=team_summary,
    )


def test_mock_report_renders_without_aws():
    settings = BedrockSettings()
    assert settings.mock_mode is True
    report = ReportWriter(settings=settings).generate(_minimal_result())
    assert "DeltaPace Race Report" in report
    assert "Max Verstappen" in report
    assert "Tyre Degradation" in report or "degradation" in report.lower()


def test_bionic_report_option():
    report = ReportWriter().generate(_minimal_result(), bionic=True)
    assert "**" in report
