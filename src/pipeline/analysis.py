"""
DeltaPace :: End-to-End Analysis Orchestrator
=============================================

``AnalysisPipeline`` wires the deterministic batch flow:

    SessionLoader → FeatureEngineer → IdentityEnricher → TireDegradationModel

into a single :meth:`run` call that returns an :class:`AnalysisResult` bundle
ready for charting (:mod:`src.report.charts`) and narrative reporting
(:mod:`src.report.writer`).

This module deliberately contains **no** LLM or plotting logic — it is pure
data orchestration so the same result object can feed a CLI, a future Streamlit
dashboard, or an AWS Lambda handler without rewrites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import RaceContext
from src.models.regression import TireDegradationModel
from src.pipeline.features import FeatureEngineer
from src.pipeline.identity import IdentityEnricher
from src.pipeline.ingest import SessionLoader
from src.storage.s3 import S3ArtifactStore

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for every DataFrame produced by a single pipeline run."""

    race: RaceContext
    laps: pd.DataFrame
    features: pd.DataFrame
    enriched: pd.DataFrame
    degradation: pd.DataFrame
    team_summary: pd.DataFrame


class AnalysisPipeline:
    """Run ingest → features → identity → regression for one race session.

    Dependencies are injectable for unit tests; defaults use production classes.
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        identity: IdentityEnricher | None = None,
        model: TireDegradationModel | None = None,
    ) -> None:
        self._features = feature_engineer or FeatureEngineer()
        self._identity = identity or IdentityEnricher()
        self._model = model or TireDegradationModel()

    def run(
        self,
        race: RaceContext,
        *,
        with_telemetry: bool = False,
        upload: bool = False,
        store: S3ArtifactStore | None = None,
    ) -> AnalysisResult:
        """Execute the full analysis chain and optionally upload artifacts to S3.

        Parameters
        ----------
        race:
            Which session to analyse.
        with_telemetry:
            Forwarded to :meth:`SessionLoader.load` (off by default).
        upload:
            When True, upload CSV artifacts via ``store`` (no-op if S3 disabled).
        store:
            S3 client wrapper; defaults to a fresh :class:`S3ArtifactStore`.
        """
        label = race.label()
        logger.info("Starting analysis pipeline for %s", label)

        # --- Stage 1: ingest ------------------------------------------------
        logger.info("[%s] Stage 1/5 — loading session data", label)
        loader = SessionLoader(race).load(with_telemetry=with_telemetry)
        laps = loader.laps
        weather = loader.weather
        results = loader.results
        logger.info("[%s] Loaded %d laps", label, len(laps))

        # --- Stage 2: feature engineering -----------------------------------
        logger.info("[%s] Stage 2/5 — engineering features", label)
        features = self._features.transform(laps, weather)
        logger.info("[%s] Engineered %d columns", label, features.shape[1])

        # --- Stage 3: identity enrichment -----------------------------------
        logger.info("[%s] Stage 3/5 — enriching driver/team identity", label)
        enriched = self._identity.enrich_laps(features, results)
        logger.info("[%s] Enriched %d lap rows", label, len(enriched))

        # --- Stage 4: degradation modelling ---------------------------------
        logger.info("[%s] Stage 4/5 — fitting tyre degradation models", label)
        degradation = self._model.fit(enriched)
        logger.info("[%s] Fit %d degradation row(s)", label, len(degradation))

        # --- Stage 5: team summary ------------------------------------------
        logger.info("[%s] Stage 5/5 — building team summary", label)
        team_summary = self._identity.build_team_summary(enriched, degradation)
        logger.info("[%s] Team summary: %d team(s)", label, len(team_summary))

        result = AnalysisResult(
            race=race,
            laps=laps,
            features=features,
            enriched=enriched,
            degradation=degradation,
            team_summary=team_summary,
        )

        if upload:
            s3 = store or S3ArtifactStore()
            logger.info("[%s] Uploading session artifacts to S3", label)
            s3.upload_session_artifacts(
                race,
                laps=laps,
                features=features,
                degradation=degradation,
                team_summary=team_summary,
            )

        logger.info("Analysis pipeline complete for %s", label)
        return result
