"""
DeltaPace :: Driver & Team Identity Enrichment
==============================================

``IdentityEnricher`` joins the classified **results** table (full names, car
numbers, official team colors, grid/finish) onto lap-level data so every chart
and report can say "Max Verstappen" instead of "VER" and paint bars in the
correct team livery.

Why a separate module?
----------------------
Ingestion deliberately keeps laps lean for modelling. Identity lives in
``SessionLoader.results`` and is merged *after* feature engineering so the ML
layer stays focused on physics, while presentation layers get human-readable
labels.

Typical flow::

    loader.load()
    features = FeatureEngineer().transform(loader.laps, loader.weather)
    enriched = IdentityEnricher().enrich_laps(features, loader.results)
    team_view = IdentityEnricher().aggregate_team_summary(enriched, degradation)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns copied from results onto every lap row.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "FullName",
    "DriverNumber",
    "TeamName",
    "TeamColor",
    "GridPosition",
    "Position",
    "Points",
)


class IdentityEnrichmentError(ValueError):
    """Raised when laps cannot be joined to results (missing Driver key)."""


class IdentityEnricher:
    """Attach driver/team identity from session results onto lap data.

    The enricher is stateless: pass DataFrames in, receive new DataFrames out.
    All aggregation helpers accept optional degradation output from the ML
    layer so team rankings can combine pace *and* tyre wear in one table.
    """

    # ------------------------------------------------------------------
    # Core join
    # ------------------------------------------------------------------
    def enrich_laps(
        self,
        laps: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Left-join identity columns from ``results`` onto ``laps``.

        Join key is the three-letter ``Driver`` code present in both frames.
        Missing identity fields are left as NaN rather than dropping laps.
        """
        if laps.empty:
            return laps.copy()

        if "Driver" not in laps.columns:
            raise IdentityEnrichmentError("Laps frame is missing 'Driver' column.")

        if results.empty or "Driver" not in results.columns:
            logger.warning("No results to join; returning laps without identity.")
            out = laps.copy()
            for col in IDENTITY_COLUMNS:
                if col not in out.columns:
                    out[col] = pd.NA
            return out

        # One row per driver in results — dedupe defensively.
        identity = results.drop_duplicates(subset=["Driver"], keep="first")
        cols = ["Driver", *[c for c in IDENTITY_COLUMNS if c in identity.columns]]
        identity = identity[cols]

        enriched = laps.merge(identity, on="Driver", how="left", suffixes=("", "_res"))

        # Prefer results TeamName over laps Team when both exist.
        if "TeamName" in enriched.columns and "Team" in enriched.columns:
            enriched["TeamName"] = enriched["TeamName"].fillna(enriched["Team"])
        elif "Team" in enriched.columns and "TeamName" not in enriched.columns:
            enriched["TeamName"] = enriched["Team"]

        logger.info(
            "Enriched %d laps with identity for %d drivers.",
            len(enriched),
            identity["Driver"].nunique(),
        )
        return enriched

    # ------------------------------------------------------------------
    # Team-level aggregations (for reports & future Streamlit views)
    # ------------------------------------------------------------------
    def aggregate_team_pace(
        self,
        enriched_laps: pd.DataFrame,
        *,
        pace_column: str = "FuelCorrectedLapTime",
    ) -> pd.DataFrame:
        """Rank teams by median fuel-corrected pace (lower = faster).

        Returns one row per ``TeamName`` with median pace, lap count, and the
        team's official color for plotting.
        """
        if enriched_laps.empty or pace_column not in enriched_laps.columns:
            return pd.DataFrame(
                columns=["TeamName", "MedianPaceSeconds", "LapCount", "TeamColor"]
            )

        valid = enriched_laps.dropna(subset=[pace_column, "TeamName"])
        grouped = (
            valid.groupby("TeamName", as_index=False)
            .agg(
                MedianPaceSeconds=(pace_column, "median"),
                LapCount=(pace_column, "count"),
                TeamColor=("TeamColor", "first"),
            )
            .sort_values("MedianPaceSeconds")
        )
        grouped["PaceRank"] = range(1, len(grouped) + 1)
        return grouped.reset_index(drop=True)

    def aggregate_team_degradation(
        self,
        degradation: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rank teams by median tyre degradation rate (lower = better tyre life).

        Expects the output of :class:`~src.models.regression.TireDegradationModel`.
        """
        if degradation.empty:
            return pd.DataFrame(
                columns=[
                    "TeamName",
                    "MedianDegradationPerLap",
                    "DriverCount",
                    "TeamColor",
                    "DegradationRank",
                ]
            )

        grouped = (
            degradation.groupby("TeamName", as_index=False)
            .agg(
                MedianDegradationPerLap=("DegradationPerLap", "median"),
                DriverCount=("Driver", "nunique"),
                TeamColor=("TeamColor", "first"),
            )
            .sort_values("MedianDegradationPerLap")
        )
        grouped["DegradationRank"] = range(1, len(grouped) + 1)
        return grouped.reset_index(drop=True)

    def compare_teammates(
        self,
        enriched_laps: pd.DataFrame,
        degradation: pd.DataFrame,
        *,
        pace_column: str = "FuelCorrectedLapTime",
    ) -> pd.DataFrame:
        """Side-by-side teammate comparison: pace delta and degradation delta.

        For each team with two drivers, computes:
        * ``PaceDeltaSeconds`` — faster driver's median pace minus slower's
        * ``DegradationDelta`` — difference in degradation rate (positive =
          second driver wears tyres faster)

        Fan-friendly columns ``DriverA`` / ``DriverB`` use ``FullName`` when
        available.
        """
        if enriched_laps.empty:
            return pd.DataFrame()

        # --- Pace per driver ---
        pace = (
            enriched_laps.dropna(subset=[pace_column])
            .groupby(["TeamName", "Driver", "FullName"], as_index=False)
            .agg(MedianPace=(pace_column, "median"), TeamColor=("TeamColor", "first"))
        )

        rows: list[dict] = []
        for team, group in pace.groupby("TeamName"):
            if len(group) < 2:
                continue
            sorted_pace = group.sort_values("MedianPace")
            fast = sorted_pace.iloc[0]
            slow = sorted_pace.iloc[1]

            deg_a = deg_b = None
            if not degradation.empty and "Driver" in degradation.columns:
                d_a = degradation.loc[degradation["Driver"] == fast["Driver"], "DegradationPerLap"]
                d_b = degradation.loc[degradation["Driver"] == slow["Driver"], "DegradationPerLap"]
                if len(d_a):
                    deg_a = float(d_a.mean())
                if len(d_b):
                    deg_b = float(d_b.mean())

            rows.append(
                {
                    "TeamName": team,
                    "TeamColor": fast.get("TeamColor"),
                    "DriverA": fast.get("FullName") or fast["Driver"],
                    "DriverB": slow.get("FullName") or slow["Driver"],
                    "PaceDeltaSeconds": float(slow["MedianPace"] - fast["MedianPace"]),
                    "DegradationA": deg_a,
                    "DegradationB": deg_b,
                    "DegradationDelta": (deg_b - deg_a) if deg_a is not None and deg_b is not None else None,
                }
            )

        return pd.DataFrame(rows)

    def build_team_summary(
        self,
        enriched_laps: pd.DataFrame,
        degradation: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combined team table: pace rank + degradation rank + teammate gaps.

        This is the primary team-level artifact for Phase 5 reports and future
        Streamlit dashboards — one DataFrame, many downstream consumers.
        """
        pace = self.aggregate_team_pace(enriched_laps)
        deg = self.aggregate_team_degradation(degradation)
        teammates = self.compare_teammates(enriched_laps, degradation)

        if pace.empty:
            return deg

        summary = pace.merge(
            deg[["TeamName", "MedianDegradationPerLap", "DegradationRank"]],
            on="TeamName",
            how="outer",
        )

        if not teammates.empty:
            tm = teammates[["TeamName", "PaceDeltaSeconds", "DegradationDelta"]]
            summary = summary.merge(tm, on="TeamName", how="left")

        return summary.sort_values("PaceRank", na_position="last").reset_index(drop=True)
