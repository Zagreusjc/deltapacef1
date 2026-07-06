"""
DeltaPace :: Feature Engineering Layer
======================================

`FeatureEngineer` turns the *raw* lap and weather frames produced by
:class:`~src.pipeline.ingest.SessionLoader` into a model-ready table where each
lap carries physically meaningful, decorrelated features.

Why this layer exists
----------------------
A raw lap time is a tangle of competing effects:

    lap_time = base_pace + fuel_effect + tyre_degradation + track_evolution + noise

To model *tyre degradation* cleanly (the goal of the ML layer) we must first
**subtract the known, deterministic effects** so the regressor sees a signal
dominated by tyre wear. This class computes two such corrections:

1. **Fuel burn-down** -- the car starts heavy (up to 110 kg) and burns fuel at a
   roughly constant rate. Heavier == slower (~0.03 s/kg). We model fuel mass as a
   straight line from lights-out to the flag and convert it to a lap-time
   penalty we can remove.
2. **Track temperature** -- hotter tarmac generally accelerates thermal tyre
   degradation. We join the weather feed onto each lap and derive temperature
   metrics (absolute, delta-from-baseline, rolling trend).

The headline output column is ``FuelCorrectedLapTime``: the lap time with the
fuel penalty removed, leaving tyre age as the dominant explanatory variable.

Downstream, :class:`~src.pipeline.identity.IdentityEnricher` attaches driver
full names and team colors, and :class:`~src.models.regression.TireDegradationModel`
fits degradation rates per driver and compound.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PHYSICS, F1Physics

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Enrich cleaned lap data with fuel-mass and track-temperature features.

    Parameters
    ----------
    physics:
        The :class:`~src.config.F1Physics` constants (max fuel, penalty rate).
        Injected so unit tests can supply alternative assumptions.

    The class is intentionally *stateless* between calls: :meth:`transform`
    takes frames in and returns a new enriched frame, never mutating its inputs.
    """

    def __init__(self, physics: F1Physics = PHYSICS) -> None:
        self.physics = physics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(
        self,
        laps: pd.DataFrame,
        weather: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Produce the enriched, model-ready lap table.

        Steps (each isolated in its own private method for readability):
            1. Add fuel mass + fuel-time penalty (linear burn-down).
            2. Add fuel-corrected lap time.
            3. Add tyre-age features (per-stint lap counter).
            4. Merge weather and derive track-temperature metrics.

        Returns a **new** DataFrame; the input frames are left untouched.
        """
        if laps.empty:
            logger.warning("Received empty laps frame; nothing to engineer.")
            return laps.copy()

        df = laps.copy()
        total_laps = int(df["LapNumber"].max())

        df = self._add_fuel_mass(df, total_laps)
        df = self._add_fuel_corrected_time(df)
        df = self._add_tyre_age(df)
        df = self._merge_weather(df, weather)

        logger.info(
            "Engineered %d features across %d laps (race length ~%d laps).",
            df.shape[1],
            len(df),
            total_laps,
        )
        return df

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------
    def _add_fuel_mass(self, df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
        """Model fuel as a linear burn-down and convert it to a time penalty.

        Assumptions
        -----------
        * The car starts lap 1 carrying ``max_fuel_kg`` and finishes the finish
          line of the last lap holding only ``fuel_reserve_kg``.
        * Consumption per lap is therefore constant:
              burn_per_lap = (max_fuel - reserve) / total_laps
        * Fuel remaining *entering* lap ``L`` (1-indexed):
              fuel(L) = max_fuel - burn_per_lap * (L - 1)

        The lap-time penalty is simply ``fuel_mass * penalty_s_per_kg``.
        """
        max_fuel = self.physics.max_fuel_kg
        reserve = self.physics.fuel_reserve_kg
        # Guard against a degenerate 1-lap session.
        burn_per_lap = (max_fuel - reserve) / max(total_laps, 1)

        df["FuelMassKg"] = (max_fuel - burn_per_lap * (df["LapNumber"] - 1)).clip(
            lower=reserve
        )
        df["FuelEffectSeconds"] = df["FuelMassKg"] * self.physics.fuel_time_penalty_s_per_kg
        df["BurnPerLapKg"] = burn_per_lap
        return df

    @staticmethod
    def _add_fuel_corrected_time(df: pd.DataFrame) -> pd.DataFrame:
        """Remove the fuel penalty so tyre wear becomes the dominant signal.

        ``FuelCorrectedLapTime`` answers: *"what would this lap have cost on a
        near-empty tank?"* -- isolating the tyre-degradation trend the ML layer
        will fit. Only computed when a numeric lap time is available.
        """
        if "LapTimeSeconds" in df.columns:
            df["FuelCorrectedLapTime"] = df["LapTimeSeconds"] - df["FuelEffectSeconds"]
        return df

    @staticmethod
    def _add_tyre_age(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a reliable per-stint tyre-age counter exists.

        FastF1 usually provides ``TyreLife`` (laps on the current set). When it
        is missing/incomplete we reconstruct it by counting laps within each
        ``(Driver, Stint)`` group -- this is the primary feature (X) for the
        degradation regression.
        """
        if "TyreLife" in df.columns and df["TyreLife"].notna().any():
            df["TyreAge"] = df["TyreLife"]
        elif {"Driver", "Stint"}.issubset(df.columns):
            df["TyreAge"] = df.groupby(["Driver", "Stint"]).cumcount() + 1
        else:
            # Fallback: assume a single continuous stint.
            df["TyreAge"] = df["LapNumber"]

        df["TyreAge"] = df["TyreAge"].astype("float")
        return df

    @staticmethod
    def _merge_weather(
        df: pd.DataFrame,
        weather: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Attach per-lap temperatures and derive track-temperature metrics.

        Strategy
        --------
        * Use a time-ordered ``merge_asof`` to snap each lap to the nearest
          preceding weather sample (weather is logged on its own cadence).
        * Derive metrics that capture *track evolution*, which correlates with
          tyre thermal degradation:
              - ``TrackTemp``         : absolute tarmac temperature (deg C)
              - ``TrackTempDelta``    : change vs. the session's opening reading
              - ``TrackTempRoll5``    : 5-lap rolling mean (smooths sensor noise)

        If no usable weather feed exists, the metric columns are filled with NaN
        so downstream code has a consistent schema to rely on.
        """
        metric_cols = ["AirTemp", "TrackTemp", "TrackTempDelta", "TrackTempRoll5"]

        usable = (
            weather is not None
            and not weather.empty
            and "TrackTemp" in weather.columns
            and "Time" in weather.columns
            and "Time" in df.columns
        )

        if not usable:
            logger.warning("No usable weather feed; temperature metrics set to NaN.")
            for col in metric_cols:
                df[col] = np.nan
            return df

        # ``merge_asof`` requires both keys sorted ascending on the join column.
        left = df.sort_values("Time")
        right = weather.sort_values("Time")

        merged = pd.merge_asof(
            left,
            right[["Time", "AirTemp", "TrackTemp"]],
            on="Time",
            direction="nearest",
        )

        baseline = merged["TrackTemp"].iloc[0]
        merged["TrackTempDelta"] = merged["TrackTemp"] - baseline
        merged["TrackTempRoll5"] = (
            merged["TrackTemp"].rolling(window=5, min_periods=1).mean()
        )

        # Restore original lap ordering (LapNumber) for readability downstream.
        sort_key = "LapNumber" if "LapNumber" in merged.columns else "Time"
        return merged.sort_values(sort_key).reset_index(drop=True)
