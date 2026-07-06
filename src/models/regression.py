"""
DeltaPace :: Tyre Degradation Regression
========================================

``TireDegradationModel`` fits a simple scikit-learn linear regression that
answers: *"as this driver's tyres age, how many seconds per lap do they lose?"*

Model (per driver + compound group)::

    FuelCorrectedLapTime = intercept + slope * TyreAge

``slope`` is the **degradation rate** (seconds lost per lap of tyre age). A
slope of +0.05 means each additional lap on that set costs roughly five
hundredths of a second — the rising line fans will see in Phase 4 charts.

Future crossover (not implemented here)
---------------------------------------
Phase 3 stops at per-driver degradation. Phase 5+ will compare cumulative time
lost over stint length against ``MLSettings.pit_stop_loss_seconds`` to estimate
where a 2-stop strategy overtakes a 3-stop — the "crossover lap". The result
dataclass and ``predict_lap_time`` method are structured so that logic can plug
in without rewriting the fitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import ML_SETTINGS, MLSettings

logger = logging.getLogger(__name__)


class DegradationModelError(ValueError):
    """Raised when input data lacks columns required for regression."""


@dataclass(frozen=True)
class DegradationFit:
    """Regression output for one (Driver, Compound) group."""

    driver: str
    full_name: str | None
    team_name: str | None
    team_color: str | None
    compound: str
    degradation_per_lap: float  # slope — seconds lost per lap of tyre age
    base_pace_seconds: float  # intercept — estimated lap time at TyreAge == 0
    r2_score: float
    sample_count: int
    stint_count: int


class TireDegradationModel:
    """Fit fuel-corrected lap time vs. tyre age, per driver and compound.

    Parameters
    ----------
    settings:
        :class:`~src.config.MLSettings` thresholds (min stint length, etc.).
    """

    TARGET = "FuelCorrectedLapTime"
    FEATURE = "TyreAge"

    def __init__(self, settings: MLSettings = ML_SETTINGS) -> None:
        self.settings = settings
        self._fits: list[DegradationFit] = []

    @property
    def fits(self) -> list[DegradationFit]:
        """All successful fits from the last :meth:`fit` call."""
        return list(self._fits)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, enriched_laps: pd.DataFrame) -> pd.DataFrame:
        """Run per-(Driver, Compound) regressions and return a summary table.

        Parameters
        ----------
        enriched_laps:
            Feature-engineered laps **with identity columns** attached
            (FullName, TeamName, TeamColor). Typically the output of
            ``IdentityEnricher.enrich_laps``.

        Returns
        -------
        pd.DataFrame
            One row per successful fit with degradation rate, R², and identity
            fields for plotting/reporting.
        """
        self._validate_input(enriched_laps)
        modelling = self._prepare_modelling_frame(enriched_laps)

        if modelling.empty:
            logger.warning("No laps passed quality filters; no models fit.")
            self._fits = []
            return self.to_dataframe()

        self._fits = []
        for (driver, compound), group in modelling.groupby(["Driver", "Compound"], sort=False):
            fit = self._fit_group(driver, compound, group, enriched_laps)
            if fit is not None:
                self._fits.append(fit)

        logger.info("Fit %d degradation model(s) across drivers/compounds.", len(self._fits))
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fitted results to a flat DataFrame (empty if nothing fit)."""
        if not self._fits:
            return pd.DataFrame(
                columns=[
                    "Driver",
                    "FullName",
                    "TeamName",
                    "TeamColor",
                    "Compound",
                    "DegradationPerLap",
                    "BasePaceSeconds",
                    "R2Score",
                    "SampleCount",
                    "StintCount",
                ]
            )

        rows = [
            {
                "Driver": f.driver,
                "FullName": f.full_name,
                "TeamName": f.team_name,
                "TeamColor": f.team_color,
                "Compound": f.compound,
                "DegradationPerLap": f.degradation_per_lap,
                "BasePaceSeconds": f.base_pace_seconds,
                "R2Score": f.r2_score,
                "SampleCount": f.sample_count,
                "StintCount": f.stint_count,
            }
            for f in self._fits
        ]
        return pd.DataFrame(rows).sort_values("DegradationPerLap").reset_index(drop=True)

    def predict_lap_time(
        self,
        driver: str,
        compound: str,
        tyre_age: float | np.ndarray,
    ) -> float | np.ndarray:
        """Predict fuel-corrected lap time for a given tyre age using a prior fit.

        Used by future crossover/strategy simulation (Phase 5+). Raises if no
        matching fit exists.
        """
        match = next(
            (f for f in self._fits if f.driver == driver and f.compound == compound),
            None,
        )
        if match is None:
            raise DegradationModelError(
                f"No fit for driver={driver!r}, compound={compound!r}."
            )
        ages = np.asarray(tyre_age, dtype=float)
        return match.base_pace_seconds + match.degradation_per_lap * ages

    def estimate_crossover_lap(
        self,
        driver: str,
        compound: str,
        *,
        stint_length_laps: int,
        num_stops_two: int = 1,
        num_stops_three: int = 2,
    ) -> int | None:
        """Placeholder for 2-stop vs 3-stop crossover lap (future Phase).

        Compares total race time under equal stint lengths with different pit
        counts. Returns the lap number where the faster strategy switches, or
        ``None`` if no crossover within a typical race distance.

        Not fully implemented in Phase 3 — returns ``None`` until strategy
        simulation is wired in Phase 5.
        """
        _ = (driver, compound, stint_length_laps, num_stops_two, num_stops_three)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = {self.TARGET, self.FEATURE, "Driver", "Compound"} - set(df.columns)
        if missing:
            raise DegradationModelError(f"Missing required columns: {sorted(missing)}")

    def _prepare_modelling_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to accurate, dry-weather-ish laps suitable for regression."""
        out = df.copy()

        # Numeric targets/features only.
        out = out.dropna(subset=[self.TARGET, self.FEATURE, "Driver", "Compound"])

        if self.settings.require_accurate_laps and "IsAccurate" in out.columns:
            out = out[out["IsAccurate"].fillna(True)]

        # Drop wet/intermediate compounds — different physics.
        if "Compound" in out.columns:
            out = out[~out["Compound"].astype(str).str.upper().isin({"INTERMEDIATE", "WET"})]

        # Require meaningful stint length per (Driver, Stint).
        if "Stint" in out.columns:
            stint_lengths = out.groupby(["Driver", "Stint"]).size()
            valid_stints = stint_lengths[stint_lengths >= self.settings.min_stint_laps].index
            out = out.set_index(["Driver", "Stint"]).loc[valid_stints].reset_index()

        return out

    def _fit_group(
        self,
        driver: str,
        compound: str,
        group: pd.DataFrame,
        source: pd.DataFrame,
    ) -> DegradationFit | None:
        """Fit one linear model; return None if sample size is too small."""
        if len(group) < self.settings.min_samples_per_group:
            logger.debug(
                "Skipping %s/%s: only %d samples (need %d).",
                driver,
                compound,
                len(group),
                self.settings.min_samples_per_group,
            )
            return None

        x = group[[self.FEATURE]].values
        y = group[self.TARGET].values

        model = LinearRegression()
        model.fit(x, y)

        r2 = float(model.score(x, y))
        identity = source.loc[source["Driver"] == driver].iloc[0]

        stint_count = int(group["Stint"].nunique()) if "Stint" in group.columns else 1

        return DegradationFit(
            driver=driver,
            full_name=identity.get("FullName") if hasattr(identity, "get") else None,
            team_name=identity.get("TeamName") if hasattr(identity, "get") else None,
            team_color=identity.get("TeamColor") if hasattr(identity, "get") else None,
            compound=str(compound),
            degradation_per_lap=float(model.coef_[0]),
            base_pace_seconds=float(model.intercept_),
            r2_score=r2,
            sample_count=len(group),
            stint_count=stint_count,
        )
