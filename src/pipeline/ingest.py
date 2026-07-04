"""
DeltaPace :: Data Ingestion Layer
=================================

`SessionLoader` is a thin, object-oriented wrapper around the ``fastf1`` API.
Its single responsibility is to **fetch and cache** the raw artifacts of a
Formula 1 session and expose them as clean pandas DataFrames. It performs *no*
feature engineering -- that is the job of ``features.py``. Keeping ingestion and
transformation separate means either half can be tested or swapped in isolation.

Design notes
------------
* The loader is **stateful but lazy**: nothing is fetched until :meth:`load`
  is called, after which results are memoised on the instance.
* It is **fully dynamic**: any ``(year, grand_prix, session)`` combination that
  FastF1 understands works without code changes, because we simply forward the
  :class:`~src.config.RaceContext` values to ``fastf1.get_session``.
* Network/parse failures are wrapped in a domain-specific
  :class:`SessionLoadError` so callers (and the LangGraph DataAgent) can react
  to a single, predictable exception type.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

from src.config import DATA_DIR, DEFAULT_RACE, RaceContext

logger = logging.getLogger(__name__)


class SessionLoadError(RuntimeError):
    """Raised when a session cannot be fetched or parsed by FastF1."""


class SessionLoader:
    """Fetch and cache a single F1 session's timing, weather, and results data.

    Parameters
    ----------
    race:
        The :class:`~src.config.RaceContext` describing which session to load.
        Defaults to the project-wide :data:`~src.config.DEFAULT_RACE`.
    cache_dir:
        Directory FastF1 uses to persist API responses on disk. Reusing this
        across runs turns slow network calls into near-instant disk reads.

    Typical usage
    -------------
    >>> loader = SessionLoader(RaceContext(2023, "Monza", "R"))
    >>> loader.load()
    >>> laps = loader.laps          # cleaned lap-by-lap DataFrame
    >>> weather = loader.weather    # per-timestamp track/air temperature
    """

    def __init__(
        self,
        race: RaceContext | None = None,
        cache_dir: Path = DATA_DIR,
    ) -> None:
        self.race: RaceContext = race or DEFAULT_RACE
        self.cache_dir: Path = Path(cache_dir)

        # Enable the on-disk cache exactly once; this is the key to fast reruns.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))

        # Populated lazily by ``load()``.
        self._session: fastf1.core.Session | None = None
        self._laps: pd.DataFrame | None = None
        self._weather: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, with_telemetry: bool = False) -> "SessionLoader":
        """Fetch the session from FastF1 and materialise its DataFrames.

        Parameters
        ----------
        with_telemetry:
            Telemetry (car channels like speed/throttle) is large and slow to
            download. It is off by default because tire-strategy analysis only
            needs lap times and weather.

        Returns
        -------
        SessionLoader
            ``self``, to allow fluent chaining (``SessionLoader(...).load()``).

        Raises
        ------
        SessionLoadError
            If FastF1 cannot resolve or download the requested session.
        """
        try:
            logger.info("Loading session: %s", self.race.label())
            session = fastf1.get_session(
                self.race.year, self.race.grand_prix, self.race.session
            )
            # ``load`` triggers the actual download / cache read. We only pull
            # the data streams we need for strategy modelling.
            session.load(
                laps=True,
                telemetry=with_telemetry,
                weather=True,
                messages=False,
            )
        except Exception as exc:  # noqa: BLE001 - normalise to one error type
            raise SessionLoadError(
                f"Failed to load {self.race.label()}: {exc}"
            ) from exc

        self._session = session
        self._laps = self._clean_laps(session.laps)
        self._weather = self._extract_weather(session)
        logger.info(
            "Loaded %d laps and %d weather samples for %s",
            len(self._laps),
            len(self._weather),
            self.race.label(),
        )
        return self

    @property
    def session(self) -> fastf1.core.Session:
        """The underlying FastF1 session object (raises if not yet loaded)."""
        self._require_loaded()
        return self._session  # type: ignore[return-value]

    @property
    def laps(self) -> pd.DataFrame:
        """Cleaned lap-by-lap timing DataFrame (one row per driver per lap)."""
        self._require_loaded()
        return self._laps  # type: ignore[return-value]

    @property
    def weather(self) -> pd.DataFrame:
        """Per-sample weather DataFrame including air and track temperature."""
        self._require_loaded()
        return self._weather  # type: ignore[return-value]

    @property
    def results(self) -> pd.DataFrame:
        """Classified session results (grid, finishing position, points)."""
        self._require_loaded()
        return self._session.results  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_loaded(self) -> None:
        """Guard clause ensuring ``load()`` ran before data access."""
        if self._session is None:
            raise SessionLoadError(
                "Session not loaded yet -- call .load() before accessing data."
            )

    @staticmethod
    def _clean_laps(raw_laps: pd.DataFrame) -> pd.DataFrame:
        """Normalise the raw FastF1 laps frame into analysis-ready columns.

        We keep only the fields the strategy model cares about and convert
        FastF1's ``Timedelta`` lap times into plain float seconds, which are far
        easier to feed into scikit-learn downstream.
        """
        laps = raw_laps.copy()

        # Convert the primary target (lap time) into seconds for modelling.
        if "LapTime" in laps.columns:
            laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

        # Retain a focused, well-named subset. ``errors='ignore'`` keeps this
        # resilient across FastF1 versions that may rename/omit columns.
        keep = [
            "Driver",
            "Team",
            "LapNumber",
            "LapTime",
            "LapTimeSeconds",
            "Stint",
            "Compound",
            "TyreLife",
            "FreshTyre",
            "PitInTime",
            "PitOutTime",
            "TrackStatus",
            "IsAccurate",
            # Session-relative timestamps kept so the feature layer can join
            # weather samples onto laps with a temporal ``merge_asof``.
            "Time",
            "LapStartTime",
        ]
        existing = [c for c in keep if c in laps.columns]
        return laps[existing].reset_index(drop=True)

    @staticmethod
    def _extract_weather(session: fastf1.core.Session) -> pd.DataFrame:
        """Pull the weather stream, guaranteeing the temperature columns exist.

        Different sessions/seasons occasionally lack a weather feed. Rather than
        crash the pipeline we return an empty, correctly-typed frame so the
        feature layer can decide how to handle missing temperatures.
        """
        expected_cols = ["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall"]
        weather = getattr(session, "weather_data", None)

        if weather is None or len(weather) == 0:
            logger.warning("No weather data for %s; returning empty frame.",
                           session.event.get("EventName", "session"))
            return pd.DataFrame(columns=expected_cols)

        weather = weather.copy()
        existing = [c for c in expected_cols if c in weather.columns]
        return weather[existing].reset_index(drop=True)
