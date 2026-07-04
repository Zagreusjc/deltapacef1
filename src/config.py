"""
DeltaPace :: Central Configuration
==================================

This module is the single source of truth for *how* and *where* the platform
runs. It groups three kinds of settings:

1. **Paths & environment** -- cache location, OpenAI credentials, feature flags.
2. **Race parameters** (`RaceContext`) -- the dynamic year / event / session
   triple that every downstream component (ingest, features, ML, agents) reads.
3. **Physical constants** (`F1Physics`) -- domain knowledge such as maximum
   fuel load and its lap-time penalty, used by the feature-engineering layer.

Everything is expressed with `dataclasses` so components can be *constructed*
with explicit parameters (great for testing) while still exposing sensible
project-wide defaults loaded from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# `python-dotenv` is optional at import time; we degrade gracefully so that the
# module can be imported in environments where it is not installed yet.
try:
    from dotenv import load_dotenv

    load_dotenv()  # Populates os.environ from a local .env file if present.
except ImportError:  # pragma: no cover - convenience only
    pass


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# Resolve paths relative to the repository root (two levels up from this file:
# src/config.py -> src -> <repo root>). This keeps the code portable regardless
# of the current working directory the process was launched from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"          # FastF1 API cache lives here.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"     # Generated Markdown reports.


def _ensure_dirs() -> None:
    """Create runtime directories on first import so callers never worry about it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


# ---------------------------------------------------------------------------
# LLM / OpenAI settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMSettings:
    """Configuration for the Writer Agent's language model.

    When ``OPENAI_API_KEY`` is absent we flip ``mock_mode`` on so the whole
    workflow can still be exercised end-to-end (in CI or offline) without ever
    contacting the network.
    """

    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("DELTAPACE_LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.3

    @property
    def mock_mode(self) -> bool:
        """True when we have no key and must synthesise the report locally."""
        return not self.api_key


# ---------------------------------------------------------------------------
# Race context -- the dynamic (year, event, session) selector
# ---------------------------------------------------------------------------
@dataclass
class RaceContext:
    """Identifies *which* piece of F1 data we are analysing.

    FastF1 accepts a flexible event identifier: an integer round number, a
    country/circuit name, or an official event name. We keep it as ``str | int``
    so the platform can dynamically handle *any* race the user requests.

    Attributes
    ----------
    year:
        Championship season, e.g. ``2023``.
    grand_prix:
        Event identifier accepted by FastF1 (e.g. ``"Monza"``, ``"Bahrain"``,
        or a round number like ``9``).
    session:
        Session code -- ``"R"`` (race), ``"Q"`` (qualifying), ``"FP1"`` etc.
    """

    year: int = 2023
    grand_prix: str | int = "Bahrain"
    session: str = "R"

    def label(self) -> str:
        """Human-readable identifier used in logs, cache keys, and reports."""
        return f"{self.year} {self.grand_prix} [{self.session}]"


# ---------------------------------------------------------------------------
# F1 physical constants used by feature engineering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class F1Physics:
    """Domain constants that turn raw timing data into physically meaningful
    features.

    These are deliberately approximate, regulation-era values. They are grouped
    here so that assumptions are auditable and easy to tune in one place rather
    than being scattered as magic numbers throughout the codebase.
    """

    # Maximum permitted race fuel load under current F1 regulations (kg).
    max_fuel_kg: float = 110.0

    # Lap-time penalty per kilogram of fuel carried (seconds/kg). A heavier car
    # is slower; as fuel burns off the car naturally speeds up. ~0.03s/kg is the
    # widely used rule-of-thumb figure.
    fuel_time_penalty_s_per_kg: float = 0.03

    # Reserve fuel that must remain at the flag (kg) for post-race sampling.
    fuel_reserve_kg: float = 1.0

    # Ordered tyre compound hardness (soft -> hard). Used to rank/compare stints.
    compound_hardness: tuple[str, ...] = (
        "SOFT",
        "MEDIUM",
        "HARD",
        "INTERMEDIATE",
        "WET",
    )


# ---------------------------------------------------------------------------
# Convenient module-level singletons
# ---------------------------------------------------------------------------
# Most callers just want the defaults; advanced callers instantiate their own.
LLM_SETTINGS = LLMSettings()
PHYSICS = F1Physics()
DEFAULT_RACE = RaceContext()
