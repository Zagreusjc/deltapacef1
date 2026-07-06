"""
DeltaPace :: Central Configuration
==================================

This module is the single source of truth for *how* and *where* the platform
runs. It groups four kinds of settings:

1. **Paths (local staging)** -- FastF1 cache and report output directories on
   the developer machine. FastF1 *requires* a local filesystem cache; S3 cannot
   replace it directly (see ``AWSSettings`` for durable cloud storage).
2. **AWS** (`AWSSettings`) -- S3 bucket, key prefixes, and credential detection
   for uploading processed artifacts and reports.
3. **Race parameters** (`RaceContext`) -- the dynamic year / event / session
   triple that every downstream component (ingest, features, ML, report) reads.
4. **Domain & LLM** -- ``F1Physics`` constants, ``MLSettings`` for regression
   thresholds, and ``BedrockSettings`` for the optional narrative report step
   (Amazon Bedrock Converse API).

Everything is expressed with ``dataclasses`` so components can be *constructed*
with explicit parameters (great for testing) while still exposing sensible
project-wide defaults loaded from the environment.

Production note: secrets (AWS keys, model IDs) should be injected via
environment variables — e.g. populated from AWS Secrets Manager by your
deployment tooling *before* the Python process starts. This module does **not**
call Secrets Manager at import time.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# `python-dotenv` is optional at import time; we degrade gracefully so that the
# module can be imported in environments where it is not installed yet.
try:
    from dotenv import load_dotenv

    load_dotenv()  # Populates os.environ from a local .env file if present.
except ImportError:  # pragma: no cover - convenience only
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project paths (local staging — FastF1 cache MUST stay on disk)
# ---------------------------------------------------------------------------
# Resolve paths relative to the repository root (two levels up from this file:
# src/config.py -> src -> <repo root>). This keeps the code portable regardless
# of the current working directory the process was launched from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"          # Local FastF1 API cache staging.
REPORTS_DIR: Path = PROJECT_ROOT / "reports"     # Local report staging before S3 upload.


def _ensure_dirs() -> None:
    """Create runtime directories on first import so callers never worry about it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


# ---------------------------------------------------------------------------
# AWS credential helpers (lazy — never call Secrets Manager here)
# ---------------------------------------------------------------------------
def _env_with_default(key: str, default: str) -> str:
    """Return the first non-empty environment value for *key*, else *default*."""
    return os.getenv(key) or default


def _env_flag(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable (``true`` / ``1`` / ``yes``)."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _aws_credentials_available() -> bool:
    """Best-effort check that boto3 can authenticate without making API calls.

    Returns True when explicit access keys are set, or when the default credential
    chain (shared config, SSO, instance profile) resolves a session.
    """
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True

    try:
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None and credentials.access_key is not None
    except Exception:  # noqa: BLE001 - credential probe only
        return False


def _slugify(value: str | int) -> str:
    """Turn a grand-prix identifier into a safe S3 path segment."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


# ---------------------------------------------------------------------------
# AWS / S3 settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AWSSettings:
    """Configuration for durable artifact storage in Amazon S3.

    **Local vs cloud split**

    * ``DATA_DIR`` (local) — FastF1 ``Cache.enable_cache()`` reads/writes here
      during a run. This directory cannot be an S3 mount.
    * S3 (this class) — processed DataFrames, reports, and optional cache
      snapshots are uploaded *after* local processing for team-wide reuse.

    Key layout (under ``s3://{bucket}/{s3_prefix}/``)::

        processed/{year}/{gp_slug}/{session}/   # laps, weather, features
        reports/{year}/{gp_slug}/{session}/     # Markdown reports
        cache/                                  # optional FastF1 cache sync
    """

    region: str = field(
        default_factory=lambda: _env_with_default(
            "AWS_REGION", _env_with_default("AWS_DEFAULT_REGION", "ap-southeast-1")
        )
    )
    s3_bucket: str | None = field(default_factory=lambda: os.getenv("DELTAPACE_S3_BUCKET"))
    s3_prefix: str = field(
        default_factory=lambda: _env_with_default("DELTAPACE_S3_PREFIX", "deltapace")
    )

    @property
    def use_s3(self) -> bool:
        """True when S3 uploads are enabled and credentials are available.

        Enabled when ``DELTAPACE_S3_BUCKET`` is set **and** either:
        * ``DELTAPACE_USE_S3=true`` is set explicitly, **or**
        * AWS credentials resolve via the default boto3 chain.
        """
        if not self.s3_bucket:
            return False
        if _env_flag("DELTAPACE_USE_S3"):
            return True
        return _aws_credentials_available()

    def _base(self) -> str:
        """Root prefix inside the bucket (no leading/trailing slashes)."""
        return self.s3_prefix.strip("/")

    def processed_prefix(self, race: "RaceContext") -> str:
        """S3 key prefix for processed session artifacts (parquet/csv)."""
        gp = _slugify(race.grand_prix)
        return f"{self._base()}/processed/{race.year}/{gp}/{race.session}"

    def reports_prefix(self, race: "RaceContext") -> str:
        """S3 key prefix for generated Markdown reports."""
        gp = _slugify(race.grand_prix)
        return f"{self._base()}/reports/{race.year}/{gp}/{race.session}"

    def cache_prefix(self) -> str:
        """S3 key prefix for optional FastF1 cache blob sync (best-effort)."""
        return f"{self._base()}/cache"


# ---------------------------------------------------------------------------
# Amazon Bedrock settings (replaces OpenAI / Lang stack)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BedrockSettings:
    """Configuration for the narrative report step via Amazon Bedrock.

    Uses the ``bedrock-runtime`` Converse API. When ``mock_mode`` is True the
    pipeline skips the network call and emits a local template report instead
    (useful for CI and offline development).

    ``mock_mode`` is True when:
    * ``DELTAPACE_MOCK_LLM=true``, or
    * ``DELTAPACE_CI=true`` (explicit CI flag), or
    * AWS credentials are unavailable (Bedrock client cannot be initialised).
    """

    model_id: str = field(
        default_factory=lambda: _env_with_default(
            "DELTAPACE_BEDROCK_MODEL_ID",
            "anthropic.claude-3-haiku-20240307-v1:0",
        )
    )
    region: str = field(
        default_factory=lambda: _env_with_default(
            "AWS_REGION", _env_with_default("AWS_DEFAULT_REGION", "ap-southeast-1")
        )
    )
    temperature: float = 0.3

    @property
    def mock_mode(self) -> bool:
        """True when Bedrock should be bypassed in favour of a local stub."""
        if _env_flag("DELTAPACE_MOCK_LLM") or _env_flag("DELTAPACE_CI"):
            return True
        return not _aws_credentials_available()


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
# Machine-learning settings (tyre degradation regression)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLSettings:
    """Thresholds and defaults for the scikit-learn degradation model.

    These guard against fitting noisy regressions on stints that are too short
    (e.g. a single out-lap before a pit stop) or on laps flagged as inaccurate
    by FastF1.
    """

    # Minimum laps on a stint before we attempt a per-(driver, compound) fit.
    min_stint_laps: int = 5

    # Minimum total samples across all stints for a driver+compound combination.
    min_samples_per_group: int = 8

    # Exclude laps where FastF1 marks IsAccurate == False (when column exists).
    require_accurate_laps: bool = True

    # Future Phase: pit-stop loss assumption (seconds) for crossover modelling.
    pit_stop_loss_seconds: float = 22.0


# ---------------------------------------------------------------------------
# Convenient module-level singletons
# ---------------------------------------------------------------------------
# Most callers just want the defaults; advanced callers instantiate their own.
AWS_SETTINGS = AWSSettings()
BEDROCK_SETTINGS = BedrockSettings()
PHYSICS = F1Physics()
ML_SETTINGS = MLSettings()
DEFAULT_RACE = RaceContext()
