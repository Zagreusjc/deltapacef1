"""
DeltaPace :: S3 Artifact Storage
================================

``S3ArtifactStore`` handles durable read/write of pipeline outputs in Amazon S3.
It is deliberately separate from :class:`~src.pipeline.ingest.SessionLoader`
because **FastF1 requires a local filesystem cache** — S3 cannot serve as the
FastF1 cache directory.

Storage pattern (Phase 2)
-------------------------
1. **Ingest** writes to ``DATA_DIR`` locally via ``fastf1.Cache.enable_cache()``.
2. **Pipeline** processes DataFrames in memory / local staging.
3. **This module** uploads finished artifacts to S3 under prefixes defined by
   :class:`~src.config.AWSSettings`::

       s3://{bucket}/{prefix}/processed/{year}/{gp}/{session}/laps.csv
       s3://{bucket}/{prefix}/reports/{year}/{gp}/{session}/report.md

4. **Optional** — ``sync_cache_to_s3()`` best-effort uploads FastF1 cache blobs
   so teammates can warm their local ``data/`` folder. Failures are logged and
   never block the main pipeline.

When ``AWSSettings.use_s3`` is False (no bucket or no creds), all methods become
no-ops or read/write locally only, so development works offline.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from src.config import AWS_SETTINGS, AWSSettings, RaceContext

logger = logging.getLogger(__name__)


class S3StorageError(RuntimeError):
    """Raised when an S3 operation fails and the caller opted into strict mode."""


class S3ArtifactStore:
    """Upload and download processed DataFrames and reports via Amazon S3.

    Parameters
    ----------
    settings:
        AWS configuration (bucket, prefix, region). Defaults to module singleton.
    strict:
        When True, S3 failures raise :class:`S3StorageError`. When False (default),
        failures are logged and the pipeline continues using local files only.
    """

    # Standard artifact names within a session prefix.
    ARTIFACT_LAPS = "laps.csv"
    ARTIFACT_WEATHER = "weather.csv"
    ARTIFACT_FEATURES = "features.csv"
    ARTIFACT_REPORT = "report.md"

    def __init__(
        self,
        settings: AWSSettings = AWS_SETTINGS,
        strict: bool = False,
    ) -> None:
        self.settings = settings
        self.strict = strict
        self._client = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """True when S3 operations should be attempted."""
        return self.settings.use_s3

    def _get_client(self):
        """Lazy-create the boto3 S3 client (avoids import-time AWS calls)."""
        if self._client is not None:
            return self._client
        import boto3

        self._client = boto3.client("s3", region_name=self.settings.region)
        return self._client

    def _handle_error(self, action: str, exc: Exception) -> None:
        """Log or raise depending on ``strict`` mode."""
        msg = f"S3 {action} failed: {exc}"
        if self.strict:
            raise S3StorageError(msg) from exc
        logger.warning(msg)

    # ------------------------------------------------------------------
    # DataFrame I/O
    # ------------------------------------------------------------------
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        key: str,
        *,
        format: str = "csv",
    ) -> bool:
        """Upload a DataFrame to ``s3://{bucket}/{key}``.

        Returns True on success, False when S3 is disabled or upload fails.
        """
        if not self.enabled or not self.settings.s3_bucket:
            logger.debug("S3 disabled; skipping upload of %s", key)
            return False

        try:
            body = self._serialize_dataframe(df, format)
            self._get_client().put_object(
                Bucket=self.settings.s3_bucket,
                Key=key,
                Body=body,
                ContentType="text/csv" if format == "csv" else "application/octet-stream",
            )
            logger.info("Uploaded s3://%s/%s", self.settings.s3_bucket, key)
            return True
        except Exception as exc:  # noqa: BLE001
            self._handle_error(f"upload {key}", exc)
            return False

    def download_dataframe(
        self,
        key: str,
        *,
        format: str = "csv",
    ) -> pd.DataFrame | None:
        """Download a DataFrame from S3. Returns None when missing or disabled."""
        if not self.enabled or not self.settings.s3_bucket:
            return None

        try:
            response = self._get_client().get_object(
                Bucket=self.settings.s3_bucket,
                Key=key,
            )
            body = response["Body"].read()
            return self._deserialize_dataframe(body, format)
        except Exception as exc:  # noqa: BLE001
            # ClientError with Code 'NoSuchKey' means the object does not exist.
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in {"NoSuchKey", "404"}:
                logger.debug("S3 object not found: %s", key)
                return None
            self._handle_error(f"download {key}", exc)
            return None

    @staticmethod
    def _serialize_dataframe(df: pd.DataFrame, fmt: str) -> bytes:
        buffer = io.BytesIO()
        if fmt == "csv":
            df.to_csv(buffer, index=False)
        elif fmt == "parquet":
            df.to_parquet(buffer, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        return buffer.getvalue()

    @staticmethod
    def _deserialize_dataframe(body: bytes, fmt: str) -> pd.DataFrame:
        buffer = io.BytesIO(body)
        if fmt == "csv":
            return pd.read_csv(buffer)
        if fmt == "parquet":
            return pd.read_parquet(buffer)
        raise ValueError(f"Unsupported format: {fmt}")

    # ------------------------------------------------------------------
    # Session-level convenience methods
    # ------------------------------------------------------------------
    def upload_session_artifacts(
        self,
        race: RaceContext,
        *,
        laps: pd.DataFrame | None = None,
        weather: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
    ) -> dict[str, bool]:
        """Upload processed session DataFrames under the standard key layout."""
        prefix = self.settings.processed_prefix(race)
        results: dict[str, bool] = {}

        if laps is not None:
            results["laps"] = self.upload_dataframe(
                laps, f"{prefix}/{self.ARTIFACT_LAPS}"
            )
        if weather is not None:
            results["weather"] = self.upload_dataframe(
                weather, f"{prefix}/{self.ARTIFACT_WEATHER}"
            )
        if features is not None:
            results["features"] = self.upload_dataframe(
                features, f"{prefix}/{self.ARTIFACT_FEATURES}"
            )
        return results

    def upload_report(self, race: RaceContext, content: str, filename: str = "report.md") -> bool:
        """Upload a Markdown report string to the reports prefix."""
        if not self.enabled or not self.settings.s3_bucket:
            return False

        key = f"{self.settings.reports_prefix(race)}/{filename}"
        try:
            self._get_client().put_object(
                Bucket=self.settings.s3_bucket,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType="text/markdown",
            )
            logger.info("Uploaded report s3://%s/%s", self.settings.s3_bucket, key)
            return True
        except Exception as exc:  # noqa: BLE001
            self._handle_error(f"upload report {key}", exc)
            return False

    # ------------------------------------------------------------------
    # Optional FastF1 cache sync (best-effort)
    # ------------------------------------------------------------------
    def sync_cache_to_s3(self, local_cache_dir: Path) -> int:
        """Best-effort upload of local FastF1 cache files to S3.

        Returns the number of files successfully uploaded. Failures are logged
        and never raise (unless ``strict=True``).
        """
        if not self.enabled or not self.settings.s3_bucket:
            return 0

        cache_dir = Path(local_cache_dir)
        if not cache_dir.exists():
            return 0

        prefix = self.settings.cache_prefix()
        uploaded = 0

        for path in cache_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(cache_dir).as_posix()
            key = f"{prefix}/{relative}"
            try:
                self._get_client().upload_file(
                    str(path),
                    self.settings.s3_bucket,
                    key,
                )
                uploaded += 1
            except Exception as exc:  # noqa: BLE001
                self._handle_error(f"sync cache {relative}", exc)

        if uploaded:
            logger.info("Synced %d cache file(s) to s3://%s/%s/", uploaded, self.settings.s3_bucket, prefix)
        return uploaded
