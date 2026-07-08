"""
DeltaPace :: CLI Entry Point
============================

Run the full offline-capable pipeline::

    python -m src.run --year 2023 --gp Monza --session R

Produces a Markdown report and PNG charts under ``reports/{slug}/``, optionally
uploading artifacts to S3 when configured.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.config import (
    AWS_SETTINGS,
    BEDROCK_SETTINGS,
    DEFAULT_RACE,
    REPORTS_DIR,
    RaceContext,
    _slugify,
)
from src.pipeline.analysis import AnalysisPipeline
from src.report.charts import generate_all_charts
from src.report.writer import ReportWriter
from src.storage.s3 import S3ArtifactStore

logger = logging.getLogger(__name__)


def _env_flag(key: str, default: bool = True) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _race_output_dir(race: RaceContext) -> Path:
    slug = f"{race.year}-{_slugify(race.grand_prix)}-{race.session}"
    return REPORTS_DIR / slug


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeltaPace — F1 tyre-strategy analysis pipeline",
    )
    parser.add_argument("--year", type=int, default=DEFAULT_RACE.year, help="Season year")
    parser.add_argument("--gp", default=str(DEFAULT_RACE.grand_prix), help="Grand Prix name or round")
    parser.add_argument("--session", default=DEFAULT_RACE.session, help="Session code (R, Q, FP1, …)")
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload artifacts and report to S3 when configured",
    )
    parser.add_argument(
        "--bionic",
        action="store_true",
        help="Apply bionic bold formatting to the Markdown report",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run analysis → report → charts; print summary paths."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    gp: str | int = args.gp
    if isinstance(gp, str) and gp.isdigit():
        gp = int(gp)

    race = RaceContext(year=args.year, grand_prix=gp, session=args.session)
    out_dir = _race_output_dir(race)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = S3ArtifactStore()
    upload = args.upload and store.enabled

    logger.info("Running DeltaPace for %s", race.label())
    result = AnalysisPipeline().run(race, upload=upload, store=store)

    report_md = ReportWriter().generate(result, bionic=args.bionic)
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Wrote report %s", report_path)

    s3_report_uri = ""
    if upload:
        key = f"{AWS_SETTINGS.reports_prefix(race)}/report.md"
        if store.upload_report(race, report_md):
            s3_report_uri = store.s3_uri(key)

    chart_paths: list[Path] = []
    charts_enabled = _env_flag("DELTAPACE_ENABLE_CHARTS", default=True) and not args.no_charts
    if charts_enabled:
        charts_dir = out_dir / "charts"
        chart_paths = generate_all_charts(
            result.enriched,
            result.degradation,
            result.team_summary,
            charts_dir,
        )
        if upload:
            reports_prefix = AWS_SETTINGS.reports_prefix(race)
            for chart in chart_paths:
                key = f"{reports_prefix}/charts/{chart.name}"
                store.upload_local_file(chart, key, content_type="image/png")

    print("")
    print("=" * 60)
    print(f"  DeltaPace — {race.label()}")
    print("=" * 60)
    print(f"  Report (local):  {report_path.resolve()}")
    if chart_paths:
        print(f"  Charts (local):  {chart_paths[0].parent.resolve()} ({len(chart_paths)} file(s))")
    else:
        print("  Charts:          skipped or none generated")
    print(f"  Bedrock mode:    {'mock (offline)' if BEDROCK_SETTINGS.mock_mode else 'live'}")
    print(f"  S3 upload:       {'enabled' if upload else 'disabled'}")
    if upload and s3_report_uri:
        print(f"  Report (S3):     {s3_report_uri}")
        processed_prefix = AWS_SETTINGS.processed_prefix(race)
        print(f"  Artifacts (S3):  s3://{AWS_SETTINGS.s3_bucket}/{processed_prefix}/")
    print("=" * 60)
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
