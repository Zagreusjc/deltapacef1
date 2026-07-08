"""
DeltaPace :: Narrative Report Writer
====================================

``ReportWriter`` turns an :class:`~src.pipeline.analysis.AnalysisResult` into a
blog-worthy Markdown document. In **mock mode** (default offline) it assembles a
local template with fan-friendly explanations. When AWS credentials and Bedrock
are available it calls the Converse API with a compact stats summary — never raw
lap telemetry.
"""

from __future__ import annotations

import json
import logging

from src.config import BEDROCK_SETTINGS, BedrockSettings
from src.pipeline.analysis import AnalysisResult
from src.pipeline.identity import IdentityEnricher
from src.report.bionic import to_bionic

logger = logging.getLogger(__name__)


class ReportWriter:
    """Generate a Markdown race report from analysis outputs."""

    def __init__(self, settings: BedrockSettings = BEDROCK_SETTINGS) -> None:
        self.settings = settings

    def generate(self, result: AnalysisResult, *, bionic: bool = False) -> str:
        """Return Markdown for *result*, optionally bionic-formatted."""
        if self.settings.mock_mode:
            report = self._build_mock_report(result)
        else:
            try:
                report = self._generate_via_bedrock(result)
            except Exception as exc:  # noqa: BLE001 — fall back offline
                logger.warning("Bedrock report failed (%s); using mock template.", exc)
                report = self._build_mock_report(result)

        if bionic:
            report = to_bionic(report)
        return report

    # ------------------------------------------------------------------
    # Mock template (offline default)
    # ------------------------------------------------------------------
    def _build_mock_report(self, result: AnalysisResult) -> str:
        """Assemble a fan-readable report without any network calls."""
        race_label = result.race.label()
        deg = result.degradation
        teams = result.team_summary

        lines: list[str] = [
            f"# DeltaPace Race Report — {race_label}",
            "",
            "> *Auto-generated analysis. Tyre degradation numbers show how many "
            "seconds each driver loses per lap as their tyres age — like a pencil "
            "getting blunt: every lap takes a little longer to complete.*",
            "",
            "## The Headline",
            "",
        ]

        if not deg.empty:
            best = deg.loc[deg["DegradationPerLap"].idxmin()]
            worst = deg.loc[deg["DegradationPerLap"].idxmax()]
            best_name = best.get("FullName") or best["Driver"]
            worst_name = worst.get("FullName") or worst["Driver"]
            lines.extend(
                [
                    f"- **Best tyre life:** {best_name} on **{best['Compound']}** "
                    f"— only **{best['DegradationPerLap']:.3f}s** lost per lap of wear.",
                    f"- **Hardest on tyres:** {worst_name} on **{worst['Compound']}** "
                    f"— **{worst['DegradationPerLap']:.3f}s** lost per lap "
                    "(the rising line on the chart means the rubber is fading fast).",
                    "",
                ]
            )
        else:
            lines.extend(["- Insufficient dry-tyre data to rank degradation this session.", ""])

        lines.extend(["## Team Pace Ranking", ""])
        if not teams.empty and "MedianPaceSeconds" in teams.columns:
            lines.append(
                "*Lower median pace = faster car. Think of it as the team's "
                "typical lap speed once fuel weight is stripped out.*"
            )
            lines.append("")
            for _, row in teams.head(10).iterrows():
                name = row.get("TeamName", "Unknown")
                pace = row.get("MedianPaceSeconds", float("nan"))
                rank = row.get("PaceRank", "?")
                lines.append(f"{rank}. **{name}** — median pace **{pace:.3f}s**")
            lines.append("")
        else:
            lines.append("_No team pace data available._")
            lines.append("")

        lines.extend(["## Tyre Degradation by Team", ""])
        if not teams.empty and "MedianDegradationPerLap" in teams.columns:
            lines.append(
                "*Lower degradation = the team looked after their tyres better. "
                "A hot track or aggressive driving pushes this number up.*"
            )
            lines.append("")
            deg_sorted = teams.dropna(subset=["MedianDegradationPerLap"]).sort_values(
                "MedianDegradationPerLap"
            )
            for _, row in deg_sorted.head(10).iterrows():
                name = row.get("TeamName", "Unknown")
                rate = row.get("MedianDegradationPerLap", float("nan"))
                rank = row.get("DegradationRank", "?")
                lines.append(f"{rank}. **{name}** — **{rate:.3f}s/lap** of wear")
            lines.append("")
        else:
            lines.append("_No team degradation summary available._")
            lines.append("")

        # Teammate battles
        identity = IdentityEnricher()
        teammates = identity.compare_teammates(result.enriched, deg)
        lines.extend(["## Teammate Comparison", ""])
        if not teammates.empty:
            lines.append(
                "*How much faster is the quicker driver in each garage? "
                "Pace delta is straight lap-time gap; degradation delta shows "
                "who wears tyres faster.*"
            )
            lines.append("")
            for _, row in teammates.iterrows():
                lines.append(
                    f"- **{row['TeamName']}:** {row['DriverA']} vs {row['DriverB']} — "
                    f"pace gap **{row['PaceDeltaSeconds']:.3f}s**"
                    + (
                        f", degradation gap **{row['DegradationDelta']:.3f}s/lap**"
                        if row.get("DegradationDelta") is not None
                        else ""
                    )
                )
            lines.append("")
        else:
            lines.append("_No teammate pairs to compare._")
            lines.append("")

        lines.extend(
            [
                "## What to Look For in the Charts",
                "",
                "1. **Degradation bars** — taller bars mean tyres fade quicker; "
                "team colours match the real liveries on track.",
                "2. **Tyre curves** — an upward slope is wear in action: each lap "
                "on the same set costs a bit more time.",
                "3. **Team pace** — who had the raw one-lap speed once fuel is "
                "accounted for.",
                "",
                "---",
                f"*Report generated locally (Bedrock mock mode) for {race_label}.*",
            ]
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Bedrock Converse API
    # ------------------------------------------------------------------
    def _build_prompt(self, result: AnalysisResult) -> str:
        """Compact stats summary for the LLM — no raw lap rows."""
        summary = {
            "race": result.race.label(),
            "degradation_top5": result.degradation.head(5).to_dict(orient="records")
            if not result.degradation.empty
            else [],
            "degradation_bottom5": result.degradation.tail(5).to_dict(orient="records")
            if not result.degradation.empty
            else [],
            "team_summary": result.team_summary.head(10).to_dict(orient="records")
            if not result.team_summary.empty
            else [],
            "lap_count": len(result.enriched),
            "driver_count": int(result.enriched["Driver"].nunique())
            if "Driver" in result.enriched.columns
            else 0,
        }
        stats_json = json.dumps(summary, indent=2, default=str)
        return (
            "You are a Formula 1 journalist writing for casual fans who are NOT "
            "data scientists. Using ONLY the JSON stats below, write a engaging "
            "Markdown blog post about this race's tyre strategy story. Explain "
            "degradation in plain English (use analogies). Include sections: "
            "Headline, Key Findings, Team Battle, Teammate Duels, What the Charts "
            "Mean. Do not invent numbers not present in the data.\n\n"
            f"```json\n{stats_json}\n```"
        )

    def _generate_via_bedrock(self, result: AnalysisResult) -> str:
        """Call Amazon Bedrock Converse and return the model's Markdown."""
        import boto3

        client = boto3.client("bedrock-runtime", region_name=self.settings.region)
        prompt = self._build_prompt(result)

        response = client.converse(
            modelId=self.settings.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={"temperature": self.settings.temperature},
        )

        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        texts = [block["text"] for block in content_blocks if "text" in block]
        if not texts:
            raise RuntimeError("Bedrock returned empty content")
        return "\n".join(texts)
