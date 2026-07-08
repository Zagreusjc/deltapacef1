"""
DeltaPace :: Race Charts
========================

Pure functions that take analysis DataFrames and write PNG figures to disk.
Each function returns a list of saved file paths (empty when input is blank).

Designed for reuse by :mod:`src.run` today and a future Streamlit dashboard
tomorrow — no side effects beyond creating image files in ``out_dir``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for CI/servers
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_team_color(color: str | None, default: str = "#888888") -> str:
    """Normalise FastF1 hex colours (may omit leading ``#``)."""
    if color is None or (isinstance(color, float) and pd.isna(color)):
        return default
    text = str(color).strip()
    if not text:
        return default
    return text if text.startswith("#") else f"#{text}"


def _ensure_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_degradation_bars(degradation: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Horizontal bar chart: degradation rate per driver, coloured by team."""
    if degradation.empty or "DegradationPerLap" not in degradation.columns:
        return []

    out_dir = _ensure_dir(out_dir)
    df = degradation.copy()
    df["Label"] = df.apply(
        lambda r: f"{r.get('FullName') or r['Driver']} ({r.get('Compound', '?')})",
        axis=1,
    )
    df = df.sort_values("DegradationPerLap", ascending=True)

    colors = [_parse_team_color(c) for c in df.get("TeamColor", pd.Series(["#888"] * len(df)))]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    ax.barh(df["Label"], df["DegradationPerLap"], color=colors)
    ax.set_xlabel("Seconds lost per lap of tyre age")
    ax.set_title("Tyre Degradation by Driver")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()

    path = out_dir / "degradation_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart %s", path)
    return [path]


def plot_team_pace(team_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Bar chart of median fuel-corrected pace by team."""
    if team_summary.empty or "MedianPaceSeconds" not in team_summary.columns:
        return []

    out_dir = _ensure_dir(out_dir)
    df = team_summary.dropna(subset=["MedianPaceSeconds", "TeamName"]).sort_values(
        "MedianPaceSeconds"
    )

    colors = [_parse_team_color(c) for c in df.get("TeamColor", pd.Series(["#888"] * len(df)))]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    ax.barh(df["TeamName"], df["MedianPaceSeconds"], color=colors)
    ax.set_xlabel("Median fuel-corrected lap time (seconds) — lower is faster")
    ax.set_title("Team Pace Comparison")
    fig.tight_layout()

    path = out_dir / "team_pace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart %s", path)
    return [path]


def plot_tyre_curves(enriched: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Scatter: fuel-corrected lap time vs tyre age, one subplot per compound."""
    needed = {"FuelCorrectedLapTime", "TyreAge", "Compound", "Driver"}
    if enriched.empty or not needed.issubset(enriched.columns):
        return []

    out_dir = _ensure_dir(out_dir)
    df = enriched.dropna(subset=["FuelCorrectedLapTime", "TyreAge", "Compound"])
    df = df[~df["Compound"].astype(str).str.upper().isin({"INTERMEDIATE", "WET"})]

    if df.empty:
        return []

    compounds = sorted(df["Compound"].astype(str).unique())
    saved: list[Path] = []

    for compound in compounds:
        subset = df[df["Compound"].astype(str) == compound]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for driver, group in subset.groupby("Driver"):
            color = _parse_team_color(group["TeamColor"].iloc[0] if "TeamColor" in group.columns else None)
            label = group["FullName"].iloc[0] if "FullName" in group.columns and pd.notna(group["FullName"].iloc[0]) else driver
            ax.scatter(group["TyreAge"], group["FuelCorrectedLapTime"], s=12, alpha=0.6, color=color, label=label)

        ax.set_xlabel("Tyre age (laps on this set)")
        ax.set_ylabel("Fuel-corrected lap time (seconds)")
        ax.set_title(f"Tyre Wear Curves — {compound}")
        if len(subset["Driver"].unique()) <= 12:
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()

        safe_compound = str(compound).lower().replace(" ", "_")
        path = out_dir / f"tyre_curves_{safe_compound}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved chart %s", path)
        saved.append(path)

    return saved


def generate_all_charts(
    enriched: pd.DataFrame,
    degradation: pd.DataFrame,
    team_summary: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """Convenience wrapper — run all chart functions and return combined paths."""
    paths: list[Path] = []
    paths.extend(plot_degradation_bars(degradation, out_dir))
    paths.extend(plot_team_pace(team_summary, out_dir))
    paths.extend(plot_tyre_curves(enriched, out_dir))
    return paths
