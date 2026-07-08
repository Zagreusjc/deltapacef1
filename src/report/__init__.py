"""Report generation: narrative writer, charts, and bionic formatting."""

from src.report.bionic import to_bionic
from src.report.charts import generate_all_charts, plot_degradation_bars, plot_team_pace, plot_tyre_curves
from src.report.writer import ReportWriter

__all__ = [
    "ReportWriter",
    "to_bionic",
    "plot_degradation_bars",
    "plot_team_pace",
    "plot_tyre_curves",
    "generate_all_charts",
]
