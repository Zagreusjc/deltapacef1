"""Tests for bionic Markdown formatting."""

from __future__ import annotations

from src.report.bionic import to_bionic


def test_bionic_bolds_first_half_of_words():
    assert to_bionic("Verstappen") == "**Verst**appen"


def test_bionic_preserves_headers():
    line = "# DeltaPace Report"
    assert to_bionic(line) == line


def test_bionic_preserves_inline_code():
    text = "Use `FuelCorrectedLapTime` column"
    assert "`FuelCorrectedLapTime`" in to_bionic(text)
    assert "**FuelCorrectedLapTime**" not in to_bionic(text)


def test_bionic_preserves_links():
    text = "[Read more](https://example.com)"
    result = to_bionic(text)
    assert "](https://example.com)" in result
    assert "**Read**" in result or "**Re**" in result
