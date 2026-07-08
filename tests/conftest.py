"""Pytest configuration — force offline mode for every test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure tests never hit AWS or Bedrock.
os.environ.setdefault("DELTAPACE_MOCK_LLM", "true")
os.environ.setdefault("DELTAPACE_USE_S3", "false")
os.environ.setdefault("DELTAPACE_CI", "true")
