"""Tests for configuration flag logic (mock mode, S3 gating)."""

from __future__ import annotations

from src.config import AWSSettings, BedrockSettings


def test_bedrock_mock_mode_when_env_set(monkeypatch):
    monkeypatch.setenv("DELTAPACE_MOCK_LLM", "true")
    monkeypatch.delenv("DELTAPACE_CI", raising=False)
    settings = BedrockSettings()
    assert settings.mock_mode is True


def test_bedrock_mock_mode_when_ci_set(monkeypatch):
    monkeypatch.setenv("DELTAPACE_CI", "true")
    monkeypatch.delenv("DELTAPACE_MOCK_LLM", raising=False)
    settings = BedrockSettings()
    assert settings.mock_mode is True


def test_s3_disabled_without_bucket(monkeypatch):
    monkeypatch.delenv("DELTAPACE_S3_BUCKET", raising=False)
    monkeypatch.setenv("DELTAPACE_USE_S3", "true")
    settings = AWSSettings()
    assert settings.use_s3 is False


def test_s3_requires_explicit_flag_when_bucket_set(monkeypatch):
    monkeypatch.setenv("DELTAPACE_S3_BUCKET", "my-test-bucket")
    monkeypatch.setenv("DELTAPACE_USE_S3", "false")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    settings = AWSSettings()
    # Without creds and without DELTAPACE_USE_S3=true, uploads stay off.
    assert settings.use_s3 is False
