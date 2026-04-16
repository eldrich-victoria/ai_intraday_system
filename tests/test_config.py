# -*- coding: utf-8 -*-
"""Tests for src.config module."""

from pathlib import Path
from zoneinfo import ZoneInfo


def test_ist_timezone():
    """IST should be Asia/Kolkata."""
    from src.config import IST
    assert IST == ZoneInfo("Asia/Kolkata")


def test_directories_exist_after_ensure(tmp_path, monkeypatch):
    """ensure_directories should create data, models, and logs dirs."""
    monkeypatch.setattr("src.config.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("src.config.LOG_DIR", tmp_path / "logs")

    from src.config import ensure_directories
    ensure_directories()

    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_config_defaults():
    """Verify sensible defaults for key configuration values."""
    from src.config import (
        BROKERAGE_PER_TRADE_INR,
        BROKERAGE_PER_LEG_INR,
        INITIAL_CAPITAL,
        MAX_OPEN_POSITIONS,
        ML_CONFIDENCE_THRESHOLD,
        RISK_PER_TRADE_PCT,
        SLIPPAGE_PCT,
    )
    assert INITIAL_CAPITAL > 0
    assert BROKERAGE_PER_TRADE_INR == 40.0
    assert BROKERAGE_PER_LEG_INR == 20.0
    assert 0 < RISK_PER_TRADE_PCT <= 0.05
    assert 0 <= SLIPPAGE_PCT < 0.01
    assert ML_CONFIDENCE_THRESHOLD > 0
    assert MAX_OPEN_POSITIONS >= 1


def test_nse_holidays_not_empty():
    """NSE holiday set should contain entries."""
    from src.config import NSE_HOLIDAYS
    assert len(NSE_HOLIDAYS) > 10


def test_compliance_note_not_empty():
    """SEBI compliance disclaimer should exist."""
    from src.config import COMPLIANCE_NOTE
    assert len(COMPLIANCE_NOTE) > 20
    assert "investment advice" in COMPLIANCE_NOTE.lower() or "paper" in COMPLIANCE_NOTE.lower()
