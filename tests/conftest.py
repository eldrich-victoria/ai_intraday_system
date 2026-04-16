# -*- coding: utf-8 -*-
"""Shared pytest fixtures for the AI Intraday Trading Tester test suite."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """
    Isolate every test from the real environment.

    - Points DB_PATH to a temporary directory.
    - Clears Telegram and Google credentials so no real API calls happen.
    - Resets the thread-local DB connection.
    """
    db_path = tmp_path / "test_trades.db"
    monkeypatch.setattr("src.config.DB_PATH", db_path)
    import src.db
    monkeypatch.setattr(src.db, "DB_PATH", db_path)

    monkeypatch.setattr("src.config.DATA_DIR", tmp_path)
    
    # Mock Models Directory
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    import src.features_ml
    monkeypatch.setattr(src.features_ml, "MODELS_DIR", tmp_path)
    
    mock_model = tmp_path / "rf_model_GLOBAL.pkl"
    import src.config
    monkeypatch.setattr(src.config, "MODEL_PATH", mock_model)

    monkeypatch.setattr("src.config.LOG_DIR", tmp_path)
    monkeypatch.setattr("src.config.LOCK_FILE", tmp_path / "test.lock")

    # Prevent real API calls.
    monkeypatch.setenv("BOT_TOKEN", "")
    monkeypatch.setenv("CHAT_ID", "")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    monkeypatch.setenv("USE_GOOGLE_SHEET", "0")

    # Reset thread-local DB connection so each test gets a fresh one.
    from src.db import close_connection
    close_connection()
    yield
    close_connection()


@pytest.fixture
def sample_signals_df():
    """Return a small DataFrame of valid BUY/SELL signals."""
    return pd.DataFrame({
        "symbol": ["RELIANCE", "TCS", "INFY"],
        "signal": ["BUY", "BUY", "SELL"],
        "buy_price": [2450.0, 3450.0, 1520.0],
        "stop_loss": [2420.0, 3400.0, 1550.0],
        "target": [2520.0, 3550.0, 1460.0],
        "timestamp": [
            "2026-04-12 09:20:00",
            "2026-04-12 09:25:00",
            "2026-04-12 09:30:00",
        ],
        "row_hash": ["hash_rel", "hash_tcs", "hash_infy"],
    })


@pytest.fixture
def synthetic_ohlcv():
    """Return synthetic OHLCV data for testing."""
    from src.market_data import synthetic_ohlcv as _synth
    return _synth(n=200, seed=42)


@pytest.fixture
def initialized_db():
    """Ensure the test database is initialized with schema."""
    from src.db import ensure_database
    ensure_database()
    return True
