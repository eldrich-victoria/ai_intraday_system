# -*- coding: utf-8 -*-
"""Tests for src.fetch_sheet module."""



def test_normalize_column_name():
    """Column name normalization should handle common variants."""
    from src.fetch_sheet import _normalize_column_name

    assert _normalize_column_name("Symbol") == "symbol"
    assert _normalize_column_name("Buy Price") == "buy_price"
    assert _normalize_column_name("BuyPrice") == "buy_price"
    assert _normalize_column_name("entry price") == "buy_price"
    assert _normalize_column_name("Stop Loss") == "stop_loss"
    assert _normalize_column_name("stoploss") == "stop_loss"
    assert _normalize_column_name("SL") == "stop_loss"
    assert _normalize_column_name("TGT") == "target"
    assert _normalize_column_name(None) == ""


def test_row_hash_deterministic():
    """Same input should produce the same hash."""
    from src.fetch_sheet import _row_hash

    row = {
        "symbol": "RELIANCE",
        "signal": "BUY",
        "buy_price": "2450.0",
        "stop_loss": "2420.0",
        "target": "2520.0",
        "timestamp": "2026-04-12 09:20:00",
    }
    h1 = _row_hash(row)
    h2 = _row_hash(row)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest length.


def test_row_hash_differs_for_different_input():
    """Different rows should produce different hashes."""
    from src.fetch_sheet import _row_hash

    row1 = {"symbol": "RELIANCE", "signal": "BUY", "buy_price": "2450"}
    row2 = {"symbol": "TCS", "signal": "BUY", "buy_price": "3450"}
    assert _row_hash(row1) != _row_hash(row2)


def test_validate_signal_row_buy_valid():
    """Valid BUY signal should pass validation."""
    from src.fetch_sheet import validate_signal_row

    row = {
        "symbol": "RELIANCE",
        "signal": "BUY",
        "buy_price": 2450.0,
        "stop_loss": 2420.0,
        "target": 2520.0,
    }
    assert validate_signal_row(row) is True


def test_validate_signal_row_sell_valid():
    """Valid SELL signal should pass validation."""
    from src.fetch_sheet import validate_signal_row

    row = {
        "symbol": "INFY",
        "signal": "SELL",
        "buy_price": 1520.0,
        "stop_loss": 1550.0,
        "target": 1460.0,
    }
    assert validate_signal_row(row) is True


def test_validate_signal_row_buy_invalid_sl():
    """BUY with stop_loss >= buy_price should fail."""
    from src.fetch_sheet import validate_signal_row

    row = {
        "signal": "BUY",
        "buy_price": 100.0,
        "stop_loss": 110.0,  # SL above entry.
        "target": 120.0,
    }
    assert validate_signal_row(row) is False


def test_validate_signal_row_sell_invalid_sl():
    """SELL with stop_loss <= buy_price should fail."""
    from src.fetch_sheet import validate_signal_row

    row = {
        "signal": "SELL",
        "buy_price": 100.0,
        "stop_loss": 90.0,  # SL below entry for short.
        "target": 80.0,
    }
    assert validate_signal_row(row) is False


def test_validate_signal_row_zero_price():
    """Zero or negative prices should fail."""
    from src.fetch_sheet import validate_signal_row

    row = {
        "signal": "BUY",
        "buy_price": 0.0,
        "stop_loss": -10.0,
        "target": 100.0,
    }
    assert validate_signal_row(row) is False


def test_load_mock_signals_csv(tmp_path):
    """Mock CSV loading should produce correct DataFrame."""
    csv_content = (
        "Symbol,Signal,Buy Price,Stop Loss,Target,Timestamp\n"
        "RELIANCE,BUY,2450.0,2420.0,2520.0,2026-04-12 09:20:00\n"
        "TCS,BUY,3450.0,3400.0,3550.0,2026-04-12 09:25:00\n"
    )
    csv_path = tmp_path / "test_signals.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    from src.fetch_sheet import load_mock_signals_csv

    df = load_mock_signals_csv(str(csv_path))
    assert len(df) == 2
    assert "symbol" in df.columns
    assert "row_hash" in df.columns
    assert df["symbol"].iloc[0] == "RELIANCE"


def test_load_mock_signals_csv_missing_file(tmp_path):
    """Missing file should return empty DataFrame."""
    from src.fetch_sheet import load_mock_signals_csv

    df = load_mock_signals_csv(str(tmp_path / "nonexistent.csv"))
    assert df.empty
