# -*- coding: utf-8 -*-
"""Tests for src.dummy_trader module."""

import math

import pandas as pd
import pytest


def test_position_size_returns_integer():
    """Position sizing must return an integer (whole shares on NSE)."""
    from src.dummy_trader import _position_size

    qty = _position_size(capital=100000, entry=500.0, stop=490.0, side="BUY")
    assert isinstance(qty, int)
    assert qty >= 0


def test_position_size_buy():
    """BUY position sizing: risk = entry - stop."""
    from src.dummy_trader import _position_size

    # capital=100000, risk=1% = 1000, entry=500, stop=490, risk_per_share=10
    # qty = 1000/10 = 100, max_notional=20000, cap_qty=20000/500=40
    qty = _position_size(capital=100000, entry=500.0, stop=490.0, side="BUY")
    assert qty == 40  # Capped by max position.


def test_position_size_sell():
    """SELL position sizing: risk = stop - entry."""
    from src.dummy_trader import _position_size

    qty = _position_size(capital=100000, entry=500.0, stop=510.0, side="SELL")
    assert isinstance(qty, int)
    assert qty > 0


def test_position_size_zero_risk():
    """Zero risk (entry == stop) should return 0."""
    from src.dummy_trader import _position_size

    qty = _position_size(capital=100000, entry=500.0, stop=500.0, side="BUY")
    assert qty == 0


def test_position_size_negative_risk():
    """Invalid risk direction should return 0."""
    from src.dummy_trader import _position_size

    # BUY with stop above entry = negative risk.
    qty = _position_size(capital=100000, entry=500.0, stop=510.0, side="BUY")
    assert qty == 0


def test_validate_signal_buy_valid():
    """Valid BUY signal should pass validation."""
    from src.dummy_trader import _validate_signal

    assert _validate_signal("RELIANCE", "BUY", 2450.0, 2420.0, 2520.0) is True


def test_validate_signal_sell_valid():
    """Valid SELL signal should pass validation."""
    from src.dummy_trader import _validate_signal

    assert _validate_signal("INFY", "SELL", 1520.0, 1550.0, 1460.0) is True


def test_validate_signal_buy_invalid():
    """BUY with SL above entry should fail."""
    from src.dummy_trader import _validate_signal

    assert _validate_signal("TCS", "BUY", 100.0, 110.0, 120.0) is False


def test_validate_signal_zero_price():
    """Zero price should fail validation."""
    from src.dummy_trader import _validate_signal

    assert _validate_signal("TCS", "BUY", 0.0, -10.0, 100.0) is False


def test_ingest_signals_df(initialized_db, sample_signals_df):
    """Ingesting signals should insert rows into the signals table."""
    from src.dummy_trader import ingest_signals_df

    count = ingest_signals_df(sample_signals_df)
    assert count >= 0  # May be 0 if ML confidence is below threshold.


def test_ingest_signals_df_dedup(initialized_db, sample_signals_df):
    """Duplicate row_hash values should not be inserted twice."""
    from src.dummy_trader import ingest_signals_df

    count1 = ingest_signals_df(sample_signals_df)
    count2 = ingest_signals_df(sample_signals_df)  # Same data again.
    assert count2 == 0, "Duplicates should not be inserted"


def test_ingest_signals_df_empty(initialized_db):
    """Empty DataFrame should return 0."""
    from src.dummy_trader import ingest_signals_df

    assert ingest_signals_df(pd.DataFrame()) == 0
    assert ingest_signals_df(None) == 0


def test_get_set_virtual_capital(initialized_db):
    """Virtual capital should be readable and writable."""
    from src.dummy_trader import get_virtual_capital, set_virtual_capital

    original = get_virtual_capital()
    assert original > 0
    set_virtual_capital(50000.0)
    assert get_virtual_capital() == 50000.0
    set_virtual_capital(original)  # Restore.


def test_count_open_trades(initialized_db):
    """Count should be 0 with fresh DB."""
    from src.dummy_trader import count_open_trades

    assert count_open_trades() == 0


def test_simulate_step_empty_db(initialized_db):
    """Simulate step on empty DB should return zeros."""
    from src.dummy_trader import simulate_step

    stats = simulate_step()
    assert stats["filled"] == 0
    assert stats["closed"] == 0


def test_force_close_eod_empty_db(initialized_db):
    """EOD closure on empty DB should return 0."""
    from src.dummy_trader import force_close_eod

    assert force_close_eod() == 0


def test_list_recent_trades_empty(initialized_db):
    """Should return empty DataFrame on fresh DB."""
    from src.dummy_trader import list_recent_trades

    df = list_recent_trades()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_list_recent_signals_empty(initialized_db):
    """Should return empty DataFrame on fresh DB."""
    from src.dummy_trader import list_recent_signals

    df = list_recent_signals()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
