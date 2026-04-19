# -*- coding: utf-8 -*-
"""Tests for src.fetch_sheet module."""

from src.fetch_sheet import fetch_signals_safe

def test_fetch_signals_returns_dataframe():
    df = fetch_signals_safe(prefer_sheet=False)
    assert not df.empty
    assert "symbol" in df.columns
    assert "buy_price" in df.columns
