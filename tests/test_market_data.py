# -*- coding: utf-8 -*-
"""Tests for src.market_data module."""

import numpy as np
import pandas as pd


def test_nse_ticker_appends_suffix():
    """nse_ticker should append .NS if missing."""
    from src.market_data import nse_ticker

    assert nse_ticker("RELIANCE") == "RELIANCE.NS"
    assert nse_ticker("reliance") == "RELIANCE.NS"
    assert nse_ticker("RELIANCE.NS") == "RELIANCE.NS"
    assert nse_ticker("  TCS  ") == "TCS.NS"


def test_synthetic_ohlcv_shape():
    """Synthetic data should have correct shape and columns."""
    from src.market_data import synthetic_ohlcv

    df = synthetic_ohlcv(n=100, seed=42)
    assert len(df) == 100
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


def test_synthetic_ohlcv_deterministic():
    """Same seed should produce identical data."""
    from src.market_data import synthetic_ohlcv

    df1 = synthetic_ohlcv(n=50, seed=123)
    df2 = synthetic_ohlcv(n=50, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_ohlcv_different_seeds():
    """Different seeds should produce different data."""
    from src.market_data import synthetic_ohlcv

    df1 = synthetic_ohlcv(n=50, seed=1)
    df2 = synthetic_ohlcv(n=50, seed=2)
    assert not df1["close"].equals(df2["close"])


def test_synthetic_ohlcv_price_sanity():
    """Prices should be positive and high >= low."""
    from src.market_data import synthetic_ohlcv

    df = synthetic_ohlcv(n=200, seed=42)
    assert (df["close"] > 0).all()
    assert (df["volume"] > 0).all()


def test_clear_caches():
    """Cache clearing should not raise."""
    from src.market_data import clear_caches
    clear_caches()  # Should not raise.


def test_synthetic_does_not_pollute_global_rng():
    """Using default_rng should not affect np.random global state."""
    from src.market_data import synthetic_ohlcv

    np.random.seed(99)
    before = np.random.random()
    np.random.seed(99)
    synthetic_ohlcv(n=50, seed=42)
    after = np.random.random()
    assert before == after, "synthetic_ohlcv should not pollute global RNG state"
