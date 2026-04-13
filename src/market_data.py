# -*- coding: utf-8 -*-
"""NSE market data via yfinance with simple in-memory caching."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import IST, LOOKBACK_BARS, NSE_SUFFIX, YF_CACHE_SECONDS

logger = logging.getLogger(__name__)

_price_cache: Dict[str, Tuple[float, float]] = {}
_hist_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}


def nse_ticker(symbol: str) -> str:
    """Append NSE suffix for yfinance when missing."""
    s = str(symbol).strip().upper()
    if s.endswith(".NS"):
        return s
    return "{}{}".format(s, NSE_SUFFIX)


def get_last_price(symbol: str) -> Optional[float]:
    """
    Near-real-time last price. Uses 1m interval when market context allows,
    otherwise falls back to daily close.
    """
    key = nse_ticker(symbol)
    now = time.time()
    if key in _price_cache:
        ts, val = _price_cache[key]
        if now - ts < YF_CACHE_SECONDS and val is not None:
            return float(val)
    try:
        t = yf.Ticker(key)
        df = t.history(period="1d", interval="1m", auto_adjust=True)
        if df is not None and not df.empty and "Close" in df.columns:
            last = float(df["Close"].iloc[-1])
            _price_cache[key] = (now, last)
            return last
        df2 = t.history(period="5d", interval="1d", auto_adjust=True)
        if df2 is not None and not df2.empty and "Close" in df2.columns:
            last = float(df2["Close"].iloc[-1])
            _price_cache[key] = (now, last)
            return last
    except Exception as exc:
        logger.warning("get_last_price failed for {}: {}".format(symbol, exc))
    return None


def get_ohlcv_history(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download OHLCV history for feature engineering / training."""
    key = "{}|{}|{}".format(nse_ticker(symbol), period, interval)
    now = time.time()
    if use_cache and key in _hist_cache:
        ts, df = _hist_cache[key]
        if now - ts < max(YF_CACHE_SECONDS, 300) and df is not None and not df.empty:
            return df.copy()
    try:
        t = yf.Ticker(nse_ticker(symbol))
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        _hist_cache[key] = (now, df)
        return df.copy()
    except Exception as exc:
        logger.error("get_ohlcv_history failed for {}: {}".format(symbol, exc))
        return pd.DataFrame()


def intraday_snapshot(symbol: str) -> pd.DataFrame:
    """Fetch recent intraday bars when available (for live feature row)."""
    try:
        t = yf.Ticker(nse_ticker(symbol))
        df = t.history(period="5d", interval="5m", auto_adjust=True)
        if df is None or df.empty:
            return get_ohlcv_history(symbol, period="3mo", interval="1d")
        return df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
    except Exception as exc:
        logger.warning("intraday_snapshot failed for {}: {}".format(symbol, exc))
        return pd.DataFrame()


def bars_for_features(symbol: str) -> pd.DataFrame:
    """Prefer intraday; fall back to daily for minimum LOOKBACK_BARS."""
    df = intraday_snapshot(symbol)
    if df is not None and len(df) >= max(30, LOOKBACK_BARS // 4):
        return df
    return get_ohlcv_history(symbol, period="1y", interval="1d")


def clear_caches():
    """Reset caches (useful in tests)."""
    _price_cache.clear()
    _hist_cache.clear()


def synthetic_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV for offline testing when yfinance is empty.
    """
    np.random.seed(seed)
    rng = np.arange(n, dtype=float)
    base = 100.0 + (rng * 0.05) % 3.0
    noise = (rng % 7 - 3) * 0.2
    close = pd.Series(base + noise, dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = close.shift(1).fillna(close)
    vol = pd.Series((rng + 1.0) * 1000.0)
    idx = pd.date_range(end=datetime.now(tz=IST), periods=n, freq="D")
    out = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    return out
