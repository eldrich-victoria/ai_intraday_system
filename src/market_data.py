# -*- coding: utf-8 -*-
"""NSE market data via yfinance with simple in-memory caching."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import IST, LOOKBACK_BARS, NSE_SUFFIX, YF_CACHE_SECONDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker formatting
# ---------------------------------------------------------------------------

def nse_ticker(symbol: str) -> str:
    """Append NSE suffix for yfinance when missing."""
    s = str(symbol).strip().upper()
    if s.endswith(".NS"):
        return s
    return "{}{}".format(s, NSE_SUFFIX)

# ---------------------------------------------------------------------------
# Abstract provider for future extensibility
# ---------------------------------------------------------------------------

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> Optional[float]:
        ...

    @abstractmethod
    def get_ohlcv_history(
        self, symbol: str, period: str, interval: str, use_cache: bool = True
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def intraday_snapshot(self, symbol: str) -> pd.DataFrame:
        ...
        
    @abstractmethod
    def clear_caches(self) -> None:
        ...

class YFinanceProvider(MarketDataProvider):
    def __init__(self):
        self._price_cache: Dict[str, Tuple[float, float]] = {}
        self._hist_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        
    def get_last_price(self, symbol: str) -> Optional[float]:
        key = nse_ticker(symbol)
        now = time.time()
        if key in self._price_cache:
            ts, val = self._price_cache[key]
            if now - ts < YF_CACHE_SECONDS and val is not None:
                return float(val)
        try:
            t = yf.Ticker(key)
            df = t.history(period="1d", interval="1m", auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                last = float(df["Close"].iloc[-1])
                self._price_cache[key] = (now, last)
                return last
            df2 = t.history(period="5d", interval="1d", auto_adjust=True)
            if df2 is not None and not df2.empty and "Close" in df2.columns:
                last = float(df2["Close"].iloc[-1])
                self._price_cache[key] = (now, last)
                return last
        except Exception as exc:
            logger.warning(f"get_last_price failed for {symbol}: {exc}")
        return None

    def get_ohlcv_history(
        self, symbol: str, period: str = "3mo", interval: str = "1d", use_cache: bool = True
    ) -> pd.DataFrame:
        key = f"{nse_ticker(symbol)}|{period}|{interval}"
        now = time.time()
        if use_cache and key in self._hist_cache:
            ts, df = self._hist_cache[key]
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
            self._hist_cache[key] = (now, df)
            return df.copy()
        except Exception as exc:
            logger.error(f"get_ohlcv_history failed for {symbol}: {exc}")
            return pd.DataFrame()

    def intraday_snapshot(self, symbol: str) -> pd.DataFrame:
        try:
            t = yf.Ticker(nse_ticker(symbol))
            df = t.history(period="5d", interval="5m", auto_adjust=True)
            if df is None or df.empty:
                return self.get_ohlcv_history(symbol, period="3mo", interval="1d")
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
            logger.warning(f"intraday_snapshot failed for {symbol}: {exc}")
            return pd.DataFrame()
            
    def clear_caches(self) -> None:
        self._price_cache.clear()
        self._hist_cache.clear()

# Global Provider Instance
_provider: MarketDataProvider = YFinanceProvider()

def set_provider(provider: MarketDataProvider) -> None:
    global _provider
    _provider = provider

def get_last_price(symbol: str) -> Optional[float]:
    return _provider.get_last_price(symbol)

def get_ohlcv_history(
    symbol: str, period: str = "3mo", interval: str = "1d", use_cache: bool = True
) -> pd.DataFrame:
    return _provider.get_ohlcv_history(symbol, period, interval, use_cache)

def intraday_snapshot(symbol: str) -> pd.DataFrame:
    return _provider.intraday_snapshot(symbol)

def bars_for_features(symbol: str) -> pd.DataFrame:
    df = intraday_snapshot(symbol)
    if df is not None and len(df) >= max(30, LOOKBACK_BARS // 4):
        return df
    return get_ohlcv_history(symbol, period="1y", interval="1d")

def clear_caches() -> None:
    _provider.clear_caches()

def synthetic_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV for offline testing when yfinance is empty.

    Uses ``np.random.default_rng`` instead of global ``np.random.seed``
    to avoid polluting the global random state.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=float)
    base = 100.0 + (indices * 0.05) % 3.0
    noise = rng.normal(0, 0.3, size=n)
    close = pd.Series(base + noise, dtype=float)
    high = close + rng.uniform(0.2, 0.8, size=n)
    low = close - rng.uniform(0.2, 0.8, size=n)
    open_ = close.shift(1).fillna(close)
    vol = pd.Series((indices + 1.0) * 1000.0 + rng.uniform(0, 500, size=n))
    end_date = pd.Timestamp("2030-01-01", tz=IST)
    idx = pd.date_range(end=end_date, periods=n, freq="D")
    out = pd.DataFrame(
        {"open": open_.values, "high": high.values, "low": low.values, "close": close.values, "volume": vol.values},
        index=idx,
    )
    return out
