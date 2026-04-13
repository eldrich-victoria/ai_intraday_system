# -*- coding: utf-8 -*-
"""Feature engineering (pandas-ta) and Random Forest signal validation."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    LOOKBACK_BARS,
    MODEL_PATH,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    TS_SPLIT_SPLITS,
    ensure_directories,
)
from src.market_data import bars_for_features, get_ohlcv_history, nse_ticker, synthetic_ohlcv

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
except ImportError:
    ta = None


FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "sma_20",
    "ema_20",
    "ret_1",
    "ret_5",
    "vol_z_20",
]


def _ensure_ta():
    if ta is None:
        raise RuntimeError("pandas_ta is required for feature engineering.")


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI(14), MACD(12,26,9), ATR(14), SMA/EMA(20) and auxiliary columns.
    Forward-fills missing values; drops rows with insufficient history.
    """
    _ensure_ta()
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()
    df = ohlcv.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            if col == "close" and "Close" in df.columns:
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
            else:
                logger.warning("OHLCV missing column: {}".format(col))
                return pd.DataFrame()
    df = df.sort_index()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0)

    rsi = ta.rsi(close, length=14)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    atr = ta.atr(high=high, low=low, close=close, length=14)
    sma20 = ta.sma(close, length=20)
    ema20 = ta.ema(close, length=20)

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = rsi
    if macd_df is not None and macd_df.shape[1] >= 3:
        out["macd"] = macd_df.iloc[:, 0]
        out["macd_signal"] = macd_df.iloc[:, 1]
        out["macd_hist"] = macd_df.iloc[:, 2]
    else:
        out["macd"] = np.nan
        out["macd_signal"] = np.nan
        out["macd_hist"] = np.nan
    out["atr_14"] = atr
    out["sma_20"] = sma20
    out["ema_20"] = ema20
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    out["vol_z_20"] = (volume - vol_mean) / vol_std

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out


def _build_training_matrix(
    symbol: str,
    horizon: int = 5,
    min_samples: int = 80,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create (X, y) from historical OHLCV; y=1 if forward return over horizon > 0.4%."""
    df_hist = get_ohlcv_history(symbol, period="2y", interval="1d")
    if df_hist is None or len(df_hist) < min_samples:
        df_hist = synthetic_ohlcv(n=220)
    feats = compute_features(df_hist)
    if feats.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    close = df_hist["close"].reindex(feats.index).ffill()
    fwd = close.shift(-horizon) / close - 1.0
    y = (fwd > 0.004).astype(int)
    aligned = feats.join(y.rename("y"), how="inner").dropna()
    if aligned.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    X = aligned[FEATURE_COLUMNS]
    yy = aligned["y"].astype(int)
    return X, yy


def train_random_forest_and_save(
    symbols: Optional[List[str]] = None,
) -> str:
    """
    Train RandomForest with TimeSeriesSplit on concatenated histories.
    Saves model to MODEL_PATH. Returns path written.
    """
    ensure_directories()
    symbols = symbols or ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    parts_x: List[pd.DataFrame] = []
    parts_y: List[pd.Series] = []
    for sym in symbols:
        X, y = _build_training_matrix(sym)
        if len(X) > 30:
            parts_x.append(X)
            parts_y.append(y)
    if not parts_x:
        X, y = _build_training_matrix("SYNTH")
        parts_x.append(X)
        parts_y.append(y)
    X_all = pd.concat(parts_x, axis=0, ignore_index=True)
    y_all = pd.concat(parts_y, axis=0, ignore_index=True)
    n = min(len(X_all), len(y_all))
    X_all = X_all.iloc[:n].reset_index(drop=True)
    y_all = y_all.iloc[:n].reset_index(drop=True)
    valid = X_all.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    X_all = X_all.loc[valid].reset_index(drop=True)
    y_all = y_all.loc[valid].reset_index(drop=True)

    n_splits = min(TS_SPLIT_SPLITS, max(2, len(X_all) // 25))
    n_splits = min(n_splits, max(2, len(X_all) - 1))
    if len(X_all) <= n_splits + 2:
        n_splits = max(2, min(3, len(X_all) - 2))
    if n_splits < 2 or len(X_all) < 10:
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        clf.fit(X_all, y_all)
        joblib.dump({"model": clf, "features": FEATURE_COLUMNS}, MODEL_PATH)
        logger.info("Saved RandomForest model to {} (small sample, no CV)".format(MODEL_PATH))
        return str(MODEL_PATH)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    cv_scores = []
    for train_idx, test_idx in tscv.split(X_all):
        clf.fit(X_all.iloc[train_idx], y_all.iloc[train_idx])
        cv_scores.append(
            float(clf.score(X_all.iloc[test_idx], y_all.iloc[test_idx]))
        )
    logger.info("TimeSeriesSplit CV accuracy folds: {}".format(cv_scores))
    clf.fit(X_all, y_all)

    joblib.dump({"model": clf, "features": FEATURE_COLUMNS}, MODEL_PATH)
    logger.info("Saved RandomForest model to {}".format(MODEL_PATH))
    return str(MODEL_PATH)


def load_model() -> Optional[Dict[str, Any]]:
    """Load persisted model bundle."""
    if not os.path.isfile(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def latest_feature_vector(symbol: str) -> Optional[pd.Series]:
    """Most recent feature row for a symbol (for live inference)."""
    ohlc = bars_for_features(symbol)
    if ohlc is None or ohlc.empty:
        ohlc = synthetic_ohlcv(n=200)
    feats = compute_features(ohlc)
    if feats is None or feats.empty:
        return None
    row = feats.iloc[-1]
    return row


def signal_confidence(
    symbol: str,
    signal_side: str,
    model_bundle: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Return confidence in [0,1]. Uses RandomForest proba of class 1 (bullish proxy).
    BUY uses p; SELL uses (1 - p).
    """
    bundle = model_bundle or load_model()
    if bundle is None:
        logger.warning("No model on disk; run retrain. Using neutral 0.5")
        return 0.5
    clf = bundle["model"]
    feats_order: List[str] = bundle.get("features", FEATURE_COLUMNS)
    vec = latest_feature_vector(symbol)
    if vec is None:
        return 0.5
    row_dict = {}
    for k in feats_order:
        v = vec.get(k, np.nan)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if np.isnan(fv) or np.isinf(fv):
            fv = 0.0
        row_dict[k] = fv
    x_df = pd.DataFrame([row_dict], columns=feats_order)
    proba = clf.predict_proba(x_df)[0]
    # class order: sklearn sorts by label - assume binary 0,1
    p_up = float(proba[-1]) if len(proba) > 1 else float(proba[0])
    side = str(signal_side).upper().strip()
    if side == "SELL":
        return max(0.0, min(1.0, 1.0 - p_up))
    return max(0.0, min(1.0, p_up))


def retrain_model(symbols: Optional[List[str]] = None) -> str:
    """Public entrypoint for scheduled or manual retraining."""
    return train_random_forest_and_save(symbols=symbols)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = retrain_model()
    print("Model written to {}".format(path))
