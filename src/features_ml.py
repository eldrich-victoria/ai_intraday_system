# -*- coding: utf-8 -*-
"""Feature engineering (ta) and Random Forest signal validation."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    LOOKBACK_BARS,
    MODELS_DIR,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    TS_SPLIT_SPLITS,
    ensure_directories,
)
from src.market_data import bars_for_features, get_ohlcv_history, nse_ticker, synthetic_ohlcv

logger = logging.getLogger(__name__)

try:
    import ta
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

# Minimum mean CV accuracy threshold.  Below this we log a warning.
_CV_ACCURACY_WARN_THRESHOLD = 0.55


def _ensure_ta() -> None:
    """Raise if ta is not installed."""
    if ta is None:
        raise RuntimeError("ta library is required for feature engineering.")

def _get_model_path(symbol: str = "GLOBAL") -> Path:
    ensure_directories()
    # Normalize slashes or special characters
    safe_sym = str(symbol).replace("^", "").replace("/", "").replace("\\", "")
    return MODELS_DIR / f"rf_model_{safe_sym}.pkl"

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI(14), MACD(12,26,9), ATR(14), SMA/EMA(20) and auxiliary columns.

    Uses ``.dropna()`` instead of ``.bfill()`` to prevent look-ahead bias /
    data leakage in the training pipeline.
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
                logger.warning(f"OHLCV missing column: {col}")
                return pd.DataFrame()
    df = df.sort_index()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0)

    rsi = ta.momentum.rsi(close, window=14)
    macd = ta.trend.macd(close, window_slow=26, window_fast=12)
    macd_signal = ta.trend.macd_signal(close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = ta.trend.macd_diff(close, window_slow=26, window_fast=12, window_sign=9)
    atr = ta.volatility.average_true_range(high=high, low=low, close=close, window=14)
    sma20 = ta.trend.sma_indicator(close, window=20)
    ema20 = ta.trend.ema_indicator(close, window=20)

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = rsi
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["atr_14"] = atr
    out["sma_20"] = sma20
    out["ema_20"] = ema20
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    out["vol_z_20"] = (volume - vol_mean) / vol_std

    # Replace infinities with NaN, then DROP rows with any NaN.
    # This prevents data leakage that would occur with .bfill().
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
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


def _build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline with StandardScaler and RandomForest.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])


def train_random_forest_and_save(
    symbols: Optional[List[str]] = None,
) -> str:
    """
    Train RandomForest per-symbol and globally.
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
            # Train symbol-specific model
            pipe = _build_pipeline()
            pipe.fit(X, y)
            joblib.dump({"model": pipe, "features": FEATURE_COLUMNS}, _get_model_path(sym))
            logger.info(f"Saved symbol model for {sym}")

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

    pipe = _build_pipeline()

    if n_splits < 2 or len(X_all) < 10:
        pipe.fit(X_all, y_all)
        path = _get_model_path("GLOBAL")
        joblib.dump({"model": pipe, "features": FEATURE_COLUMNS}, path)
        logger.info("Saved global model (small sample)")
        return str(path)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores: List[float] = []
    for train_idx, test_idx in tscv.split(X_all):
        pipe.fit(X_all.iloc[train_idx], y_all.iloc[train_idx])
        cv_scores.append(
            float(pipe.score(X_all.iloc[test_idx], y_all.iloc[test_idx]))
        )

    mean_cv = float(np.mean(cv_scores))
    logger.info(f"TimeSeriesSplit CV accuracy (mean={mean_cv:.4f})")
    if mean_cv < _CV_ACCURACY_WARN_THRESHOLD:
        logger.warning(
            f"Mean CV accuracy {mean_cv:.4f} is below threshold {_CV_ACCURACY_WARN_THRESHOLD}. "
            "Model quality may be poor."
        )

    pipe.fit(X_all, y_all)
    path = _get_model_path("GLOBAL")
    joblib.dump({"model": pipe, "features": FEATURE_COLUMNS}, path)
    logger.info("Saved global model")
    return str(path)


def load_model(symbol: str = "GLOBAL") -> Optional[Dict[str, Any]]:
    """Load persisted model bundle."""
    path = _get_model_path(symbol)
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


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
    Return confidence in [0,1]. fallback symbol -> global.
    """
    if model_bundle is not None:
        bundle = model_bundle
    else:
        bundle = load_model(symbol)
        if bundle is None:
            bundle = load_model("GLOBAL")

    if bundle is None:
        logger.warning("No model on disk; run retrain. Using neutral 0.5")
        return 0.5
        
    pipe = bundle["model"]
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
    proba = pipe.predict_proba(x_df)[0]
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
    print(f"Global model written to {path}")
