# -*- coding: utf-8 -*-
"""Tests for src.features_ml module."""

import numpy as np
import pandas as pd
import pytest


def test_compute_features_returns_correct_columns(synthetic_ohlcv):
    """Feature computation should produce all expected columns."""
    from src.features_ml import compute_features, FEATURE_COLUMNS

    feats = compute_features(synthetic_ohlcv)
    if feats.empty:
        pytest.skip("Not enough data for feature computation")
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, "Missing feature column: {}".format(col)


def test_compute_features_no_nan_after_dropna(synthetic_ohlcv):
    """After dropna, features should contain no NaN values."""
    from src.features_ml import compute_features

    feats = compute_features(synthetic_ohlcv)
    if feats.empty:
        pytest.skip("Not enough data")
    assert not feats.isna().any().any(), "Features contain NaN after dropna"


def test_compute_features_no_inf(synthetic_ohlcv):
    """Features should not contain infinite values."""
    from src.features_ml import compute_features

    feats = compute_features(synthetic_ohlcv)
    if feats.empty:
        pytest.skip("Not enough data")
    assert not np.isinf(feats.values).any(), "Features contain inf values"


def test_compute_features_empty_input():
    """Empty OHLCV should return empty DataFrame."""
    from src.features_ml import compute_features

    result = compute_features(pd.DataFrame())
    assert result.empty


def test_compute_features_none_input():
    """None input should return empty DataFrame."""
    from src.features_ml import compute_features

    result = compute_features(None)
    assert result.empty


def test_feature_columns_constant():
    """FEATURE_COLUMNS list should have expected length."""
    from src.features_ml import FEATURE_COLUMNS

    assert len(FEATURE_COLUMNS) == 10


def test_signal_confidence_no_model():
    """Without a model file, confidence should be neutral 0.5."""
    from src.features_ml import signal_confidence

    conf = signal_confidence("RELIANCE", "BUY")
    assert conf == 0.5


def test_signal_confidence_range():
    """Confidence should always be in [0, 1]."""
    from src.features_ml import signal_confidence

    for side in ("BUY", "SELL"):
        conf = signal_confidence("RELIANCE", side)
        assert 0.0 <= conf <= 1.0
