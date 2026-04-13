# -*- coding: utf-8 -*-
"""Lightweight tests for pure metric helpers (no external services)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.performance import compute_max_drawdown, compute_profit_factor, compute_win_rate, metrics_from_pnls


def test_win_rate_basic():
    assert abs(compute_win_rate(__import__("numpy").array([1.0, -1.0, 2.0])) - (2 / 3)) < 1e-6


def test_profit_factor_basic():
    pf = compute_profit_factor(__import__("numpy").array([10.0, -5.0, 5.0]))
    assert abs(pf - (15.0 / 5.0)) < 1e-6


def test_max_drawdown_monotonic():
    import numpy as np

    eq = np.array([100.0, 110.0, 105.0, 120.0])
    mdd = compute_max_drawdown(eq)
    assert mdd > 0
    assert mdd < 0.1


def test_metrics_from_pnls_smoke():
    m = metrics_from_pnls([100.0, -40.0, 80.0], initial_capital=100000.0)
    assert "win_rate" in m and "max_drawdown_pct" in m
