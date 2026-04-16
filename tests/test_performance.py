# -*- coding: utf-8 -*-
"""Tests for src.performance module — pure metric helpers and DB-backed analytics."""


import numpy as np

from src.performance import (
    build_equity_curve,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_win_rate,
    evaluate_success_gates,
    metrics_from_pnls,
)


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_basic(self):
        assert abs(compute_win_rate(np.array([1.0, -1.0, 2.0])) - (2 / 3)) < 1e-6

    def test_all_winners(self):
        assert compute_win_rate(np.array([1.0, 2.0, 3.0])) == 1.0

    def test_all_losers(self):
        assert compute_win_rate(np.array([-1.0, -2.0])) == 0.0

    def test_empty(self):
        assert compute_win_rate(np.array([])) == 0.0

    def test_zero_trades(self):
        """Zero P&L trades should not count as wins."""
        assert compute_win_rate(np.array([0.0, 0.0])) == 0.0


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_basic(self):
        pf = compute_profit_factor(np.array([10.0, -5.0, 5.0]))
        assert abs(pf - 3.0) < 1e-6

    def test_no_losses(self):
        pf = compute_profit_factor(np.array([10.0, 5.0]))
        assert pf == float("inf")

    def test_no_gains(self):
        pf = compute_profit_factor(np.array([-10.0, -5.0]))
        assert pf == 0.0

    def test_empty(self):
        pf = compute_profit_factor(np.array([]))
        assert pf == 0.0


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_basic(self):
        pnls = np.array([100.0, -50.0, 80.0, -30.0, 60.0])
        sharpe = compute_sharpe_ratio(pnls)
        assert isinstance(sharpe, float)
        # With positive mean, sharpe should be positive.
        assert sharpe > 0

    def test_single_trade(self):
        assert compute_sharpe_ratio(np.array([100.0])) == 0.0

    def test_empty(self):
        assert compute_sharpe_ratio(np.array([])) == 0.0

    def test_zero_std(self):
        """All identical P&L should yield 0 (std=0)."""
        assert compute_sharpe_ratio(np.array([10.0, 10.0, 10.0])) == 0.0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotonic_up(self):
        eq = np.array([100.0, 110.0, 120.0, 130.0])
        assert compute_max_drawdown(eq) == 0.0

    def test_with_drawdown(self):
        eq = np.array([100.0, 110.0, 105.0, 120.0])
        mdd = compute_max_drawdown(eq)
        assert mdd > 0
        assert mdd < 0.1  # ~4.5% drawdown.

    def test_severe_drawdown(self):
        eq = np.array([100.0, 50.0, 60.0])
        mdd = compute_max_drawdown(eq)
        assert abs(mdd - 0.5) < 1e-6

    def test_empty(self):
        assert compute_max_drawdown(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_basic(self):
        eq = build_equity_curve(100000.0, np.array([100.0, -50.0, 200.0]))
        assert len(eq) == 3
        assert eq[0] == 100100.0
        assert eq[-1] == 100250.0

    def test_empty(self):
        eq = build_equity_curve(100000.0, np.array([]))
        assert len(eq) == 1
        assert eq[0] == 100000.0


# ---------------------------------------------------------------------------
# Success gates
# ---------------------------------------------------------------------------

class TestSuccessGates:
    def test_pass(self):
        assert evaluate_success_gates(0.65, 2.0, 0.05) is True

    def test_fail_win_rate(self):
        assert evaluate_success_gates(0.50, 2.0, 0.05) is False

    def test_fail_profit_factor(self):
        assert evaluate_success_gates(0.65, 1.0, 0.05) is False

    def test_fail_drawdown(self):
        assert evaluate_success_gates(0.65, 2.0, 0.15) is False


# ---------------------------------------------------------------------------
# Combined metrics
# ---------------------------------------------------------------------------

class TestMetricsFromPnls:
    def test_smoke(self):
        m = metrics_from_pnls([100.0, -40.0, 80.0], initial_capital=100000.0)
        assert "win_rate" in m
        assert "max_drawdown_pct" in m
        assert "profit_factor" in m
        assert "sharpe_ratio" in m
        assert "net_pnl" in m

    def test_values(self):
        m = metrics_from_pnls([100.0, -40.0, 80.0])
        assert m["net_pnl"] == 140.0
        assert m["win_rate"] > 0.6

    def test_empty(self):
        m = metrics_from_pnls([])
        assert m["net_pnl"] == 0.0
        assert m["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# DB-backed metrics (requires initialized_db fixture)
# ---------------------------------------------------------------------------

def test_compute_all_metrics_empty_db(initialized_db):
    """All metrics should return zeros on empty DB."""
    from src.performance import compute_all_metrics

    m = compute_all_metrics(100000.0)
    assert m["total_trades"] == 0
    assert m["win_rate"] == 0.0


def test_persist_and_read_metrics(initialized_db):
    """Metrics should be persistable and retrievable."""
    from src.performance import persist_metrics_row, latest_metrics

    metrics = {
        "computed_at": "2026-04-12 10:00:00",
        "win_rate": 0.65,
        "total_trades": 10,
        "net_pnl": 5000.0,
        "profit_factor": 2.0,
        "sharpe_ratio": 1.5,
        "max_drawdown_pct": 0.05,
        "avg_rr": 1.8,
        "trades_per_day": 3.0,
        "success_gate": 1,
    }
    persist_metrics_row(metrics)
    latest = latest_metrics()
    assert latest is not None
    assert latest["win_rate"] == 0.65
    assert latest["total_trades"] == 10
