# -*- coding: utf-8 -*-
"""Integration tests: full pipeline from signals to metrics."""

import pandas as pd
import pytest


def test_full_pipeline_mock_signals(initialized_db, sample_signals_df, monkeypatch):
    """
    End-to-end pipeline test:
      1. Ingest mock signals.
      2. Create trades for validated signals.
      3. Simulate step (no real prices, so fills may not trigger).
      4. Compute metrics.
      5. Persist metrics.

    This test verifies that the full pipeline runs without errors,
    NOT that trades actually execute (that requires real market data).
    """
    monkeypatch.setenv("USE_GOOGLE_SHEET", "0")

    from src.dummy_trader import (
        create_trades_for_validated_signals,
        ingest_signals_df,
        simulate_step,
        list_recent_signals,
    )
    from src.performance import compute_all_metrics, persist_metrics_row

    # Step 1: Ingest signals.
    count = ingest_signals_df(sample_signals_df)
    # Count may be 0 if ML confidence threshold is not met (no model).
    assert count >= 0

    # Step 2: Check signals are in DB.
    signals = list_recent_signals()
    assert isinstance(signals, pd.DataFrame)

    # Step 3: Create trades.
    trades_created = create_trades_for_validated_signals()
    assert trades_created >= 0

    # Step 4: Simulate (no real prices, so fills won't trigger).
    stats = simulate_step()
    assert "filled" in stats
    assert "closed" in stats

    # Step 5: Compute and persist metrics.
    metrics = compute_all_metrics(100000.0)
    assert "win_rate" in metrics
    assert "total_trades" in metrics
    assert "net_pnl" in metrics
    persist_metrics_row(metrics)


def test_pipeline_idempotency(initialized_db, sample_signals_df):
    """Running the pipeline twice with same signals should not duplicate."""
    from src.dummy_trader import ingest_signals_df

    count1 = ingest_signals_df(sample_signals_df)
    count2 = ingest_signals_df(sample_signals_df)
    assert count2 == 0, "Second ingestion should not insert duplicates"


def test_pipeline_empty_signals(initialized_db):
    """Pipeline should handle empty signal input gracefully."""
    from src.dummy_trader import ingest_signals_df, simulate_step
    from src.performance import compute_all_metrics

    ingest_signals_df(pd.DataFrame())
    stats = simulate_step()
    assert stats["filled"] == 0
    assert stats["closed"] == 0

    metrics = compute_all_metrics(100000.0)
    assert metrics["total_trades"] == 0


def test_equity_and_drawdown_empty(initialized_db):
    """Equity series should be empty with no closed trades."""
    from src.performance import equity_series_for_plot

    eq = equity_series_for_plot(100000.0)
    assert eq.empty or len(eq) == 0


def test_fetch_signals_safe_mock_mode(initialized_db, monkeypatch, tmp_path):
    """fetch_signals_safe should fall back to mock CSV."""
    monkeypatch.setenv("USE_GOOGLE_SHEET", "0")

    # Create a mock CSV in the temp directory.
    csv_content = (
        "Symbol,Signal,Buy Price,Stop Loss,Target,Timestamp\n"
        "RELIANCE,BUY,2450.0,2420.0,2520.0,2026-04-12 09:20:00\n"
    )
    csv_path = tmp_path / "mock_signals.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    monkeypatch.setattr("src.config.MOCK_SIGNALS_PATH", csv_path)

    from src.fetch_sheet import fetch_signals_safe

    df = fetch_signals_safe(prefer_sheet=False)
    assert len(df) >= 1
    assert "symbol" in df.columns


def test_retry_decorator():
    """retry_with_backoff should retry on failure and succeed eventually."""
    from src.retry import retry_with_backoff

    call_count = {"n": 0}

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def flaky_func():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("temporary failure")
        return "success"

    result = flaky_func()
    assert result == "success"
    assert call_count["n"] == 3


def test_retry_decorator_exhausted():
    """retry_with_backoff should raise after all retries exhausted."""
    from src.retry import retry_with_backoff

    @retry_with_backoff(max_retries=1, base_delay=0.01)
    def always_fail():
        raise ValueError("permanent failure")

    with pytest.raises(ValueError, match="permanent failure"):
        always_fail()
