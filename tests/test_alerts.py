# -*- coding: utf-8 -*-
"""Tests for src.alerts module."""



def test_send_telegram_disabled_without_creds():
    """Sending should return False when credentials are missing."""
    from src.alerts import send_telegram_message

    # conftest already clears BOT_TOKEN and CHAT_ID.
    result = send_telegram_message("test message")
    assert result is False


def test_alert_trade_entry_no_creds():
    """Entry alert should gracefully return False without creds."""
    from src.alerts import alert_trade_entry

    result = alert_trade_entry("RELIANCE", "BUY", 10, 2450.0)
    assert result is False


def test_alert_trade_exit_no_creds():
    """Exit alert should gracefully return False without creds."""
    from src.alerts import alert_trade_exit

    result = alert_trade_exit("RELIANCE", "target", 500.0)
    assert result is False


def test_alert_daily_summary_no_creds():
    """Daily summary should gracefully return False without creds."""
    from src.alerts import alert_daily_summary

    result = alert_daily_summary("Test summary")
    assert result is False


def test_format_daily_summary():
    """Summary formatting should produce readable text."""
    from src.alerts import format_daily_summary

    metrics = {
        "win_rate": 0.65,
        "profit_factor": 2.0,
        "net_pnl": 5000.0,
        "max_drawdown_pct": 0.05,
        "sharpe_ratio": 1.5,
        "total_trades": 10,
        "success_gate": 1,
    }
    text = format_daily_summary(metrics)
    assert "65.00%" in text
    assert "5000.00" in text
    assert "PASS" in text


def test_format_daily_summary_failing():
    """Failing gate should show FAIL."""
    from src.alerts import format_daily_summary

    metrics = {
        "win_rate": 0.40,
        "profit_factor": 0.8,
        "net_pnl": -2000.0,
        "max_drawdown_pct": 0.15,
        "sharpe_ratio": -0.5,
        "total_trades": 5,
        "success_gate": 0,
    }
    text = format_daily_summary(metrics)
    assert "FAIL" in text
