# -*- coding: utf-8 -*-
"""Performance metrics: win rate, profit factor, Sharpe, drawdown, R:R, trades/day."""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    COMPLIANCE_NOTE,
    IST,
    TARGET_MAX_DRAWDOWN_PCT,
    TARGET_PROFIT_FACTOR,
    TARGET_WIN_RATE,
)
from src.db import atomic, ensure_database, get_connection

logger = logging.getLogger(__name__)


def closed_trades_dataframe() -> pd.DataFrame:
    """Load closed trades for analytics."""
    ensure_database()
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM trades WHERE status='closed' ORDER BY exit_time ASC",
        conn,
    )
    return df


def compute_win_rate(pnls: np.ndarray) -> float:
    """Share of positive net P&L trades."""
    if pnls.size == 0:
        return 0.0
    wins = np.sum(pnls > 0)
    return float(wins) / float(pnls.size)


def compute_profit_factor(pnls: np.ndarray) -> float:
    """Gross profit / gross loss (absolute)."""
    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_sharpe_ratio(
    pnls: np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    """
    Sharpe on per-trade returns approximated from P&L / notional=1 scale.
    Uses sample std with ddof=1; returns 0 if undefined.
    """
    if pnls.size < 2:
        return 0.0
    mu = float(np.mean(pnls))
    sd = float(np.std(pnls, ddof=1))
    if sd <= 0 or math.isnan(sd):
        return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as positive fraction of peak (e.g. 0.1 = 10%)."""
    if equity_curve.size == 0:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def average_risk_reward(
    entries: np.ndarray,
    exits: np.ndarray,
    sides: List[str],
    stops: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Average absolute (reward/risk) per closed trade when computable."""
    ratios: List[float] = []
    n = len(entries)
    for i in range(n):
        e = float(entries[i])
        x = float(exits[i])
        sl = float(stops[i])
        _ = float(targets[i])
        side = str(sides[i]).upper()
        if side == "BUY":
            risk = max(e - sl, 1e-6)
            reward = max(x - e, 0.0)
        else:
            risk = max(sl - e, 1e-6)
            reward = max(e - x, 0.0)
        if risk > 0:
            ratios.append(reward / risk)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def trades_per_day(exit_times: pd.Series) -> float:
    """Average closed trades per calendar day."""
    if exit_times is None or exit_times.empty:
        return 0.0
    days = pd.to_datetime(exit_times).dt.normalize().nunique()
    if days <= 0:
        return 0.0
    return float(len(exit_times)) / float(days)


def build_equity_curve(initial: float, pnls: np.ndarray) -> np.ndarray:
    """Cumulative equity from sequential net P&L."""
    if pnls.size == 0:
        return np.array([initial], dtype=float)
    return initial + np.cumsum(pnls)


def evaluate_success_gates(
    win_rate: float,
    profit_factor: float,
    max_dd: float,
) -> bool:
    """Project success definition: WR>60%, PF>1.5, MDD<10%."""
    pf_ok = profit_factor > TARGET_PROFIT_FACTOR
    wr_ok = win_rate > TARGET_WIN_RATE
    dd_ok = max_dd < TARGET_MAX_DRAWDOWN_PCT
    return bool(pf_ok and wr_ok and dd_ok)


def compute_all_metrics(
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Aggregate metrics from closed trades in SQLite.
    Returns dict suitable for DB insert and dashboard.
    """
    df = closed_trades_dataframe()
    if df.empty:
        return {
            "computed_at": datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S"),
            "win_rate": 0.0,
            "total_trades": 0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_rr": 0.0,
            "trades_per_day": 0.0,
            "success_gate": 0,
        }
    pnls = df["pnl_net"].astype(float).values
    win_rate = compute_win_rate(pnls)
    pf = compute_profit_factor(pnls)
    sharpe = compute_sharpe_ratio(pnls)
    equity = build_equity_curve(initial_capital, pnls)
    max_dd = compute_max_drawdown(equity)

    sides = df["side"].astype(str).tolist()
    entries = df["entry_price"].astype(float).values
    exits = df["exit_price"].astype(float).values
    conn = get_connection()
    aux = pd.read_sql(
        """
        SELECT t.id, s.stop_loss, s.target
        FROM trades t JOIN signals s ON s.id = t.signal_id
        WHERE t.status='closed'
        ORDER BY t.exit_time ASC
        """,
        conn,
    )
    stops = aux["stop_loss"].astype(float).values
    targets = aux["target"].astype(float).values
    avg_rr = average_risk_reward(entries, exits, sides, stops, targets)
    tpd = trades_per_day(df["exit_time"])
    net_pnl = float(np.sum(pnls))
    success = evaluate_success_gates(win_rate, pf, max_dd)
    pf_out = float(pf) if math.isfinite(pf) else 9999.0
    if pf_out > 9999.0:
        pf_out = 9999.0

    return {
        "computed_at": datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S"),
        "win_rate": float(win_rate),
        "total_trades": int(len(df)),
        "net_pnl": float(net_pnl),
        "profit_factor": pf_out,
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "avg_rr": float(avg_rr),
        "trades_per_day": float(tpd),
        "success_gate": int(1 if success else 0),
    }


def persist_metrics_row(metrics: Dict[str, Any]) -> None:
    """Insert a snapshot row into performance_metrics."""
    ensure_database()
    with atomic() as conn:
        conn.execute(
            """
            INSERT INTO performance_metrics(
                computed_at, win_rate, total_trades, net_pnl, profit_factor,
                sharpe_ratio, max_drawdown_pct, avg_rr, trades_per_day,
                success_gate, compliance_note
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                metrics["computed_at"],
                metrics["win_rate"],
                metrics["total_trades"],
                metrics["net_pnl"],
                metrics["profit_factor"],
                metrics["sharpe_ratio"],
                metrics["max_drawdown_pct"],
                metrics["avg_rr"],
                metrics["trades_per_day"],
                metrics["success_gate"],
                COMPLIANCE_NOTE,
            ),
        )


def latest_metrics() -> Optional[Dict[str, Any]]:
    """Most recent persisted metrics row."""
    ensure_database()
    import sqlite3
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM performance_metrics ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    conn.row_factory = None  # Reset for other callers.
    if not row:
        return None
    return dict(row)


def equity_series_for_plot(initial_capital: float) -> pd.DataFrame:
    """DataFrame with exit_time and equity for Streamlit charts."""
    df = closed_trades_dataframe()
    if df.empty:
        return pd.DataFrame(columns=["exit_time", "equity"])
    df = df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("exit_time")
    eq = build_equity_curve(initial_capital, df["pnl_net"].astype(float).values)
    out = pd.DataFrame({"exit_time": df["exit_time"].values, "equity": eq})
    return out


# --- Unit-test oriented pure helpers (no DB) ---


def metrics_from_pnls(
    pnls: List[float],
    initial_capital: float = 100000.0,
) -> Dict[str, float]:
    """Compute core metrics from a list of net P&L values (testing)."""
    arr = np.array(pnls, dtype=float)
    wr = compute_win_rate(arr)
    pf = compute_profit_factor(arr)
    sh = compute_sharpe_ratio(arr)
    eq = build_equity_curve(initial_capital, arr)
    mdd = compute_max_drawdown(eq)
    return {
        "win_rate": wr,
        "profit_factor": float(pf) if math.isfinite(pf) else 0.0,
        "sharpe_ratio": sh,
        "max_drawdown_pct": mdd,
        "net_pnl": float(arr.sum()),
    }
