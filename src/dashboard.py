# -*- coding: utf-8 -*-
"""Streamlit dashboard: live trades, equity curve, metrics, drawdown."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from src.config import COMPLIANCE_NOTE, IST, INITIAL_CAPITAL
from src.db import ensure_database, get_connection
from src.dummy_trader import get_virtual_capital
from src.performance import compute_all_metrics, equity_series_for_plot, latest_metrics

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


@st.cache_data(ttl=30.0)
def _cached_trades() -> pd.DataFrame:
    """Fetch trades from DB with caching."""
    ensure_database()
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM trades ORDER BY id DESC LIMIT 200",
        conn,
    )


@st.cache_data(ttl=30.0)
def _cached_signals() -> pd.DataFrame:
    """Fetch signals from DB with caching."""
    ensure_database()
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM signals ORDER BY id DESC LIMIT 200",
        conn,
    )


@st.cache_data(ttl=30.0)
def _cached_equity(initial: float) -> pd.DataFrame:
    """Fetch equity series with caching."""
    return equity_series_for_plot(initial)


def _apply_filters(df: pd.DataFrame, symbol: str, status: str,
                    date_start: object, date_end: object,
                    time_col: str) -> pd.DataFrame:
    """Apply sidebar filters to a DataFrame."""
    if df.empty:
        return df
    filtered = df.copy()
    if symbol != "All":
        filtered = filtered[filtered["symbol"] == symbol]
    if status != "All" and "status" in filtered.columns:
        filtered = filtered[filtered["status"] == status]
    if time_col in filtered.columns:
        try:
            times = pd.to_datetime(filtered[time_col], errors="coerce")
            mask = times.notna()
            if date_start is not None:
                mask = mask & (times.dt.date >= date_start)
            if date_end is not None:
                mask = mask & (times.dt.date <= date_end)
            filtered = filtered[mask]
        except Exception:
            pass
    return filtered


def main() -> None:
    """Main Streamlit dashboard entrypoint."""
    st.set_page_config(
        page_title="AI Intraday Trading Tester",
        page_icon="\U0001f4c8",
        layout="wide",
    )

    # Auto-refresh every 30 seconds if streamlit-autorefresh is installed.
    if st_autorefresh is not None:
        st_autorefresh(interval=30000, limit=None, key="auto_refresh")

    st.title("\U0001f4c8 AI Intraday Trading Tester (Paper)")

    # SEBI compliance disclaimer — prominent placement.
    st.warning(COMPLIANCE_NOTE, icon="\u26a0\ufe0f")

    # ----- Sidebar filters -----
    st.sidebar.header("Filters")

    trades_df = _cached_trades()
    signals_df = _cached_signals()

    # Build symbol list from trades and signals.
    all_symbols = set()
    if not trades_df.empty and "symbol" in trades_df.columns:
        all_symbols.update(trades_df["symbol"].dropna().unique())
    if not signals_df.empty and "symbol" in signals_df.columns:
        all_symbols.update(signals_df["symbol"].dropna().unique())
    symbol_options = ["All"] + sorted(all_symbols)
    selected_symbol = st.sidebar.selectbox(
        "Symbol", symbol_options, index=0, key="filter_symbol"
    )

    status_options = ["All", "pending", "active", "closed"]
    selected_status = st.sidebar.selectbox(
        "Trade status", status_options, index=0, key="filter_status"
    )

    date_range = st.sidebar.date_input(
        "Date range", value=[], key="filter_dates"
    )
    date_start = date_range[0] if len(date_range) >= 1 else None
    date_end = date_range[1] if len(date_range) >= 2 else date_start

    # ----- Top metrics strip -----
    initial = float(INITIAL_CAPITAL)
    capital = get_virtual_capital()
    live_metrics = compute_all_metrics(initial)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Virtual capital", "\u20b9{:,.0f}".format(capital))
    col2.metric("Win rate", "{:.1%}".format(live_metrics["win_rate"]))
    col3.metric("Profit factor", "{:.2f}".format(live_metrics["profit_factor"]))
    col4.metric("Max drawdown", "{:.1%}".format(live_metrics["max_drawdown_pct"]))
    col5.metric("Net P&L", "\u20b9{:,.2f}".format(live_metrics["net_pnl"]))

    st.subheader("Success gates (project definition)")
    gate_ok = bool(live_metrics.get("success_gate"))
    if gate_ok:
        st.success(
            "Gates (WR>60%, PF>1.5, MDD<10%): \u2705 PASS"
        )
    else:
        st.error(
            "Gates (WR>60%, PF>1.5, MDD<10%): \u274c FAIL"
        )

    # ----- Tabs -----
    tab1, tab2, tab3 = st.tabs(
        ["\U0001f4ca Trades & Equity", "\U0001f4e1 Signals", "\U0001f4be Stored Snapshots"]
    )

    with tab1:
        filtered_trades = _apply_filters(
            trades_df, selected_symbol, selected_status,
            date_start, date_end, "created_at"
        )
        st.dataframe(filtered_trades, use_container_width=True)

        eq = _cached_equity(initial)
        if not eq.empty:
            eq = eq.copy()
            eq["peak"] = eq["equity"].cummax()
            eq["drawdown"] = (eq["peak"] - eq["equity"]) / eq["peak"]

            fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

            # Equity curve.
            ax[0].plot(
                eq["exit_time"], eq["equity"],
                color="#2196F3", linewidth=1.5, label="Equity"
            )
            ax[0].axhline(
                y=initial, color="#9E9E9E", linestyle="--",
                alpha=0.6, label="Initial capital"
            )
            ax[0].set_ylabel("Equity (\u20b9)")
            ax[0].legend(loc="upper left")
            ax[0].grid(True, alpha=0.3)

            # Drawdown.
            ax[1].fill_between(
                eq["exit_time"],
                eq["drawdown"],
                color="#F44336",
                alpha=0.4,
                label="Drawdown",
            )
            ax[1].set_ylabel("Drawdown")
            ax[1].set_xlabel("Time")
            ax[1].legend(loc="upper left")
            ax[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No closed trades yet for equity/drawdown charts.")

    with tab2:
        filtered_signals = _apply_filters(
            signals_df, selected_symbol, "All",
            date_start, date_end, "fetched_at"
        )
        st.dataframe(filtered_signals, use_container_width=True)

    with tab3:
        snap = latest_metrics()
        if snap:
            st.json(snap)
        else:
            st.info("No persisted metrics yet (run scheduler).")

    # ----- Footer -----
    if st.button("Refresh data now", key="btn_refresh"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Data refreshes automatically every ~30s or when you click Refresh. "
        "Timezone: {}".format(str(IST))
    )


if __name__ == "__main__":
    main()
