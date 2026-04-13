# -*- coding: utf-8 -*-
"""Streamlit dashboard: live trades, equity curve, metrics, drawdown."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.config import COMPLIANCE_NOTE, DB_PATH, IST, INITIAL_CAPITAL
from src.dummy_trader import ensure_database, get_virtual_capital
from src.performance import compute_all_metrics, equity_series_for_plot, latest_metrics


def _sql_engine():
    uri = "sqlite:///{}".format(DB_PATH.as_posix())
    return create_engine(uri, future=True)


@st.cache_data(ttl=30.0)
def _cached_trades():
    ensure_database()
    return pd.read_sql(
        "SELECT * FROM trades ORDER BY id DESC LIMIT 200",
        _sql_engine(),
    )


@st.cache_data(ttl=30.0)
def _cached_signals():
    ensure_database()
    return pd.read_sql(
        "SELECT * FROM signals ORDER BY id DESC LIMIT 200",
        _sql_engine(),
    )


@st.cache_data(ttl=30.0)
def _cached_equity(initial: float):
    return equity_series_for_plot(initial)


def main():
    st.set_page_config(
        page_title="AI Intraday Trading Tester",
        layout="wide",
    )
    st.title("AI Intraday Trading Tester (Paper)")
    st.caption(COMPLIANCE_NOTE)

    initial = float(INITIAL_CAPITAL)
    capital = get_virtual_capital()
    col1, col2, col3, col4 = st.columns(4)
    live_metrics = compute_all_metrics(initial)
    col1.metric("Virtual capital", "{:,.0f}".format(capital))
    col2.metric("Win rate", "{:.1%}".format(live_metrics["win_rate"]))
    col3.metric("Profit factor", "{:.2f}".format(live_metrics["profit_factor"]))
    col4.metric("Max drawdown", "{:.1%}".format(live_metrics["max_drawdown_pct"]))

    st.subheader("Success gates (project definition)")
    gate_ok = bool(live_metrics.get("success_gate"))
    st.write(
        "Gates (WR>60%, PF>1.5, MDD<10%): {}".format("PASS" if gate_ok else "FAIL")
    )

    tab1, tab2, tab3 = st.tabs(["Trades & equity", "Signals", "Stored snapshots"])

    with tab1:
        trades = _cached_trades()
        st.dataframe(trades, use_container_width=True)
        eq = _cached_equity(initial)
        if not eq.empty:
            eq = eq.copy()
            eq["peak"] = eq["equity"].cummax()
            eq["drawdown"] = (eq["peak"] - eq["equity"]) / eq["peak"]
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot(eq["exit_time"], eq["equity"], color="tab:blue")
            ax[0].set_ylabel("Equity")
            ax[0].grid(True, alpha=0.3)
            ax[1].fill_between(
                eq["exit_time"],
                eq["drawdown"],
                color="tab:red",
                alpha=0.3,
            )
            ax[1].set_ylabel("Drawdown")
            ax[1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No closed trades yet for equity/drawdown charts.")

    with tab2:
        sig = _cached_signals()
        st.dataframe(sig, use_container_width=True)

    with tab3:
        snap = latest_metrics()
        if snap:
            st.json(snap)
        else:
            st.info("No persisted metrics yet (run scheduler).")

    if st.button("Refresh data now"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Data refreshes from cache every ~30s or when you click Refresh. "
        "Timezone: {}".format(str(IST))
    )


if __name__ == "__main__":
    main()
