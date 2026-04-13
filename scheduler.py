# -*- coding: utf-8 -*-
"""
Main orchestrator: fetch signals, ML validation hooks, paper trades, metrics.
Runs a loop with ~60s sleep during NSE hours (IST, Mon–Fri 09:00–16:00).
"""

import logging
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

import schedule

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import alerts
from src.config import INITIAL_CAPITAL, IST, LOG_LEVEL, MODEL_PATH, ensure_directories
from src.dummy_trader import (
    create_trades_for_validated_signals,
    ensure_database,
    get_virtual_capital,
    ingest_signals_df,
    simulate_step,
)
from src.fetch_sheet import fetch_signals_safe
from src.features_ml import retrain_model
from src.performance import compute_all_metrics, persist_metrics_row


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def is_nse_session_open(now=None) -> bool:
    """Return True during regular NSE cash session (09:00–16:00 IST, Mon–Fri)."""
    now = now or datetime.now(tz=IST)
    if now.weekday() >= 5:
        return False
    t = now.timetz().replace(tzinfo=None) if now.tzinfo else now.time()
    open_t = dtime(9, 0)
    close_t = dtime(16, 0)
    return open_t <= t < close_t


def _get_state(key: str, default: str = "") -> str:
    import sqlite3

    from src.config import DB_PATH

    with sqlite3.connect(str(DB_PATH)) as c:
        cur = c.execute("SELECT value FROM state WHERE key=?", (key,))
        row = cur.fetchone()
        return str(row[0]) if row and row[0] is not None else default


def _set_state(key: str, value: str):
    import sqlite3

    from src.config import DB_PATH

    with sqlite3.connect(str(DB_PATH)) as c:
        c.execute(
            """
            INSERT INTO state(key, value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )
        c.commit()


def maybe_send_daily_summary():
    """Once per calendar day (IST), send Telegram summary after first run past open."""
    today = datetime.now(tz=IST).strftime("%Y-%m-%d")
    last = _get_state("last_daily_summary_ymd", "")
    if last == today:
        return
    m = compute_all_metrics(INITIAL_CAPITAL)
    text = alerts.format_daily_summary(m)
    alerts.alert_daily_summary(text)
    _set_state("last_daily_summary_ymd", today)


def run_pipeline_cycle(prefer_sheet: bool = True):
    """Single end-to-end cycle (intended to run about every minute)."""
    ensure_directories()
    ensure_database()
    df = fetch_signals_safe(prefer_sheet=prefer_sheet)
    ingest_signals_df(df)
    create_trades_for_validated_signals()
    simulate_step()
    metrics = compute_all_metrics(INITIAL_CAPITAL)
    persist_metrics_row(metrics)
    maybe_send_daily_summary()
    logging.getLogger(__name__).info(
        "Cycle done: win_rate={:.2%} pf={:.2f} mdd={:.2%}".format(
            metrics["win_rate"],
            metrics["profit_factor"],
            metrics["max_drawdown_pct"],
        )
    )


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    setup_logging()
    log = logging.getLogger(__name__)
    ensure_directories()
    ensure_database()
    if not Path(MODEL_PATH).is_file():
        log.info("No RF model found; training bootstrap model")
        retrain_model()
    log.info("Scheduler started; NSE open window 09:00–16:00 IST")

    def _job():
        try:
            if is_nse_session_open():
                prefer = os.environ.get("USE_GOOGLE_SHEET", "1").strip() == "1"
                run_pipeline_cycle(prefer_sheet=prefer)
            else:
                log.debug("Outside NSE hours; scheduler tick skipped")
        except Exception as exc:
            log.error("Pipeline cycle error: {}".format(exc), exc_info=True)

    schedule.every(1).minutes.do(_job)
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        setup_logging()
        ensure_directories()
        ensure_database()
        if not Path(MODEL_PATH).is_file():
            retrain_model()
        prefer = os.environ.get("USE_GOOGLE_SHEET", "1").strip() == "1"
        run_pipeline_cycle(prefer_sheet=prefer)
        raise SystemExit(0)
    main()
