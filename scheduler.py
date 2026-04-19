# -*- coding: utf-8 -*-
"""
Main orchestrator: fetch signals, ML validation hooks, paper trades, metrics.
Runs a loop with ~60s sleep during NSE hours (IST, Mon-Fri 09:00-16:00).

Usage:
    python scheduler.py           # Continuous loop (local / cloud VM)
    python scheduler.py --once    # Single cycle (GitHub Actions / cron)
    python scheduler.py --retrain # Force model retrain and exit
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schedule

from src import alerts
from src.config import (
    INITIAL_CAPITAL,
    IST,
    LOCK_FILE,
    LOG_DIR,
    LOG_LEVEL,
    MODEL_PATH,
    NSE_HOLIDAYS,
    ensure_directories,
)
from src.db import ensure_database, get_connection
from src.dummy_trader import (
    create_trades_for_validated_signals,
    force_close_eod,
    ingest_signals_df,
    simulate_step,
)
from src.fetch_sheet import fetch_signals_safe
from src.features_ml import retrain_model
from src.performance import compute_all_metrics, persist_metrics_row

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure root logger with console and rotating file handlers."""
    ensure_directories()
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(console)

    # Rotating file handler (5 MB per file, 3 backups).
    log_path = LOG_DIR / "scheduler.log"
    try:
        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
    except Exception as exc:
        root.warning("Could not create log file {}: {}".format(log_path, exc))


# ---------------------------------------------------------------------------
# Market hours and holiday awareness
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    """
    Return True during regular NSE cash session.

    Checks:
      - Weekday (Mon-Fri).
      - Not an NSE holiday.
      - Between 09:00 and 16:00 IST.
    """
    now = datetime.now(tz=IST)
    # Weekend check.
    if now.weekday() >= 5:
        return False
    # Holiday check.
    today_str = now.strftime("%Y-%m-%d")
    if today_str in NSE_HOLIDAYS:
        return False
    # Time window check.
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def is_near_market_close() -> bool:
    """Return True if within 5 minutes of market close (15:55-16:00 IST)."""
    now = datetime.now(tz=IST)
    close_warn = now.replace(hour=15, minute=55, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return close_warn <= now <= close_time


# ---------------------------------------------------------------------------
# Scheduler lock (prevents duplicate executions)
# ---------------------------------------------------------------------------

def acquire_lock() -> bool:
    """
    Acquire a file-based lock to prevent duplicate scheduler runs.

    Returns True if the lock was acquired, False if another instance holds it.
    """
    if LOCK_FILE.exists():
        # Check if the lock is stale (older than 15 minutes).
        try:
            lock_age = time.time() - LOCK_FILE.stat().st_mtime
            if lock_age < 900:  # 15 minutes
                return False
            # Stale lock; remove and re-acquire.
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            return False

    try:
        LOCK_FILE.write_text(
            str(os.getpid()), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def release_lock() -> None:
    """Release the file-based scheduler lock."""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _get_state(key: str, default: str = "") -> str:
    """Read a value from the state table."""
    conn = get_connection()
    cur = conn.execute("SELECT value FROM state WHERE key=?", (key,))
    row = cur.fetchone()
    return str(row[0]) if row and row[0] is not None else default


def _set_state(key: str, value: str) -> None:
    """Write a value to the state table."""
    from src.db import atomic
    with atomic() as conn:
        conn.execute(
            """
            INSERT INTO state(key, value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )


# ---------------------------------------------------------------------------
# Daily summary (once per day)
# ---------------------------------------------------------------------------

def maybe_send_daily_summary() -> None:
    """Once per calendar day (IST), send Telegram summary after first run past open."""
    today = datetime.now(tz=IST).strftime("%Y-%m-%d")
    last = _get_state("last_daily_summary_ymd", "")
    if last == today:
        return
    m = compute_all_metrics(INITIAL_CAPITAL)
    text = alerts.format_daily_summary(m)
    alerts.alert_daily_summary(text)
    _set_state("last_daily_summary_ymd", today)


# ---------------------------------------------------------------------------
# Pipeline cycle
# ---------------------------------------------------------------------------

def run_pipeline_cycle(prefer_sheet: bool = True) -> None:
    """Single end-to-end cycle (intended to run about every minute)."""
    log = logging.getLogger(__name__)
    ensure_directories()
    ensure_database()

    df = fetch_signals_safe(prefer_sheet=prefer_sheet)
    ingest_signals_df(df)
    create_trades_for_validated_signals()
    simulate_step()

    # Force close active positions near market close.
    if is_near_market_close():
        closed = force_close_eod()
        if closed:
            log.info("EOD forced closure: {} trades closed".format(closed))

    metrics = compute_all_metrics(INITIAL_CAPITAL)
    persist_metrics_row(metrics)
    maybe_send_daily_summary()

    log.info(
        "Cycle done: win_rate={:.2%} pf={:.2f} mdd={:.2%} trades={}".format(
            metrics["win_rate"],
            metrics["profit_factor"],
            metrics["max_drawdown_pct"],
            metrics["total_trades"],
        )
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main_loop() -> None:
    """Continuous scheduler loop for local or cloud VM deployment."""
    log = logging.getLogger(__name__)

    if not acquire_lock():
        log.error("Another scheduler instance is running. Exiting.")
        raise SystemExit(1)

    try:
        log.info("Scheduler started; NSE open window 09:00-16:00 IST")

        def _job() -> None:
            try:
                if is_market_open():
                    prefer = os.environ.get("USE_GOOGLE_SHEET", "1").strip() == "1"
                    run_pipeline_cycle(prefer_sheet=prefer)
                else:
                    log.debug("Outside NSE hours; scheduler tick skipped")
            except Exception as exc:
                log.error("Pipeline cycle error: {}".format(exc), exc_info=True)

        schedule.every(5).minutes.do(_job)
        while True:
            schedule.run_pending()
            time.sleep(1)
    finally:
        release_lock()


def run_realtime_loop() -> None:
    """Bounded loop for simulated real-time execution in CI environments."""
    log = logging.getLogger(__name__)

    if not acquire_lock():
        log.warning("Another scheduler instance is running. Exiting.")
        raise SystemExit(0)

    try:
        cycles = 5
        log.info("Realtime loop started")
        
        for i in range(cycles):
            log.info(f"Cycle {i+1}/{cycles} started")
            
            if not is_market_open():
                log.info("Market closed - exiting loop")
                break
            
            try:
                prefer = os.environ.get("USE_GOOGLE_SHEET", "1").strip() == "1"
                run_pipeline_cycle(prefer_sheet=prefer)
            except Exception as exc:
                log.error("Pipeline cycle error: {}".format(exc), exc_info=True)
            
            log.info(f"Cycle {i+1}/{cycles} completed")
            
            if i < cycles - 1:
                log.info("Sleeping for 120 seconds")
                time.sleep(120)
    finally:
        release_lock()


def main_once() -> None:
    """Single-run mode for GitHub Actions / cron."""
    log = logging.getLogger(__name__)

    if not acquire_lock():
        log.warning("Another scheduler instance is running. Exiting.")
        raise SystemExit(0)

    try:
        if is_market_open():
            prefer = os.environ.get("USE_GOOGLE_SHEET", "1").strip() == "1"
            run_pipeline_cycle(prefer_sheet=prefer)
        else:
            log.info("Market is closed; --once cycle skipped.")
    finally:
        release_lock()


def main_retrain() -> None:
    """Force model retrain and exit."""
    log = logging.getLogger(__name__)
    log.info("Forcing model retrain...")
    retrain_model()
    log.info("Retrain complete.")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="AI Intraday Trading Tester Scheduler"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pipeline cycle and exit (for CI/cron).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force ML model retrain and exit.",
    )
    args = parser.parse_args()

    setup_logging()
    ensure_directories()
    ensure_database()

    # Ensure model exists.
    if not Path(MODEL_PATH).is_file() and not args.retrain:
        logging.getLogger(__name__).info(
            "No RF model found; training bootstrap model"
        )
        retrain_model()

    if args.retrain:
        main_retrain()
    elif args.once:
        main_once()
    else:
        run_realtime_loop()