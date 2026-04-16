# -*- coding: utf-8 -*-
"""
Centralised SQLite database layer.

Responsibilities:
  - Connection factory with WAL mode and foreign-key enforcement.
  - Schema initialisation (signals, trades, performance_metrics, state).
  - Index creation on frequently-queried columns.
  - Atomic transaction context manager.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator, Optional

from src.config import DB_PATH, INITIAL_CAPITAL, ensure_directories

logger = logging.getLogger(__name__)

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """
    Return a thread-local SQLite connection with WAL mode and FK enforcement.

    Each thread gets its own connection; connections are reused within a thread.
    """
    conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            # Connection was closed; create a new one.
            pass

    ensure_directories()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    _local.conn = conn
    return conn


@contextmanager
def atomic() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for atomic transactions.

    Usage::

        with atomic() as conn:
            conn.execute("INSERT INTO ...")
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    row_hash        TEXT    UNIQUE,
    symbol          TEXT    NOT NULL,
    signal          TEXT    NOT NULL,
    buy_price       REAL,
    stop_loss       REAL,
    target          REAL,
    sheet_timestamp TEXT,
    fetched_at      TEXT,
    ml_confidence   REAL,
    ml_validated    INTEGER DEFAULT 0,
    status          TEXT    DEFAULT 'new'
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER REFERENCES signals(id),
    row_hash        TEXT,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    quantity        INTEGER,
    entry_price     REAL,
    exit_price      REAL,
    entry_time      TEXT,
    exit_time       TEXT,
    status          TEXT    DEFAULT 'pending',
    exit_reason     TEXT,
    pnl_gross       REAL,
    pnl_net         REAL,
    brokerage       REAL,
    slippage        REAL    DEFAULT 0.0,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at     TEXT,
    win_rate        REAL,
    total_trades    INTEGER,
    net_pnl         REAL,
    profit_factor   REAL,
    sharpe_ratio    REAL,
    max_drawdown_pct REAL,
    avg_rr          REAL,
    trades_per_day  REAL,
    success_gate    INTEGER,
    compliance_note TEXT
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);",
    "CREATE INDEX IF NOT EXISTS idx_signals_status_ml ON signals(status, ml_validated);",
    "CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);",
    "CREATE INDEX IF NOT EXISTS idx_signals_row_hash ON signals(row_hash);",
    "CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id);",
]


def ensure_database() -> None:
    """
    Create all tables and indexes if they do not exist.

    Safe to call multiple times; uses IF NOT EXISTS guards.
    """
    conn = get_connection()
    conn.executescript(_SCHEMA_SQL)
    for idx_sql in _INDEX_SQL:
        conn.execute(idx_sql)
    # Seed the virtual capital if not present.
    conn.execute(
        "INSERT OR IGNORE INTO state(key, value) VALUES ('virtual_capital', ?)",
        (str(INITIAL_CAPITAL),),
    )
    conn.commit()
    logger.debug("Database schema ensured at {}".format(DB_PATH))


def close_connection() -> None:
    """Close the thread-local connection if open."""
    conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


def reset_for_testing(db_path: str) -> None:
    """
    Override DB_PATH at runtime for test isolation.

    Must be called before any other db operations in the test.
    """
    import src.config as cfg

    cfg.DB_PATH = type(cfg.DB_PATH)(db_path)
    close_connection()
