# -*- coding: utf-8 -*-
"""Paper trading simulation: pending → active → closed with risk and costs."""

import logging
import math
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import (
    BROKERAGE_PER_TRADE_INR,
    DB_PATH,
    IST,
    INITIAL_CAPITAL,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT_CAPITAL,
    ML_CONFIDENCE_THRESHOLD,
    RISK_PER_TRADE_PCT,
    ensure_directories,
)
from src import alerts
from src.features_ml import signal_confidence
from src.market_data import get_last_price

logger = logging.getLogger(__name__)


def _conn() -> sqlite3.Connection:
    ensure_directories()
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def ensure_database():
    """Create SQLite schema if not present."""
    ensure_directories()
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                row_hash TEXT UNIQUE,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                buy_price REAL,
                stop_loss REAL,
                target REAL,
                sheet_timestamp TEXT,
                fetched_at TEXT,
                ml_confidence REAL,
                ml_validated INTEGER DEFAULT 0,
                status TEXT DEFAULT 'new'
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                row_hash TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time TEXT,
                exit_time TEXT,
                status TEXT DEFAULT 'pending',
                exit_reason TEXT,
                pnl_gross REAL,
                pnl_net REAL,
                brokerage REAL,
                created_at TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at TEXT,
                win_rate REAL,
                total_trades INTEGER,
                net_pnl REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                avg_rr REAL,
                trades_per_day REAL,
                success_gate INTEGER,
                compliance_note TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        c.execute(
            """
            INSERT OR IGNORE INTO state(key, value) VALUES ('virtual_capital', ?)
            """,
            (str(INITIAL_CAPITAL),),
        )
        c.commit()


def get_virtual_capital() -> float:
    ensure_database()
    with _conn() as c:
        cur = c.execute("SELECT value FROM state WHERE key='virtual_capital'")
        row = cur.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    return float(INITIAL_CAPITAL)


def set_virtual_capital(val: float):
    with _conn() as c:
        c.execute(
            """
            INSERT INTO state(key, value) VALUES('virtual_capital', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (str(val),),
        )
        c.commit()


def count_open_trades() -> int:
    with _conn() as c:
        cur = c.execute(
            "SELECT COUNT(*) FROM trades WHERE status IN ('pending','active')"
        )
        return int(cur.fetchone()[0])


def _position_size(
    capital: float,
    entry: float,
    stop: float,
    side: str,
) -> float:
    """Shares/lots by 1% risk rule, capped by max position notional."""
    side_u = str(side).upper()
    if side_u == "BUY":
        risk_per_share = entry - stop
    else:
        risk_per_share = stop - entry
    if risk_per_share <= 0:
        return 0.0
    risk_inr = capital * RISK_PER_TRADE_PCT
    qty = risk_inr / risk_per_share
    max_notional = capital * MAX_POSITION_PCT_CAPITAL
    if entry > 0:
        cap_qty = max_notional / entry
        qty = min(qty, cap_qty)
    return max(0.0, float(qty))


def ingest_signals_df(df: pd.DataFrame) -> int:
    """
    Insert new signals from DataFrame (columns per fetch_sheet). Returns count inserted.
    """
    if df is None or df.empty:
        return 0
    ensure_database()
    inserted = 0
    now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        for _, row in df.iterrows():
            rh = str(row.get("row_hash", ""))
            if not rh:
                continue
            cur = c.execute("SELECT id FROM signals WHERE row_hash=?", (rh,))
            if cur.fetchone():
                continue
            try:
                bp = float(row["buy_price"])
                sl = float(row["stop_loss"])
                tg = float(row["target"])
            except (TypeError, ValueError):
                logger.warning("Skipping signal with invalid prices: {}".format(rh))
                continue
            if any(map(math.isnan, [bp, sl, tg])):
                continue
            conf = signal_confidence(
                str(row["symbol"]),
                str(row["signal"]),
            )
            validated = 1 if conf >= ML_CONFIDENCE_THRESHOLD else 0
            c.execute(
                """
                INSERT INTO signals(
                    row_hash, symbol, signal, buy_price, stop_loss, target,
                    sheet_timestamp, fetched_at, ml_confidence, ml_validated, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rh,
                    str(row["symbol"]),
                    str(row["signal"]),
                    float(row["buy_price"]),
                    float(row["stop_loss"]),
                    float(row["target"]),
                    str(row.get("timestamp", "")),
                    now,
                    float(conf),
                    validated,
                    "new",
                ),
            )
            inserted += 1
        c.commit()
    logger.info("Ingested {} new signals".format(inserted))
    return inserted


def create_trades_for_validated_signals() -> int:
    """For validated signals without trades, open pending trade rows."""
    ensure_database()
    created = 0
    capital = get_virtual_capital()
    with _conn() as c:
        open_n = count_open_trades()
        if open_n >= MAX_OPEN_POSITIONS:
            logger.warning("Max open positions reached; skip new trades")
            return 0
        cur = c.execute(
            """
            SELECT s.id, s.row_hash, s.symbol, s.signal, s.buy_price, s.stop_loss, s.target
            FROM signals s
            LEFT JOIN trades t ON t.signal_id = s.id
            WHERE s.ml_validated = 1 AND t.id IS NULL AND s.status = 'new'
            """
        )
        rows = cur.fetchall()
        for sid, rh, sym, side, buy, sl, tgt in rows:
            if open_n >= MAX_OPEN_POSITIONS:
                break
            qty = _position_size(capital, float(buy), float(sl), str(side))
            if qty <= 0:
                continue
            now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
            c.execute(
                """
                INSERT INTO trades(
                    signal_id, row_hash, symbol, side, quantity, entry_price,
                    exit_price, entry_time, exit_time, status, exit_reason,
                    pnl_gross, pnl_net, brokerage, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(sid),
                    str(rh),
                    str(sym),
                    str(side).upper(),
                    float(qty),
                    float(buy),
                    None,
                    None,
                    None,
                    "pending",
                    None,
                    0.0,
                    0.0,
                    0.0,
                    now,
                ),
            )
            c.execute(
                "UPDATE signals SET status='queued' WHERE id=?",
                (int(sid),),
            )
            created += 1
            open_n += 1
        c.commit()
    if created:
        logger.info("Created {} pending paper trades".format(created))
    return created


def _check_fill(side: str, last: float, entry: float) -> bool:
    su = str(side).upper()
    if su == "BUY":
        return last <= entry * 1.002
    return last >= entry * 0.998


def _check_exit_long(
    high: float,
    low: float,
    target: float,
    stop: float,
    entry: float,
) -> Tuple[Optional[float], Optional[str]]:
    hit_sl = low <= stop
    hit_tg = high >= target
    if hit_sl and hit_tg:
        return stop, "stop_loss"
    if hit_sl:
        return stop, "stop_loss"
    if hit_tg:
        return target, "target"
    return None, None


def _check_exit_short(
    high: float,
    low: float,
    target: float,
    stop: float,
    entry: float,
) -> Tuple[Optional[float], Optional[str]]:
    hit_sl = high >= stop
    hit_tg = low <= target
    if hit_sl and hit_tg:
        return stop, "stop_loss"
    if hit_sl:
        return stop, "stop_loss"
    if hit_tg:
        return target, "target"
    return None, None


def simulate_step() -> Dict[str, int]:
    """
    One pipeline step: try fills and exits using last price as proxy for bar H/L.
    Uses last price for both low and high approximation when intraday OHLC unavailable.
    """
    ensure_database()
    stats = {"filled": 0, "closed": 0}
    capital = get_virtual_capital()
    with _conn() as c:
        cur = c.execute(
            """
            SELECT t.id, t.symbol, t.side, t.quantity, t.entry_price, t.status,
                   s.stop_loss, s.target
            FROM trades t
            JOIN signals s ON s.id = t.signal_id
            WHERE t.status IN ('pending','active')
            """
        )
        rows = cur.fetchall()
        for (
            tid,
            sym,
            side,
            qty,
            entry,
            status,
            sl,
            tgt,
        ) in rows:
            last = get_last_price(str(sym))
            if last is None:
                continue
            high = max(float(last), float(entry))
            low = min(float(last), float(entry))
            su = str(side).upper()
            if status == "pending":
                if _check_fill(su, float(last), float(entry)):
                    now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
                    c.execute(
                        """
                        UPDATE trades SET status='active', entry_time=?, entry_price=?
                        WHERE id=?
                        """,
                        (now, float(entry), int(tid)),
                    )
                    stats["filled"] += 1
                    try:
                        alerts.alert_trade_entry(
                            str(sym), str(side), float(qty), float(entry)
                        )
                    except Exception:
                        logger.debug("Entry alert skipped", exc_info=True)
                continue
            if status == "active":
                ep = float(entry)
                if su == "BUY":
                    px, reason = _check_exit_long(
                        high, low, float(tgt), float(sl), ep
                    )
                    if px is None:
                        continue
                    pnl_g = (float(px) - ep) * float(qty)
                else:
                    px, reason = _check_exit_short(
                        high, low, float(tgt), float(sl), ep
                    )
                    if px is None:
                        continue
                    pnl_g = (ep - float(px)) * float(qty)
                brk = BROKERAGE_PER_TRADE_INR
                pnl_n = pnl_g - brk
                now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
                c.execute(
                    """
                    UPDATE trades SET status='closed', exit_price=?, exit_time=?,
                    exit_reason=?, pnl_gross=?, pnl_net=?, brokerage=?
                    WHERE id=?
                    """,
                    (float(px), now, str(reason), float(pnl_g), float(pnl_n), float(brk), int(tid)),
                )
                c.execute(
                    "UPDATE signals SET status='done' WHERE id=(SELECT signal_id FROM trades WHERE id=?)",
                    (int(tid),),
                )
                capital = capital + pnl_n
                stats["closed"] += 1
                try:
                    alerts.alert_trade_exit(
                        str(sym), str(reason), float(pnl_n)
                    )
                except Exception:
                    logger.debug("Exit alert skipped", exc_info=True)
        c.commit()
    set_virtual_capital(capital)
    return stats


def list_recent_trades(limit: int = 50) -> pd.DataFrame:
    ensure_database()
    with _conn() as c:
        return pd.read_sql(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
            c,
            params=(int(limit),),
        )


def list_recent_signals(limit: int = 50) -> pd.DataFrame:
    ensure_database()
    with _conn() as c:
        return pd.read_sql(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
            c,
            params=(int(limit),),
        )
