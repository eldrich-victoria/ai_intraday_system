# -*- coding: utf-8 -*-
"""Paper trading simulation: pending -> active -> closed with risk and costs."""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import (
    BROKERAGE_PER_LEG_INR,
    BROKERAGE_PER_TRADE_INR,
    IST,
    INITIAL_CAPITAL,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT_CAPITAL,
    ML_CONFIDENCE_THRESHOLD,
    RISK_PER_TRADE_PCT,
    SLIPPAGE_PCT,
    ensure_directories,
)
from src.db import atomic, ensure_database, get_connection
from src import alerts
from src.features_ml import signal_confidence
from src.market_data import get_last_price

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Virtual capital (state table)
# ---------------------------------------------------------------------------

def get_virtual_capital() -> float:
    """Read current virtual capital from the state table."""
    ensure_database()
    conn = get_connection()
    cur = conn.execute("SELECT value FROM state WHERE key='virtual_capital'")
    row = cur.fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return float(INITIAL_CAPITAL)


def set_virtual_capital(val: float) -> None:
    """Persist virtual capital to the state table."""
    with atomic() as conn:
        conn.execute(
            """
            INSERT INTO state(key, value) VALUES('virtual_capital', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (str(val),),
        )


# ---------------------------------------------------------------------------
# Position queries
# ---------------------------------------------------------------------------

def count_open_trades() -> int:
    """Count pending and active trades."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status IN ('pending','active')"
    )
    return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def _position_size(
    capital: float,
    entry: float,
    stop: float,
    side: str,
) -> int:
    """
    Shares by 1% risk rule, capped by max position notional.

    Returns an **integer** quantity (you cannot buy fractional shares on NSE).
    """
    side_u = str(side).upper()
    if side_u == "BUY":
        risk_per_share = entry - stop
    else:
        risk_per_share = stop - entry
    if risk_per_share <= 0:
        return 0
    risk_inr = capital * RISK_PER_TRADE_PCT
    qty = risk_inr / risk_per_share
    max_notional = capital * MAX_POSITION_PCT_CAPITAL
    if entry > 0:
        cap_qty = max_notional / entry
        qty = min(qty, cap_qty)
    return max(0, math.floor(qty))


def _apply_slippage(price: float, side: str, direction: str) -> float:
    """
    Apply slippage to a price.

    Args:
        price: The base price.
        side: "BUY" or "SELL" (trade side).
        direction: "entry" or "exit".

    Returns:
        Adjusted price with slippage.
    """
    if SLIPPAGE_PCT <= 0:
        return price
    if side == "BUY":
        # Entry: pay more; Exit: receive less adversely not applicable
        # for long, exit could be at SL (lower) or TGT (higher).
        if direction == "entry":
            return price * (1.0 + SLIPPAGE_PCT)
        return price * (1.0 - SLIPPAGE_PCT)
    else:
        # SELL (short): Entry: receive less; Exit: pay more.
        if direction == "entry":
            return price * (1.0 - SLIPPAGE_PCT)
        return price * (1.0 + SLIPPAGE_PCT)


# ---------------------------------------------------------------------------
# Signal validation
# ---------------------------------------------------------------------------

def _validate_signal(
    symbol: str, signal: str, buy_price: float, stop_loss: float, target: float
) -> bool:
    """
    Validate signal integrity before ingestion.

    Rules:
      - All prices must be positive.
      - BUY: stop_loss < buy_price < target.
      - SELL: stop_loss > buy_price > target.
    """
    if buy_price <= 0 or stop_loss <= 0 or target <= 0:
        logger.warning(
            "Signal rejected: non-positive price for {}".format(symbol)
        )
        return False
    side = str(signal).upper().strip()
    if side == "BUY" and not (stop_loss < buy_price < target):
        logger.warning(
            "BUY signal rejected for {}: SL({}) >= Entry({}) or Entry >= TGT({})".format(
                symbol, stop_loss, buy_price, target
            )
        )
        return False
    if side == "SELL" and not (stop_loss > buy_price > target):
        logger.warning(
            "SELL signal rejected for {}: SL({}) <= Entry({}) or Entry <= TGT({})".format(
                symbol, stop_loss, buy_price, target
            )
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Signal ingestion
# ---------------------------------------------------------------------------

def ingest_signals_df(df: pd.DataFrame) -> int:
    """
    Insert new signals from DataFrame (columns per fetch_sheet).
    Returns count inserted.
    """
    if df is None or df.empty:
        return 0
    ensure_database()
    inserted = 0
    now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
    with atomic() as conn:
        for _, row in df.iterrows():
            rh = str(row.get("row_hash", ""))
            if not rh:
                continue
            cur = conn.execute("SELECT id FROM signals WHERE row_hash=?", (rh,))
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
            # Validate signal integrity.
            if not _validate_signal(str(row["symbol"]), str(row["signal"]), bp, sl, tg):
                continue
            conf = signal_confidence(
                str(row["symbol"]),
                str(row["signal"]),
            )
            validated = 1 if conf >= ML_CONFIDENCE_THRESHOLD else 0
            conn.execute(
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
                    bp,
                    sl,
                    tg,
                    str(row.get("timestamp", "")),
                    now,
                    float(conf),
                    validated,
                    "new",
                ),
            )
            inserted += 1
    logger.info("Ingested {} new signals".format(inserted))
    return inserted


# ---------------------------------------------------------------------------
# Trade creation
# ---------------------------------------------------------------------------

def create_trades_for_validated_signals() -> int:
    """For validated signals without trades, open pending trade rows."""
    ensure_database()
    created = 0
    capital = get_virtual_capital()
    with atomic() as conn:
        open_n = count_open_trades()
        if open_n >= MAX_OPEN_POSITIONS:
            logger.warning("Max open positions reached; skip new trades")
            return 0
        cur = conn.execute(
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
            # Apply slippage to entry price.
            entry_with_slippage = _apply_slippage(float(buy), str(side).upper(), "entry")
            # Deduct entry-leg brokerage from capital tracking.
            now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                INSERT INTO trades(
                    signal_id, row_hash, symbol, side, quantity, entry_price,
                    exit_price, entry_time, exit_time, status, exit_reason,
                    pnl_gross, pnl_net, brokerage, slippage, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(sid),
                    str(rh),
                    str(sym),
                    str(side).upper(),
                    int(qty),
                    float(entry_with_slippage),
                    None,
                    None,
                    None,
                    "pending",
                    None,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    now,
                ),
            )
            conn.execute(
                "UPDATE signals SET status='queued' WHERE id=?",
                (int(sid),),
            )
            created += 1
            open_n += 1
    if created:
        logger.info("Created {} pending paper trades".format(created))
    return created


# ---------------------------------------------------------------------------
# Fill / exit logic
# ---------------------------------------------------------------------------

def _check_fill(side: str, last: float, entry: float) -> bool:
    """Check if a pending order should be filled given the last price."""
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
    """Check long exit conditions."""
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
    """Check short exit conditions."""
    hit_sl = high >= stop
    hit_tg = low <= target
    if hit_sl and hit_tg:
        return stop, "stop_loss"
    if hit_sl:
        return stop, "stop_loss"
    if hit_tg:
        return target, "target"
    return None, None


# ---------------------------------------------------------------------------
# Simulation step
# ---------------------------------------------------------------------------

def simulate_step() -> Dict[str, int]:
    """
    One pipeline step: try fills and exits using last price as proxy for bar H/L.
    Uses last price for both low and high approximation when intraday OHLC unavailable.
    """
    ensure_database()
    stats: Dict[str, int] = {"filled": 0, "closed": 0}
    capital = get_virtual_capital()
    with atomic() as conn:
        cur = conn.execute(
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
                    # Apply entry-leg brokerage.
                    conn.execute(
                        """
                        UPDATE trades SET status='active', entry_time=?,
                        entry_price=?, brokerage=?
                        WHERE id=?
                        """,
                        (now, float(entry), float(BROKERAGE_PER_LEG_INR), int(tid)),
                    )
                    stats["filled"] += 1
                    try:
                        alerts.alert_trade_entry(
                            str(sym), str(side), int(qty), float(entry)
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
                    # Apply slippage to exit.
                    exit_px = _apply_slippage(float(px), su, "exit")
                    pnl_g = (exit_px - ep) * int(qty)
                else:
                    px, reason = _check_exit_short(
                        high, low, float(tgt), float(sl), ep
                    )
                    if px is None:
                        continue
                    exit_px = _apply_slippage(float(px), su, "exit")
                    pnl_g = (ep - exit_px) * int(qty)
                # Round-trip brokerage: entry leg already recorded, add exit leg.
                brk_total = float(BROKERAGE_PER_TRADE_INR)
                slippage_cost = abs(exit_px - float(px)) * int(qty)
                pnl_n = pnl_g - brk_total
                now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    UPDATE trades SET status='closed', exit_price=?, exit_time=?,
                    exit_reason=?, pnl_gross=?, pnl_net=?, brokerage=?,
                    slippage=?
                    WHERE id=?
                    """,
                    (
                        float(exit_px),
                        now,
                        str(reason),
                        float(pnl_g),
                        float(pnl_n),
                        float(brk_total),
                        float(slippage_cost),
                        int(tid),
                    ),
                )
                conn.execute(
                    "UPDATE signals SET status='done' WHERE id="
                    "(SELECT signal_id FROM trades WHERE id=?)",
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
    set_virtual_capital(capital)
    return stats


# ---------------------------------------------------------------------------
# End-of-day forced closure
# ---------------------------------------------------------------------------

def force_close_eod() -> int:
    """
    Close all active trades at the current last price (end-of-day closure).

    Intraday positions should not be held overnight; this ensures all
    active positions are squared off before market close.

    Returns:
        Number of trades force-closed.
    """
    ensure_database()
    closed = 0
    capital = get_virtual_capital()
    with atomic() as conn:
        cur = conn.execute(
            """
            SELECT t.id, t.symbol, t.side, t.quantity, t.entry_price
            FROM trades t
            WHERE t.status = 'active'
            """
        )
        rows = cur.fetchall()
        for tid, sym, side, qty, entry in rows:
            last = get_last_price(str(sym))
            if last is None:
                logger.warning(
                    "EOD close skipped for {} (no price available)".format(sym)
                )
                continue
            su = str(side).upper()
            exit_px = _apply_slippage(float(last), su, "exit")
            if su == "BUY":
                pnl_g = (exit_px - float(entry)) * int(qty)
            else:
                pnl_g = (float(entry) - exit_px) * int(qty)
            brk_total = float(BROKERAGE_PER_TRADE_INR)
            slippage_cost = abs(exit_px - float(last)) * int(qty)
            pnl_n = pnl_g - brk_total
            now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                UPDATE trades SET status='closed', exit_price=?, exit_time=?,
                exit_reason=?, pnl_gross=?, pnl_net=?, brokerage=?,
                slippage=?
                WHERE id=?
                """,
                (
                    float(exit_px),
                    now,
                    "eod_forced",
                    float(pnl_g),
                    float(pnl_n),
                    float(brk_total),
                    float(slippage_cost),
                    int(tid),
                ),
            )
            conn.execute(
                "UPDATE signals SET status='done' WHERE id="
                "(SELECT signal_id FROM trades WHERE id=?)",
                (int(tid),),
            )
            capital = capital + pnl_n
            closed += 1
            logger.info(
                "EOD forced close: {} {} qty={} pnl_net={:.2f}".format(
                    sym, side, qty, pnl_n
                )
            )
    if closed:
        set_virtual_capital(capital)
    return closed


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def list_recent_trades(limit: int = 50) -> pd.DataFrame:
    """Return recent trades as a DataFrame."""
    ensure_database()
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
        conn,
        params=(int(limit),),
    )


def list_recent_signals(limit: int = 50) -> pd.DataFrame:
    """Return recent signals as a DataFrame."""
    ensure_database()
    conn = get_connection()
    return pd.read_sql(
        "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
        conn,
        params=(int(limit),),
    )
