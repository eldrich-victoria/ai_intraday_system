# -*- coding: utf-8 -*-
"""Telegram notifications for entries, exits, and daily summaries."""

import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum retry attempts for Telegram API calls.
_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


async def _send_async(token: str, chat_id: str, text: str) -> bool:
    """Send a single Telegram message asynchronously."""
    try:
        from telegram import Bot

        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text)
        return True
    except Exception as exc:
        logger.error("Telegram send failed: {}".format(exc), exc_info=True)
        return False


def _run_async(coro) -> bool:
    """Run an async coroutine from synchronous context."""
    try:
        return bool(asyncio.run(coro))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return bool(loop.run_until_complete(coro))
        finally:
            loop.close()


def send_telegram_message(text: str) -> bool:
    """
    Send a message using python-telegram-bot with retry and backoff.

    Returns True if credentials are present and send succeeded.
    """
    token = os.environ.get("BOT_TOKEN", "").strip()
    chat_id = os.environ.get("CHAT_ID", "").strip()
    if not token or not chat_id:
        logger.info("Telegram disabled (BOT_TOKEN or CHAT_ID missing)")
        return False

    delay = _BASE_DELAY
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = _run_async(_send_async(token, chat_id, text))
            if result:
                return True
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Telegram attempt {}/{} failed ({}). Retrying in {:.1f}s...".format(
                    attempt, _MAX_RETRIES, exc, delay
                )
            )

        if attempt < _MAX_RETRIES:
            time.sleep(delay)
            delay = min(delay * _BACKOFF_FACTOR, 30.0)

    if last_exc:
        logger.error(
            "All {} Telegram retries exhausted: {}".format(_MAX_RETRIES, last_exc)
        )
    return False


def alert_trade_entry(
    symbol: str, side: str, qty: int, price: float
) -> bool:
    """Alert on paper trade entry."""
    msg = (
        "Paper ENTRY\n"
        "Symbol: {}\n"
        "Side: {}\n"
        "Qty: {}\n"
        "Price: {:.2f}"
    ).format(symbol, side, qty, price)
    return send_telegram_message(msg)


def alert_trade_exit(
    symbol: str,
    reason: str,
    pnl_net: float,
) -> bool:
    """Alert on paper trade exit."""
    emoji = "\u2705" if pnl_net >= 0 else "\u274c"
    msg = (
        "Paper EXIT {}\n"
        "Symbol: {}\n"
        "Reason: {}\n"
        "Net P&L: {:.2f}"
    ).format(emoji, symbol, reason, pnl_net)
    return send_telegram_message(msg)


def alert_daily_summary(summary_text: str) -> bool:
    """Send daily performance summary."""
    return send_telegram_message(summary_text)


def format_daily_summary(metrics: dict) -> str:
    """Build human-readable daily summary from metrics dict."""
    lines = [
        "\U0001f4ca Daily Paper-Trading Summary",
        "Win rate: {:.2%}".format(float(metrics.get("win_rate", 0.0))),
        "Profit factor: {:.2f}".format(float(metrics.get("profit_factor", 0.0))),
        "Net PnL: \u20b9{:.2f}".format(float(metrics.get("net_pnl", 0.0))),
        "Max DD: {:.2%}".format(float(metrics.get("max_drawdown_pct", 0.0))),
        "Sharpe (trade approx): {:.2f}".format(
            float(metrics.get("sharpe_ratio", 0.0))
        ),
        "Total trades: {}".format(int(metrics.get("total_trades", 0))),
        "Success gate: {}".format(
            "\u2705 PASS" if metrics.get("success_gate") else "\u274c FAIL"
        ),
    ]
    return "\n".join(lines)
