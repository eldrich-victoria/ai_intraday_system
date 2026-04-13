# -*- coding: utf-8 -*-
"""Telegram notifications for entries, exits, and daily summaries."""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


async def _send_async(token: str, chat_id: str, text: str) -> bool:
    try:
        from telegram import Bot

        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text)
        return True
    except Exception as exc:
        logger.error("Telegram send failed: {}".format(exc), exc_info=True)
        return False


def send_telegram_message(text: str) -> bool:
    """
    Send a message using python-telegram-bot (async under the hood).
    Returns True if credentials present and send attempted successfully.
    """
    token = os.environ.get("BOT_TOKEN", "").strip()
    chat_id = os.environ.get("CHAT_ID", "").strip()
    if not token or not chat_id:
        logger.info("Telegram disabled (BOT_TOKEN or CHAT_ID missing)")
        return False
    try:
        return bool(asyncio.run(_send_async(token, chat_id, text)))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return bool(loop.run_until_complete(_send_async(token, chat_id, text)))
        finally:
            loop.close()


def alert_trade_entry(symbol: str, side: str, qty: float, price: float):
    msg = "Paper ENTRY: {} {} qty={} @ {}".format(symbol, side, qty, price)
    return send_telegram_message(msg)


def alert_trade_exit(
    symbol: str,
    reason: str,
    pnl_net: float,
):
    msg = "Paper EXIT: {} reason={} net_pnl={}".format(symbol, reason, pnl_net)
    return send_telegram_message(msg)


def alert_daily_summary(summary_text: str):
    return send_telegram_message(summary_text)


def format_daily_summary(metrics: dict) -> str:
    """Build human-readable daily summary from metrics dict."""
    lines = [
        "Daily paper-trading summary",
        "Win rate: {:.2%}".format(float(metrics.get("win_rate", 0.0))),
        "Profit factor: {:.2f}".format(float(metrics.get("profit_factor", 0.0))),
        "Net PnL: {:.2f}".format(float(metrics.get("net_pnl", 0.0))),
        "Max DD: {:.2%}".format(float(metrics.get("max_drawdown_pct", 0.0))),
        "Sharpe (trade approx): {:.2f}".format(float(metrics.get("sharpe_ratio", 0.0))),
        "Success gate: {}".format("PASS" if metrics.get("success_gate") else "FAIL"),
    ]
    return "\n".join(lines)
