# -*- coding: utf-8 -*-
"""Configuration variables and constants for the AI Intraday Trading Tester System."""

import os
from pathlib import Path
from typing import FrozenSet
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Paths (project root = parent of src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "trades.db"
MODEL_PATH = MODELS_DIR / "rf_model_GLOBAL.pkl"
MOCK_SIGNALS_PATH = DATA_DIR / "mock_signals.csv"
LOCK_FILE = PROJECT_ROOT / "scheduler.lock"

GOOGLE_SHEET_URL = os.environ.get(
    "GOOGLE_SHEET_URL",
    "https://docs.google.com/spreadsheets/d/1mw5tL2q98s-TcmLfb2U_uFKKQZA33WFd-GynZA0_XaU/edit",
)

# ---------------------------------------------------------------------------
# NSE session (IST)
# ---------------------------------------------------------------------------
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# NSE holidays for 2025–2026 (dates in MM-DD format for the given year).
# Source: https://www.nseindia.com/resources/exchange-communication-holidays
NSE_HOLIDAYS_2025: FrozenSet[str] = frozenset({
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Maha Shivaratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-ul-Fitr (Eid)
    "2025-04-10",  # Shri Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # May Day / Maharashtra Day
    "2025-06-07",  # Bakri Eid (Eid-ul-Adha)
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Mahatma Gandhi Jayanti
    "2025-10-21",  # Diwali Amavasya (Laxmi Pooja)
    "2025-10-22",  # Diwali Balipratipada
    "2025-11-05",  # Guru Nanak Jayanti (Prakash Gurpurab)
    "2025-12-25",  # Christmas Day
})

NSE_HOLIDAYS_2026: FrozenSet[str] = frozenset({
    "2026-01-26",  # Republic Day
    "2026-02-17",  # Maha Shivaratri
    "2026-03-03",  # Holi
    "2026-03-20",  # Id-ul-Fitr (Eid)
    "2026-03-30",  # Shri Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # May Day / Maharashtra Day
    "2026-05-28",  # Bakri Eid (Eid-ul-Adha)
    "2026-08-15",  # Independence Day
    "2026-08-17",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-09",  # Dussehra
    "2026-10-28",  # Diwali Amavasya (Laxmi Pooja)
    "2026-11-24",  # Guru Nanak Jayanti (Prakash Gurpurab)
    "2026-12-25",  # Christmas Day
})

NSE_HOLIDAYS: FrozenSet[str] = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026

# ---------------------------------------------------------------------------
# Paper trading
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "100000"))
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "0.01"))
# Round-trip brokerage: Rs 20 entry + Rs 20 exit = Rs 40 total.
BROKERAGE_PER_TRADE_INR = float(os.environ.get("BROKERAGE_PER_TRADE_INR", "40"))
BROKERAGE_PER_LEG_INR = BROKERAGE_PER_TRADE_INR / 2.0
# Slippage as fraction of price (0.05% = 0.0005).
SLIPPAGE_PCT = float(os.environ.get("SLIPPAGE_PCT", "0.0005"))
ML_CONFIDENCE_THRESHOLD = float(os.environ.get("ML_CONFIDENCE_THRESHOLD", "0.7"))
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))
MAX_POSITION_PCT_CAPITAL = float(os.environ.get("MAX_POSITION_PCT_CAPITAL", "0.2"))

# ---------------------------------------------------------------------------
# Success criteria (after costs), for reporting / gates
# ---------------------------------------------------------------------------
TARGET_WIN_RATE = 0.60
TARGET_PROFIT_FACTOR = 1.5
TARGET_MAX_DRAWDOWN_PCT = 0.10

# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------
YF_CACHE_SECONDS = int(os.environ.get("YF_CACHE_SECONDS", "45"))
NSE_SUFFIX = ".NS"

# ---------------------------------------------------------------------------
# ML / features
# ---------------------------------------------------------------------------
RF_RANDOM_STATE = 42
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", "200"))
TS_SPLIT_SPLITS = 5
LOOKBACK_BARS = 120

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# SEBI-aligned paper-trading disclaimers (no real orders)
# ---------------------------------------------------------------------------
COMPLIANCE_NOTE = (
    "Educational paper simulation only. Not investment advice. "
    "Past performance does not guarantee future results. "
    "This system does not place real orders or connect to any broker. "
    "Please comply with SEBI regulations for research and advisory."
)


def ensure_directories() -> None:
    """Create data, models, and logs directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
