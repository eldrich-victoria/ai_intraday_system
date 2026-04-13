# -*- coding: utf-8 -*-
"""Configuration variables and constants for the AI Intraday Trading Tester System."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

# Paths (project root = parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "trades.db"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
MOCK_SIGNALS_PATH = DATA_DIR / "mock_signals.csv"
GOOGLE_SHEET_URL = os.environ.get(
    "GOOGLE_SHEET_URL",
    "https://docs.google.com/spreadsheets/d/1mw5tL2q98s-TcmLfb2U_uFKKQZA33WFd-GynZA0_XaU/edit",
)

# NSE session (IST)
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Paper trading
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "100000"))
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "0.01"))
BROKERAGE_PER_TRADE_INR = float(os.environ.get("BROKERAGE_PER_TRADE_INR", "20"))
ML_CONFIDENCE_THRESHOLD = float(os.environ.get("ML_CONFIDENCE_THRESHOLD", "0.7"))
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))
MAX_POSITION_PCT_CAPITAL = float(os.environ.get("MAX_POSITION_PCT_CAPITAL", "0.2"))

# Success criteria (after costs), for reporting / gates
TARGET_WIN_RATE = 0.60
TARGET_PROFIT_FACTOR = 1.5
TARGET_MAX_DRAWDOWN_PCT = 0.10

# yfinance
YF_CACHE_SECONDS = int(os.environ.get("YF_CACHE_SECONDS", "45"))
NSE_SUFFIX = ".NS"

# ML / features
RF_RANDOM_STATE = 42
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", "200"))
TS_SPLIT_SPLITS = 5
LOOKBACK_BARS = 120

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# SEBI-aligned paper-trading disclaimers (no real orders)
COMPLIANCE_NOTE = (
    "Educational paper simulation only. Not investment advice. "
    "Past performance does not guarantee future results."
)


def ensure_directories():
    """Create data and models directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
