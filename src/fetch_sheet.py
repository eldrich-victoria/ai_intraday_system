# -*- coding: utf-8 -*-
"""Fetch live trading signals from Google Sheets using gspread (service account)."""

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import GOOGLE_SHEET_URL, IST, MOCK_SIGNALS_PATH
from src.retry import retry_with_backoff

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "symbol",
    "signal",
    "buy_price",
    "stop_loss",
    "target",
    "timestamp",
    "row_hash",
]

_SIGNAL_CACHE: Dict[int, Tuple[float, pd.DataFrame]] = {}
CACHE_TTL_SECONDS = 30


def _open_sheet_client():
    import gspread
    from google.oauth2.service_account import Credentials

    creds_val = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    if not creds_val:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    if creds_val.startswith("{"):
        logger.info("Using service account JSON from environment")
        try:
            creds_info = json.loads(creds_val)
        except json.JSONDecodeError:
            logger.error("Failed to parse Google Service Account JSON string")
            raise
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    else:
        logger.info(f"Using service account file path: {creds_val}")
        if not os.path.isfile(creds_val):
            raise FileNotFoundError(
                f"Credentials file not found at path: {creds_val}"
            )
        creds = Credentials.from_service_account_file(creds_val, scopes=scopes)

    return gspread.authorize(creds)


def validate_signal_row(row: Dict[str, Any]) -> bool:
    try:
        bp = float(row.get("buy_price", 0))
        sl = float(row.get("stop_loss", 0))
        tg = float(row.get("target", 0))
    except (TypeError, ValueError):
        return False

    if bp <= 0 or sl <= 0 or tg <= 0:
        return False

    side = str(row.get("signal", "")).upper().strip()
    if side == "BUY":
        return sl < bp < tg
    elif side == "SELL":
        return sl > bp > tg
    return False


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
def _fetch_sheet_rows(worksheet_index: int = 0) -> List[Dict[str, Any]]:
    client = _open_sheet_client()
    sheet_id = GOOGLE_SHEET_URL.split("/d/")[1].split("/")[0]
    sh = client.open_by_key(sheet_id)
    ws = sh.get_worksheet(worksheet_index)
    return ws.get_all_records()


def fetch_signals_from_sheet(
    worksheet_index: int = 0,
    use_cache: bool = True,
) -> pd.DataFrame:
    if use_cache and worksheet_index in _SIGNAL_CACHE:
        ts, df = _SIGNAL_CACHE[worksheet_index]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return df.copy()

    rows = _fetch_sheet_rows(worksheet_index)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    normalized: List[Dict[str, Any]] = []
    for raw in rows:
        item: Dict[str, Any] = {}
        for k, v in raw.items():
            item[k.strip().lower().replace(" ", "_")] = v

        item["timestamp"] = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")

        if not validate_signal_row(item):
            continue

        item["row_hash"] = hashlib.sha256(str(item).encode()).hexdigest()
        normalized.append(item)

    df = pd.DataFrame(normalized)
    _SIGNAL_CACHE[worksheet_index] = (time.time(), df)

    return df


def load_mock_signals_csv(path: Optional[str] = None) -> pd.DataFrame:
    p = path or str(MOCK_SIGNALS_PATH)
    if not os.path.isfile(p):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    df = pd.read_csv(p)
    
    df.columns = [
        col.strip().lower().replace(" ", "_")
        for col in df.columns
    ]
    
    required_cols = ["symbol", "signal", "buy_price", "stop_loss", "target"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            
    logger.info(f"Mock CSV loaded with columns: {df.columns.tolist()}")
            
    return df


def fetch_signals_safe(prefer_sheet: bool = True, use_cache: bool = True) -> pd.DataFrame:
    if prefer_sheet:
        logger.info("Attempting to fetch signals from Google Sheets")
        try:
            df = fetch_signals_from_sheet(use_cache=use_cache)
            logger.info(f"Fetched {len(df)} signals from sheet")
            return df
        except Exception as exc:
            logger.error("Google Sheets fetch FAILED - check credentials")
            logger.error(f"Full error: {exc}", exc_info=True)
            raise
    
    logger.warning("System is running on mock data")
    return load_mock_signals_csv()