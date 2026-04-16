# -*- coding: utf-8 -*-
"""Fetch live trading signals from Google Sheets using gspread (service account)."""

import asyncio
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


def _normalize_column_name(name: str) -> str:
    if name is None:
        return ""
    key = str(name).strip().lower().replace("_", " ")
    mapping = {
        "symbol": "symbol",
        "signal": "signal",
        "buy price": "buy_price",
        "buyprice": "buy_price",
        "entry": "buy_price",
        "entry price": "buy_price",
        "stop loss": "stop_loss",
        "stoploss": "stop_loss",
        "sl": "stop_loss",
        "target": "target",
        "tgt": "target",
        "timestamp": "timestamp",
        "time": "timestamp",
        "date": "timestamp",
    }
    return mapping.get(key, key.replace(" ", "_"))


def _row_hash(row: Dict[str, Any]) -> str:
    parts = [
        str(row.get("symbol", "")),
        str(row.get("signal", "")),
        str(row.get("buy_price", "")),
        str(row.get("stop_loss", "")),
        str(row.get("target", "")),
        str(row.get("timestamp", "")),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _open_sheet_client():
    import gspread
    from google.oauth2.service_account import Credentials

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise FileNotFoundError(
            "GOOGLE_SERVICE_ACCOUNT_JSON not set or empty"
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    if os.path.isfile(raw):
        creds = Credentials.from_service_account_file(raw, scopes=scopes)
    else:
        try:
            info = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raise FileNotFoundError(
                f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {raw[:60]}..."
            )
        creds = Credentials.from_service_account_info(info, scopes=scopes)

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
            nk = _normalize_column_name(k)
            if nk:
                item[nk] = v
        for req in ("symbol", "signal", "buy_price", "stop_loss", "target"):
            if req not in item:
                item[req] = None
        if "timestamp" not in item or item["timestamp"] in (None, ""):
            item["timestamp"] = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
        
        if not validate_signal_row(item):
            logger.warning(
                f"Skipping invalid signal row: { {k: item.get(k) for k in ('symbol', 'signal', 'buy_price', 'stop_loss', 'target')} }"
            )
            continue
        item["row_hash"] = _row_hash(item)
        normalized.append(item)

    if not normalized:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(normalized)
    for col in ("buy_price", "stop_loss", "target"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df_out = df[[c for c in OUTPUT_COLUMNS if c in df.columns]].copy()
    
    _SIGNAL_CACHE[worksheet_index] = (time.time(), df_out)
    return df_out


async def fetch_signals_async(worksheet_index: int = 0, use_cache: bool = True) -> pd.DataFrame:
    """Optional asynchronous fetching wrapper."""
    return await asyncio.to_thread(fetch_signals_from_sheet, worksheet_index, use_cache)


def load_mock_signals_csv(path: Optional[str] = None) -> pd.DataFrame:
    p = path or str(MOCK_SIGNALS_PATH)
    if not os.path.isfile(p):
        logger.warning(f"Mock signals file not found: {p}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.read_csv(p)
    ren = {}
    for c in df.columns:
        ren[c] = _normalize_column_name(c)
    df = df.rename(columns=ren)
    for col in ("buy_price", "stop_loss", "target"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" not in df.columns:
        df["timestamp"] = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["row_hash"] = df.apply(lambda r: _row_hash(r.to_dict()), axis=1)
    return df[[c for c in OUTPUT_COLUMNS if c in df.columns]]


def fetch_signals_safe(prefer_sheet: bool = True, use_cache: bool = True) -> pd.DataFrame:
    if prefer_sheet:
        try:
            df = fetch_signals_from_sheet(use_cache=use_cache)
            logger.info(f"Fetched {len(df)} rows from Google Sheet")
            return df
        except Exception as exc:
            logger.error(
                f"Sheet fetch failed ({exc}). Falling back to mock CSV.",
                exc_info=True,
            )
    return load_mock_signals_csv()
