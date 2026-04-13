# -*- coding: utf-8 -*-
"""Fetch live trading signals from Google Sheets using gspread (service account)."""

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import GOOGLE_SHEET_URL, IST, MOCK_SIGNALS_PATH

logger = logging.getLogger(__name__)

# Normalized column names we emit
OUTPUT_COLUMNS = [
    "symbol",
    "signal",
    "buy_price",
    "stop_loss",
    "target",
    "timestamp",
    "row_hash",
]


def _normalize_column_name(name: str) -> str:
    """Map sheet header variants to snake_case keys."""
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
    """Stable hash for deduplication."""
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
    """Build gspread client from service account JSON path."""
    import gspread
    from google.oauth2.service_account import Credentials

    json_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not json_path or not os.path.isfile(json_path):
        raise FileNotFoundError(
            "GOOGLE_SERVICE_ACCOUNT_JSON not set or file missing: {}".format(json_path)
        )
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(json_path, scopes=scopes)
    return gspread.authorize(creds)


def fetch_signals_from_sheet(
    worksheet_index: int = 0,
) -> pd.DataFrame:
    """
    Pull all rows from the configured Google Sheet.

    Returns a DataFrame with columns: symbol, signal, buy_price, stop_loss,
    target, timestamp, row_hash.
    """
    client = _open_sheet_client()
    sheet_id = GOOGLE_SHEET_URL.split("/d/")[1].split("/")[0]
    sh = client.open_by_key(sheet_id)
    ws = sh.get_worksheet(worksheet_index)
    rows = ws.get_all_records()
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
        item["row_hash"] = _row_hash(item)
        normalized.append(item)

    df = pd.DataFrame(normalized)
    for col in ("buy_price", "stop_loss", "target"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return df[[c for c in OUTPUT_COLUMNS if c in df.columns]]


def load_mock_signals_csv(path: Optional[str] = None) -> pd.DataFrame:
    """Load signals from local CSV for offline / fallback testing."""
    p = path or str(MOCK_SIGNALS_PATH)
    if not os.path.isfile(p):
        logger.warning("Mock signals file not found: {}".format(p))
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


def fetch_signals_safe(prefer_sheet: bool = True) -> pd.DataFrame:
    """
    Try Google Sheet first; on any failure load mock CSV so the pipeline keeps running.
    """
    if prefer_sheet:
        try:
            df = fetch_signals_from_sheet()
            logger.info("Fetched {} rows from Google Sheet".format(len(df)))
            return df
        except Exception as exc:
            logger.error(
                "Sheet fetch failed ({}). Falling back to mock CSV.".format(exc),
                exc_info=True,
            )
    return load_mock_signals_csv()
