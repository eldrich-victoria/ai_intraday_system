# -*- coding: utf-8 -*-
"""Tests for src.db module — schema, WAL, indexes, transactions."""

import sqlite3


def test_ensure_database_creates_tables(initialized_db):
    """All required tables should exist after ensure_database."""
    from src.db import get_connection
    conn = get_connection()
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cur.fetchall()}
    assert "signals" in tables
    assert "trades" in tables
    assert "performance_metrics" in tables
    assert "state" in tables


def test_wal_mode_enabled(initialized_db):
    """Database should use WAL journal mode."""
    from src.db import get_connection
    conn = get_connection()
    cur = conn.execute("PRAGMA journal_mode")
    mode = cur.fetchone()[0].lower()
    assert mode == "wal"


def test_foreign_keys_enabled(initialized_db):
    """Foreign key enforcement should be ON."""
    from src.db import get_connection
    conn = get_connection()
    cur = conn.execute("PRAGMA foreign_keys")
    assert cur.fetchone()[0] == 1


def test_indexes_exist(initialized_db):
    """Performance indexes should be created."""
    from src.db import get_connection
    conn = get_connection()
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    )
    indexes = {row[0] for row in cur.fetchall()}
    assert "idx_trades_status" in indexes
    assert "idx_signals_status_ml" in indexes
    assert "idx_trades_exit_time" in indexes


def test_atomic_commit(initialized_db):
    """Atomic context manager should commit on success."""
    from src.db import atomic, get_connection
    with atomic() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO state(key, value) VALUES ('test_key', 'test_val')"
        )
    conn = get_connection()
    cur = conn.execute("SELECT value FROM state WHERE key='test_key'")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "test_val"


def test_atomic_rollback(initialized_db):
    """Atomic context manager should rollback on exception."""
    from src.db import atomic, get_connection
    try:
        with atomic() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO state(key, value) VALUES ('rollback_key', 'val')"
            )
            raise ValueError("deliberate error")
    except ValueError:
        pass
    conn = get_connection()
    cur = conn.execute("SELECT value FROM state WHERE key='rollback_key'")
    row = cur.fetchone()
    assert row is None


def test_virtual_capital_seeded(initialized_db):
    """Virtual capital should be seeded with INITIAL_CAPITAL."""
    from src.db import get_connection
    from src.config import INITIAL_CAPITAL
    conn = get_connection()
    cur = conn.execute("SELECT value FROM state WHERE key='virtual_capital'")
    row = cur.fetchone()
    assert row is not None
    assert float(row[0]) == float(INITIAL_CAPITAL)
