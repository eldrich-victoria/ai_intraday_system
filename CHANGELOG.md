# Changelog

All notable changes to the AI Intraday Trading Tester System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-04-16

### Added
- **Database layer** (`src/db.py`): Centralised connection factory with WAL mode, foreign key enforcement, and thread-local connections.
- **Performance indexes** on `trades(status)`, `signals(status, ml_validated)`, `trades(exit_time)`.
- **Retry with exponential backoff** (`src/retry.py`): Reusable decorator for Google Sheets and Telegram API resilience.
- **Signal validation**: Reject signals where prices are non-positive or directional logic is incorrect (e.g., BUY with SL above entry).
- **Integer position sizing**: `math.floor()` ensures whole shares on NSE.
- **Round-trip brokerage**: ₹20 per leg (₹40 total) for realistic cost modelling.
- **Configurable slippage** (`SLIPPAGE_PCT`): Applies adverse slippage to entry and exit prices.
- **End-of-day forced closure** (`force_close_eod()`): Squares off all active positions near market close.
- **NSE holiday awareness**: Scheduler skips known holidays for 2025–2026.
- **File-based scheduler lock**: Prevents duplicate concurrent executions.
- **`RotatingFileHandler`**: Structured logging to `logs/scheduler.log` with rotation.
- **`--retrain` flag**: Manual model retraining via `python scheduler.py --retrain`.
- **ML Pipeline improvements**:
  - `StandardScaler` wrapped in `sklearn.Pipeline` for consistent scaling.
  - CV accuracy warning when mean accuracy falls below 55%.
- **Inline JSON support** for `GOOGLE_SERVICE_ACCOUNT_JSON` (GitHub Actions secrets).
- **Comprehensive test suite**: 12 test modules with shared fixtures (`conftest.py`).
- **Streamlit dashboard enhancements**:
  - Auto-refresh via `st_autorefresh`.
  - Sidebar filters for symbol, date range, and trade status.
  - INR formatting and improved chart styling.
- **GitHub Actions CI/CD**:
  - Lint with `ruff`.
  - Test with `pytest`.
  - Artifact caching for `trades.db` and `rf_model.pkl`.
- **Docker support**: Multi-stage `Dockerfile` with non-root user.
- **`.env.example`**: Documented template with all environment variables.
- **`CHANGELOG.md`**: This file.
- **`LICENSE`**: MIT License.

### Changed
- **Brokerage model**: Changed from ₹20 single-leg (exit only) to ₹40 round-trip (₹20 entry + ₹20 exit).
- **ML data handling**: Replaced `.bfill()` with `.dropna()` in `compute_features()` to prevent look-ahead bias / data leakage.
- **Synthetic data**: Replaced `np.random.seed()` with `np.random.default_rng()` to avoid polluting global RNG state.
- **Timezone handling**: Replaced all `pytz` usage with `zoneinfo.ZoneInfo` (stdlib in Python 3.9+).
- **GitHub Actions workflow**: Changed from `python scheduler.py` (infinite loop) to `python scheduler.py --once`; fixed env var name mismatches.
- **Dependencies**: Added upper-bound version constraints to all packages.
- **Dashboard**: Removed `sqlalchemy` dependency; uses `sqlite3` via `src.db`.

### Removed
- **`pytz`** dependency (replaced by `zoneinfo`).
- **`xgboost`** dependency (unused — RF is the default model).
- **`sqlalchemy`** dependency (replaced by direct `sqlite3` via `src.db`).
- **Hardcoded secrets** from `.env` and repository.

### Security
- Removed committed Telegram bot token and chat ID from `.env`.
- Updated `.gitignore` to exclude `*.json`, `*.db`, `*.pkl`, and log files.
- Added `GOOGLE_SERVICE_ACCOUNT_JSON` to `.gitignore` patterns.
- All secrets now read exclusively from environment variables.

### Fixed
- `USE_GOOGLE_SHEET=I` typo in `.env` (letter I instead of digit 1).
- GitHub Actions env var names mismatched: `TELEGRAM_BOT_TOKEN` → `BOT_TOKEN`, `TELEGRAM_CHAT_ID` → `CHAT_ID`, `GOOGLE_APPLICATION_CREDENTIALS` → `GOOGLE_SERVICE_ACCOUNT_JSON`.
- Duplicate `_conn()` function across `dummy_trader.py` and `performance.py`.
- Position sizing returned `float` instead of `int` — now uses `math.floor()`.
