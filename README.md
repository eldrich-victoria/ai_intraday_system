# AI Intraday Trading Tester System

> **Paper-only** research harness for validating intraday trading signals on the NSE using machine learning, with automated execution via GitHub Actions.

[![CI](https://github.com/YOUR_USERNAME/ai_intraday_system/actions/workflows/trading.yml/badge.svg)](https://github.com/YOUR_USERNAME/ai_intraday_system/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Disclaimer:** This software does not place real orders, does not connect to any broker, and is **not** investment advice. It is an educational paper-trading simulation only. Validate any vendor "accuracy" claims independently. Follow SEBI norms for research and advertising. Past performance does not guarantee future results.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        scheduler.py                                 │
│  (--once for CI/cron  |  continuous loop for local/VM)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ fetch_sheet   │───▶│ dummy_trader  │───▶│ performance          │  │
│  │ (Google Sheet │    │ (paper sim   │    │ (metrics, equity,    │  │
│  │  or CSV)      │    │  + risk mgmt)│    │  Sharpe, drawdown)   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                        │              │
│  ┌──────▼───────┐    ┌──────▼───────┐    ┌──────────▼───────────┐  │
│  │ market_data   │    │ features_ml  │    │ alerts               │  │
│  │ (yfinance     │    │ (pandas-ta + │    │ (Telegram bot)       │  │
│  │  + cache)     │    │  RF Pipeline)│    └──────────────────────┘  │
│  └──────────────┘    └──────────────┘                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    db.py (SQLite + WAL)                       │   │
│  │  signals │ trades │ performance_metrics │ state               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               dashboard.py (Streamlit)                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**NSE Session:** Monday–Friday, 09:00–16:00 IST (with holiday awareness).

---

## Repository Layout

```
ai_intraday_system/
├── src/
│   ├── __init__.py
│   ├── config.py          # Centralised configuration
│   ├── db.py              # SQLite connection factory, schema, indexes
│   ├── retry.py           # Exponential backoff decorator
│   ├── fetch_sheet.py     # Google Sheets / CSV signal ingestion
│   ├── market_data.py     # yfinance price data with caching
│   ├── features_ml.py     # Technical indicators + RF Pipeline
│   ├── dummy_trader.py    # Paper trade simulation engine
│   ├── performance.py     # Metrics computation
│   ├── alerts.py          # Telegram notifications
│   └── dashboard.py       # Streamlit dashboard
├── tests/
│   ├── conftest.py        # Shared fixtures (DB isolation, mocks)
│   ├── test_config.py
│   ├── test_db.py
│   ├── test_fetch_sheet.py
│   ├── test_market_data.py
│   ├── test_features_ml.py
│   ├── test_dummy_trader.py
│   ├── test_performance.py
│   ├── test_alerts.py
│   ├── test_scheduler.py
│   └── test_integration.py
├── data/                  # trades.db (runtime), mock_signals.csv
├── models/                # rf_model.pkl (runtime)
├── logs/                  # scheduler.log (runtime)
├── .github/workflows/
│   └── trading.yml        # GitHub Actions CI/CD
├── scheduler.py           # Main orchestrator
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- Internet access for `yfinance` (and Google APIs if using Sheets)
- Optional: Google Cloud **service account** JSON for Sheets integration
- Optional: Telegram bot for alerts

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/ai_intraday_system.git
cd ai_intraday_system
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your values (Telegram bot, Google credentials, etc.)
```

### 3. Train the ML model

```bash
python -m src.features_ml
```

The scheduler also trains automatically if the model file is missing.

### 4. Run the scheduler

```bash
# Continuous loop (local / cloud VM):
python scheduler.py

# Single cycle (CI / cron):
python scheduler.py --once

# Force model retrain:
python scheduler.py --retrain
```

### 5. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BOT_TOKEN` | No | _(empty)_ | Telegram bot token from @BotFather |
| `CHAT_ID` | No | _(empty)_ | Telegram chat ID for alerts |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | No | _(empty)_ | Path to SA JSON file OR inline JSON string |
| `GOOGLE_SHEET_URL` | No | _(sample)_ | URL of the Google Sheet with signals |
| `USE_GOOGLE_SHEET` | No | `0` | `1` to use Sheets, `0` for mock CSV |
| `INITIAL_CAPITAL` | No | `100000` | Starting virtual capital (INR) |
| `RISK_PER_TRADE_PCT` | No | `0.01` | Risk per trade as fraction of capital |
| `BROKERAGE_PER_TRADE_INR` | No | `40` | Round-trip brokerage (₹20/leg) |
| `SLIPPAGE_PCT` | No | `0.0005` | Slippage as fraction of price (0.05%) |
| `ML_CONFIDENCE_THRESHOLD` | No | `0.7` | Min ML confidence for signal validation |
| `MAX_OPEN_POSITIONS` | No | `5` | Max concurrent open positions |
| `RF_N_ESTIMATORS` | No | `200` | Random Forest tree count |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## GitHub Actions Deployment

### Setting up secrets

In your GitHub repository, go to **Settings → Secrets and variables → Actions** and add:

| Secret Name | Value |
|------------|-------|
| `BOT_TOKEN` | Your Telegram bot token |
| `CHAT_ID` | Your Telegram chat ID |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The full JSON content of your service account key |

### How it works

The workflow (`.github/workflows/trading.yml`) runs every 5 minutes during NSE trading hours (03:30–10:30 UTC = 09:00–16:00 IST, weekdays only):

1. **Lint** — `ruff check .`
2. **Test** — `pytest tests/ -v`
3. **Run** — `python scheduler.py --once`

The SQLite database and ML model are **cached** between runs using GitHub Actions artifact caching.

### Manual trigger

You can also trigger the workflow manually via the GitHub Actions UI (workflow_dispatch).

---

## Google Sheets Integration

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable **Google Sheets API** + **Google Drive API**.
2. Create a **service account** and download the JSON key.
3. Set `GOOGLE_SERVICE_ACCOUNT_JSON` in `.env` (file path) or GitHub Secrets (inline JSON).
4. **Share your spreadsheet** with the service account email (Viewer is sufficient).
5. Expected columns: `Symbol`, `Signal` (BUY/SELL), `Buy Price`, `Stop Loss`, `Target`, `Timestamp`.

If the sheet is unreachable, the system **falls back** to `data/mock_signals.csv`.

---

## Telegram Alerts

1. Create a bot via [@BotFather](https://t.me/BotFather) → copy the token to `BOT_TOKEN`.
2. Send a message to your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to get `CHAT_ID`.
3. Alerts fire on: **entry**, **exit**, and **once-per-day summary**.

All Telegram calls include exponential backoff retry (3 attempts).

---

## Paper Trading Rules

| Parameter | Value | Description |
|-----------|-------|-------------|
| Brokerage | ₹40 round-trip | ₹20 entry + ₹20 exit |
| Risk | 1% of capital | Per trade vs. entry–stop distance |
| Position sizing | Integer shares | `math.floor()` for NSE compliance |
| Slippage | 0.05% | Configurable adverse price impact |
| Max positions | 5 | Concurrent open positions cap |
| EOD closure | Automatic | All positions squared off at 15:55 IST |

---

## Machine Learning

- **Features**: RSI(14), MACD(12,26,9), ATR(14), SMA/EMA(20), 1-day & 5-day returns, volume z-score.
- **Model**: RandomForest wrapped in `sklearn.Pipeline` with `StandardScaler`.
- **Validation**: `TimeSeriesSplit` cross-validation with accuracy logging.
- **Warning**: Logs alert if mean CV accuracy drops below 55%.
- **Gate**: Trades only created when `ml_confidence >= 0.7` (configurable).

---

## Testing

```bash
# Run all tests:
pytest tests/ -v

# Run with coverage:
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests use temporary SQLite databases (via `tmp_path`) and mock environment variables — no real API calls are made.

---

## Docker

```bash
# Build:
docker build -t ai-intraday-tester .

# Run single cycle:
docker run --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ai-intraday-tester

# Run continuous loop:
docker run --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ai-intraday-tester python scheduler.py

# Run dashboard:
docker run --env-file .env \
  -v $(pwd)/data:/app/data \
  -p 8501:8501 \
  ai-intraday-tester streamlit run src/dashboard.py --server.port 8501
```

---

## Security Best Practices

- **Never** commit `.env`, service account JSON, or real tokens to version control.
- Use `.env.example` as a template; keep `.env` local only.
- For CI/CD, use GitHub Actions **secrets** (encrypted at rest).
- The `.gitignore` excludes `*.json`, `*.db`, `*.pkl`, `.env`, and logs.
- Rotate any credentials that were previously exposed.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pandas_ta` | `pip install pandas-ta>=0.3.14b` |
| `FileNotFoundError: GOOGLE_SERVICE_ACCOUNT_JSON` | Set the env var to a valid file path or inline JSON |
| `Market is closed` on `--once` | Expected — the scheduler only runs during NSE hours |
| `Another scheduler instance is running` | Delete `scheduler.lock` and retry |
| `TimeSeriesSplit` errors | Insufficient data — ensure yfinance returns data or use mock CSV |
| Dashboard doesn't refresh | Install `streamlit-autorefresh`: `pip install streamlit-autorefresh` |
| DB locked errors | Enable WAL mode (default in v2.0.0) or reduce concurrent access |

---

## Success Criteria (Reporting)

The dashboard and `performance_metrics` table track whether **all** of these hold simultaneously:

- Win rate **> 60%** (after costs)
- Profit factor **> 1.5**
- Max drawdown **< 10%**

Displayed as `success_gate`: PASS or FAIL.

---

## License

[MIT](LICENSE) — use and modify freely for research. Verify compliance with data vendors (Google, Yahoo) and Indian regulations before any production or client-facing use.
