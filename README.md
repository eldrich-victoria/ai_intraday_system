# AI Intraday Trading Tester System

## Overview

This project is a **paper-only** pipeline that ingests intraday-style signals (for example from a Google Sheet of NSE names), enriches them with technical features, scores them with a **Random Forest** classifier trained under **TimeSeriesSplit**, simulates **dummy trades** with explicit costs and position sizing, persists results in **SQLite**, and exposes **Streamlit** analytics plus optional **Telegram** alerts. It is designed for **local** use or **Google Colab** using free, open-source components.

**Compliance:** This software does not place orders, does not connect to brokers, and is **not** investment advice. Validate any vendor “accuracy” claims independently. Follow SEBI norms for research and advertising; use this only as an internal research harness.

## Architecture (text)

```
Google Sheet (optional, gspread)
        → fetch / fallback mock CSV
        → SQLite (signals, trades, performance_metrics)
        → yfinance NSE prices (.NS) + short TTL cache
        → pandas-ta features → Random Forest probability
        → Paper simulator (pending → active → closed)
        → Metrics (win rate, PF, Sharpe, drawdown, …)
        → Streamlit dashboard + Telegram (optional)
        → scheduler.py loop (~60s) during NSE cash session (IST)
```

NSE regular session assumed: **Monday–Friday, 09:00–16:00 IST** (scheduler only runs the heavy pipeline inside this window).

## Repository layout

```
ai_intraday_trading_system/
├── src/
│   ├── config.py
│   ├── fetch_sheet.py
│   ├── market_data.py
│   ├── features_ml.py
│   ├── dummy_trader.py
│   ├── performance.py
│   ├── alerts.py
│   └── dashboard.py
├── data/              # trades.db created at runtime
├── models/            # rf_model.pkl after training
├── scheduler.py
├── requirements.txt
├── README.md
├── .env
└── data/mock_signals.csv   # offline fallback
```

## Prerequisites

- Python **3.10+**
- Internet access for `yfinance` (and Google APIs if using Sheets)
- Optional: Google Cloud **service account** JSON for Sheets

## Setup

1. Create and activate a virtual environment (recommended).

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env` and fill secrets (or edit the provided file in place for local use only):

   - `BOT_TOKEN`, `CHAT_ID` — from Telegram BotFather (optional).
   - `GOOGLE_SERVICE_ACCOUNT_JSON` — absolute path to service account JSON (optional).
   - `USE_GOOGLE_SHEET=0` forces mock CSV + yfinance only (good for first run).

4. **Train / refresh the Random Forest** (creates `models/rf_model.pkl`):

   ```bash
   python -m src.features_ml
   ```

   The scheduler also trains automatically if the model file is missing.

## Google Sheets API (service account)

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project, enable **Google Sheets API** and **Google Drive API**.
2. Create a **service account**, download the JSON key.
3. Set `GOOGLE_SERVICE_ACCOUNT_JSON` in `.env` to the full path of that file.
4. **Share your spreadsheet** with the service account email (Viewer is enough) — same sheet as in `GOOGLE_SHEET_URL`.
5. Expected columns (headers can vary slightly; mapping is normalized in code):

   - Symbol, Signal (BUY/SELL), Buy Price, Stop Loss, Target, Timestamp

If the sheet is unreachable, the system **falls back** to `data/mock_signals.csv` so the pipeline keeps running.

## Run the scheduler

**Continuous loop** (wakes every ~60s; runs pipeline only during NSE hours):

```bash
python scheduler.py
```

**Single cycle** (useful for tests):

```bash
python scheduler.py --once
```

### Cron (Linux / macOS)

Run every minute during sessions (example runs only if your wrapper checks time, or rely on the internal sleep):

```cron
* 9-15 * * 1-5 cd /path/to/ai_intraday_trading_system && . .venv/bin/activate && python scheduler.py --once
```

Adjust hours if you use a long-lived `python scheduler.py` process instead.

### Windows Task Scheduler

1. Create Task → **Triggers**: Daily, repeat every **1 minute** for **7 hours**, limited to weekdays (or one trigger per weekday).
2. **Action**: Start a program  
   - Program: `C:\path\to\.venv\Scripts\python.exe`  
   - Arguments: `scheduler.py --once`  
   - Start in: `d:\ai_intraday_trading_system`
3. Alternatively start `python scheduler.py` once at 09:00 IST and let it loop (align machine timezone or run on a server in IST).

## Streamlit dashboard

From the project root:

```bash
streamlit run src/dashboard.py
```

Open the printed local URL. Use **Refresh data now** or wait for the ~30s cache TTL. Charts use **matplotlib** (equity and drawdown).

### Expose with ngrok (Colab or remote)

1. Install [ngrok](https://ngrok.com/) and authenticate.
2. Run Streamlit on port 8501 (default).
3. In another terminal: `ngrok http 8501` and use the HTTPS URL shown.

**Colab sketch:**

```python
!pip install -r requirements.txt
# upload project or clone repo, then:
get_ipython().system_raw("streamlit run src/dashboard.py &")
get_ipython().system_raw("ngrok http 8501")
```

Use Colab’s ngrok/pyngrok snippets if you prefer a Python tunnel.

## Telegram bot

1. Talk to [@BotFather](https://t.me/BotFather), create a bot, copy the token → `BOT_TOKEN`.
2. Send a message to your bot, then visit  
   `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`  
   and read `message.chat.id` → `CHAT_ID`.
3. Set both in `.env`. Alerts fire on simulated **entry**, **exit**, and **once-per-day** summary (first cycle that day while the scheduler runs).

## Machine learning validation

- Features: RSI(14), MACD(12,26,9), ATR(14), SMA/EMA(20), short returns, volume z-score.
- **Random Forest** with **TimeSeriesSplit** cross-validation (logged), final fit on full training matrix.
- **Execution gate:** dummy trades are only created when `ml_confidence >= 0.7` (configurable via `ML_CONFIDENCE_THRESHOLD` in environment if you extend `config.py`).

Retrain manually:

```bash
python -m src.features_ml
```

`xgboost` is listed for optional extensions; the default path is **Random Forest** only.

## Paper trading rules (implemented)

- **Costs:** ₹**20** brokerage per **closed** round-trip (subtracted from net P&amp;L once on exit).
- **Risk:** ~**1%** of current virtual capital per trade vs. entry–stop distance (see `_position_size` in `dummy_trader.py`), capped by `MAX_POSITION_PCT_CAPITAL`.
- **Lifecycle:** Pending (wait for fill) → Active → Closed at target or stop (single-price proxy using last quote when intraday OHLC is unavailable).

## Success criteria (reporting)

The dashboard and `performance_metrics` table record whether:

- Win rate **> 60%** (after costs),
- Profit factor **> 1.5**,
- Max drawdown **< 10%**,

all hold simultaneously (`success_gate`).

## Testing

```bash
pip install pytest
pytest tests/test_performance.py -q
```

Mock signals: edit `data/mock_signals.csv`. Set `USE_GOOGLE_SHEET=0` to avoid Google APIs.

## GitHub practices

- Do **not** commit live `BOT_TOKEN`, service-account JSON, or private sheets.
- Keep `.env` out of public repos (use `.env.example` with blanks for sharing).
- Track `requirements.txt` and lock major versions in production if needed.

## Example `.env`

```env
BOT_TOKEN=
CHAT_ID=
GOOGLE_SERVICE_ACCOUNT_JSON=D:\secrets\service_account.json
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/1mw5tL2q98s-TcmLfb2U_uFKKQZA33WFd-GynZA0_XaU/edit
USE_GOOGLE_SHEET=1
INITIAL_CAPITAL=100000
ML_CONFIDENCE_THRESHOLD=0.7
LOG_LEVEL=INFO
```

## Future enhancements

- Intrabar OHLC from a free tick/minute provider for cleaner fill/exit realism.
- Walk-forward backtests separate from live paper loop.
- Optional **XGBoost** ensemble and calibrated probabilities.
- Postgres instead of SQLite for multi-user dashboards.
- Stricter SEBI-aligned disclosures in the UI for any public demo.

## License

Use and modify freely for research; verify compliance with data vendors (Google, Yahoo) and Indian regulations before any production or client-facing use.
