# ==============================================================================
# AI Intraday Trading Tester System — Dockerfile
# ==============================================================================
# Multi-stage build for minimal image size.
# Usage:
#   docker build -t ai-intraday-tester .
#   docker run --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ai-intraday-tester
# ==============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Production stage
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from base stage.
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Create non-root user (security best practice).
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy application code.
COPY src/ ./src/
COPY scheduler.py .
COPY data/mock_signals.csv ./data/

# Create runtime directories.
RUN mkdir -p data models logs && \
    chown -R appuser:appuser /app

USER appuser

# Default command: run single cycle (for CI/cron).
# Override with `python scheduler.py` for continuous loop.
CMD ["python", "scheduler.py", "--once"]

# Health check (optional, for Docker Compose / orchestrators).
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from src.db import ensure_database; ensure_database(); print('ok')" || exit 1

# Expose Streamlit port (if running dashboard).
EXPOSE 8501
