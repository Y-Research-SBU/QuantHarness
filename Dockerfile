FROM python:3.11-slim

WORKDIR /app

# Install only dashboard dependencies
RUN pip install --no-cache-dir flask gunicorn

# Copy only what the dashboard needs
COPY dashboard.py .
COPY templates/ templates/
COPY static/ static/
COPY paper_trades.db .
COPY backtest_results/ backtest_results/
COPY market_config.py .

EXPOSE ${PORT:-5001}

CMD gunicorn dashboard:app --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120
