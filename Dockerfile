FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask gunicorn

# Copy only dashboard files
COPY dashboard.py .
COPY templates/ templates/
COPY static/ static/

# Create empty directories the app expects
RUN mkdir -p backtest_results

EXPOSE ${PORT:-5001}

CMD gunicorn dashboard:app --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120
