FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask gunicorn

# Copy only dashboard files
COPY dashboard.py .
COPY templates/ templates/
COPY static/ static/

# Create empty directories the app expects
RUN mkdir -p backtest_results

# Railway sets PORT dynamically; default to 5001 for local dev
ENV PORT=5001

EXPOSE 5001

CMD ["sh", "-c", "echo Starting on port $PORT && exec gunicorn dashboard:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --log-level info --access-logfile - --error-logfile -"]
