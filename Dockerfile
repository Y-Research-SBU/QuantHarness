FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    flask \
    flask-socketio \
    python-socketio \
    websocket-client \
    eventlet \
    gunicorn \
    yfinance \
    pandas \
    numpy

# Copy only dashboard files
COPY dashboard.py .
COPY price_feed.py .
COPY market_config.py .
COPY db_schema.py .
COPY templates/ templates/
COPY static/ static/

# Create empty directories the app expects
RUN mkdir -p backtest_results

# Railway sets PORT dynamically; default to 5001 for local dev
ENV PORT=5001
# Pick an async mode compatible with gunicorn's eventlet worker. The
# dashboard module honors SOCKETIO_ASYNC_MODE when building the SocketIO
# server, so they stay aligned.
ENV SOCKETIO_ASYNC_MODE=eventlet

EXPOSE 5001

# flask-socketio requires a single eventlet (or gevent) worker so the
# WebSocket transport works — the default sync worker cannot hold long-lived
# connections. Workers must stay at 1: socket.io sessions are sticky to the
# worker that accepted the handshake.
CMD ["sh", "-c", "echo Starting on port $PORT && exec gunicorn dashboard:app --bind 0.0.0.0:$PORT --worker-class eventlet --workers 1 --timeout 120 --log-level info --access-logfile - --error-logfile -"]
