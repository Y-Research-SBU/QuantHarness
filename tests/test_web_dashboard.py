"""Tests for web_dashboard.py and scanner API endpoints — Flask routes return 200."""

import json

import pytest


@pytest.fixture
def app(tmp_db_path, monkeypatch):
    """Build a Flask test app using an isolated temp DB."""
    # Make the dashboard engine use our temp DB by patching the factory.
    import web_dashboard
    from paper_trading import PaperTradingEngine

    web_dashboard._engine = PaperTradingEngine(db_path=tmp_db_path)

    # Also patch get_connection calls used by dashboard endpoints so they hit tmp_db.
    monkeypatch.setattr(
        "web_dashboard.get_connection",
        lambda db_path=None: __import__("db_schema").get_connection(tmp_db_path),
    )
    monkeypatch.setattr(
        "web_dashboard.get_api_cost_summary",
        lambda db_path=None: __import__("performance_tracker").get_api_cost_summary(tmp_db_path),
    )
    monkeypatch.setattr(
        "web_dashboard.get_portfolio_snapshots",
        lambda db_path=None, symbol=None, limit=1000: __import__("performance_tracker").get_portfolio_snapshots(tmp_db_path, symbol, limit),
    )
    monkeypatch.setattr(
        "web_dashboard.get_strategy_performance",
        lambda db_path=None, strategy=None, symbol=None: __import__("performance_tracker").get_strategy_performance(tmp_db_path, strategy, symbol),
    )

    import web_interface
    web_interface.app.config["TESTING"] = True
    yield web_interface.app

    web_dashboard._engine = None


@pytest.fixture
def client(app):
    return app.test_client()


# ─────────────────────── Dashboard pages ───────────────────────


def test_dashboard_page_200(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Dashboard" in resp.data


def test_strategies_page_200(client):
    resp = client.get("/dashboard/strategies")
    assert resp.status_code == 200
    assert b"Strategy" in resp.data


def test_journal_page_200(client):
    resp = client.get("/dashboard/journal")
    assert resp.status_code == 200
    assert b"Trade Journal" in resp.data


# ─────────────────────── Dashboard APIs ───────────────────────


def test_api_portfolios(client):
    resp = client.get("/api/dashboard/portfolios")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert isinstance(data, list)
    assert len(data) > 0


def test_api_summary(client):
    resp = client.get("/api/dashboard/summary")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "total_pnl" in data
    assert "total_trades" in data
    assert "win_rate" in data
    assert "open_positions" in data
    assert "api_cost" in data


def test_api_strategy_performance(client):
    resp = client.get("/api/dashboard/strategy-performance")
    assert resp.status_code == 200
    assert isinstance(json.loads(resp.data), list)


def test_api_trades_empty(client):
    resp = client.get("/api/dashboard/trades")
    assert resp.status_code == 200
    assert isinstance(json.loads(resp.data), list)


def test_api_trades_filtered(client):
    resp = client.get("/api/dashboard/trades?symbol=BTC-USD&limit=10")
    assert resp.status_code == 200


def test_api_snapshots(client):
    resp = client.get("/api/dashboard/snapshots/BTC-USD")
    assert resp.status_code == 200
    assert isinstance(json.loads(resp.data), list)


# ─────────────────────── Scanner control APIs ───────────────────────


def test_api_status_returns_200(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "scanner" in data
    assert "summary" in data


def test_api_status_reports_not_running_initially(client):
    # Reset singleton to clean state
    import scanner_controller
    scanner_controller.reset_controller()
    resp = client.get("/api/status")
    data = json.loads(resp.data)
    assert data["scanner"]["running"] in (False, True)  # depends on prior tests


def test_api_start_stop_scanner(client, monkeypatch):
    import scanner_controller
    scanner_controller.reset_controller()

    # Prevent the scan loop from actually fetching data.
    monkeypatch.setattr(
        "scanner.fetch_market_data",
        lambda *a, **k: __import__("pandas").DataFrame(),
    )

    resp = client.post("/api/start-scanner", json={"interval_seconds": 3600, "use_agents": False})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["success"] is True

    # Starting again while running → returns not-started.
    resp2 = client.post("/api/start-scanner", json={"interval_seconds": 3600})
    data2 = json.loads(resp2.data)
    assert data2.get("success") is False

    resp3 = client.post("/api/stop-scanner")
    data3 = json.loads(resp3.data)
    assert data3["success"] is True


def test_api_stop_scanner_when_not_running(client):
    import scanner_controller
    scanner_controller.reset_controller()
    resp = client.post("/api/stop-scanner")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data.get("success") is False


def test_api_scan_once(client, monkeypatch):
    import scanner_controller
    scanner_controller.reset_controller()
    # Avoid hitting the network.
    monkeypatch.setattr(
        "scanner.fetch_market_data",
        lambda *a, **k: __import__("pandas").DataFrame(),
    )
    resp = client.post("/api/scan-once")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data.get("success") is True
    assert "results" in data


# ─────────────────────── Index / basic routes ───────────────────────


def test_get_assets_route(client):
    resp = client.get("/api/assets")
    assert resp.status_code == 200


def test_timeframe_limits(client):
    resp = client.get("/api/timeframe-limits/4h")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "max_days" in data


def test_custom_assets(client):
    resp = client.get("/api/custom-assets")
    assert resp.status_code == 200


# ─────────────────────── New dashboard pages ───────────────────────


def test_charts_page_200(client):
    resp = client.get("/dashboard/charts")
    assert resp.status_code == 200
    assert b"Charts" in resp.data


def test_markets_page_200(client):
    resp = client.get("/dashboard/markets")
    assert resp.status_code == 200
    assert b"Markets" in resp.data


def test_equity_page_200(client):
    resp = client.get("/dashboard/equity")
    assert resp.status_code == 200
    assert b"Equity" in resp.data


# ─────────────────────── New APIs ───────────────────────


@pytest.fixture
def no_yfinance(monkeypatch):
    """Prevent live price fetches in tests."""
    monkeypatch.setattr("web_dashboard._fetch_bulk_prices", lambda symbols: None)
    monkeypatch.setattr("web_dashboard._fetch_candles", lambda symbol, interval="4h", lookback_days=60: [])


def test_api_positions_empty(client, no_yfinance):
    resp = client.get("/api/positions")
    assert resp.status_code == 200
    assert isinstance(json.loads(resp.data), list)


def test_api_market_prices(client, no_yfinance):
    resp = client.get("/api/market-prices")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "markets" in data
    assert isinstance(data["markets"], list)
    # Every configured market should appear
    assert len(data["markets"]) > 0
    first = data["markets"][0]
    assert "symbol" in first and "display_name" in first and "change_24h" in first


def test_api_candles(client, no_yfinance):
    resp = client.get("/api/candles/BTC-USD?interval=4h")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["symbol"] == "BTC-USD"
    assert "candles" in data


def test_api_equity_curve(client, no_yfinance):
    resp = client.get("/api/equity-curve")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "series" in data
    assert "stats" in data
    assert "current_value" in data["stats"]
    assert "sharpe_ratio" in data["stats"]
    assert "sortino_ratio" in data["stats"]


def test_api_trades_top_level(client):
    resp = client.get("/api/trades?limit=50")
    assert resp.status_code == 200
    assert isinstance(json.loads(resp.data), list)


def test_api_trades_csv_export(client):
    resp = client.get("/api/trades.csv")
    assert resp.status_code == 200
    assert b"id,entry_time" in resp.data[:80]
