"""
Tests for the Hyperliquid integration.

The executor talks to ``Exchange`` and ``Info`` clients. Every test here
injects fake clients via the ``clients_factory`` hook so we never hit the
network. That also lets us assert on the exact calls the executor makes
(order type, reduce_only flags, SL/TP triggers, cancellations, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import hyperliquid_config as hc
from hyperliquid_executor import (
    HyperliquidExecutor,
    HyperliquidExecutorError,
    _Clients,
)
from market_config import StrategyType
from position_sizing import PositionSizeResult
from strategies import Signal


# ─────────────────────── Fake clients ───────────────────────


class FakeExchange:
    """In-memory stand-in for ``hyperliquid.exchange.Exchange``.

    Records every call and returns Hyperliquid-shaped responses so the
    executor's response-parsing code runs for real.
    """

    def __init__(self) -> None:
        self.orders: List[Dict[str, Any]] = []
        self.cancels: List[Dict[str, Any]] = []
        self.leverage_calls: List[Dict[str, Any]] = []
        self.market_closes: List[Dict[str, Any]] = []
        self._next_oid = 1000
        # Control-plane hooks for tests:
        self.next_entry_avg_px: Optional[float] = None
        self.close_avg_px: float = 100.0
        self.fail_entry: bool = False

    def _alloc_oid(self) -> int:
        self._next_oid += 1
        return self._next_oid

    def update_leverage(self, leverage: int, name: str, is_cross: bool = True) -> Dict[str, Any]:
        self.leverage_calls.append({"leverage": leverage, "name": name, "is_cross": is_cross})
        return {"status": "ok"}

    def order(
        self,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: Dict[str, Any],
        reduce_only: bool = False,
        cloid: Any = None,
        builder: Any = None,
    ) -> Dict[str, Any]:
        oid = self._alloc_oid()
        self.orders.append({
            "name": name, "is_buy": is_buy, "sz": sz, "limit_px": limit_px,
            "order_type": order_type, "reduce_only": reduce_only, "oid": oid,
        })
        if self.fail_entry and not reduce_only:
            return {"status": "err", "response": "simulated rejection"}
        # Trigger orders rest (they activate on price) — return "resting"
        # Entry limit orders rest until filled.
        return {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {"statuses": [{"resting": {"oid": oid}}]},
            },
        }

    def market_open(
        self, name: str, is_buy: bool, sz: float, px: Optional[float] = None,
        slippage: float = 0.05, cloid: Any = None, builder: Any = None,
    ) -> Dict[str, Any]:
        oid = self._alloc_oid()
        self.orders.append({
            "name": name, "is_buy": is_buy, "sz": sz, "market": True, "slippage": slippage, "oid": oid,
        })
        return {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {"statuses": [{"filled": {"oid": oid, "avgPx": self.next_entry_avg_px or px or 100.0, "totalSz": sz}}]},
            },
        }

    def market_close(self, coin: str, sz: Optional[float] = None, px: Optional[float] = None,
                     slippage: float = 0.05, cloid: Any = None, builder: Any = None) -> Dict[str, Any]:
        self.market_closes.append({"coin": coin, "sz": sz, "slippage": slippage})
        oid = self._alloc_oid()
        return {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {"statuses": [{"filled": {"oid": oid, "avgPx": self.close_avg_px, "totalSz": sz or 0}}]},
            },
        }

    def cancel(self, name: str, oid: int) -> Dict[str, Any]:
        self.cancels.append({"name": name, "oid": oid})
        return {"status": "ok"}


class FakeInfo:
    """Stand-in for ``hyperliquid.info.Info``. State configurable per-test."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "marginSummary": {
                "accountValue": "5000.0",
                "totalMarginUsed": "0.0",
                "totalNtlPos": "0.0",
                "totalRawUsd": "5000.0",
            },
            "crossMarginSummary": {"accountValue": "5000.0"},
            "withdrawable": "5000.0",
            "assetPositions": [],
        }
        self.fills: List[Dict[str, Any]] = []
        self.open_orders_list: List[Dict[str, Any]] = []

    def user_state(self, address: str) -> Dict[str, Any]:
        return self.state

    def user_fills(self, address: str) -> List[Dict[str, Any]]:
        return list(self.fills)

    def open_orders(self, address: str, dex: str = "") -> List[Dict[str, Any]]:
        return list(self.open_orders_list)


def _fake_factory_pair():
    """Return (factory, exchange, info) — factory is the clients_factory hook."""
    exch = FakeExchange()
    info = FakeInfo()

    def factory(private_key: str, testnet: bool) -> _Clients:
        # Capture for test assertions
        factory.last_private_key = private_key     # type: ignore[attr-defined]
        factory.last_testnet = testnet              # type: ignore[attr-defined]
        return _Clients(exchange=exch, info=info, address="0xTESTADDRESS")

    return factory, exch, info


# ─────────────────────── Fixtures ───────────────────────


@pytest.fixture
def make_executor(tmp_db_path):
    """Factory: create a HyperliquidExecutor with mocked clients."""
    def _make(**overrides) -> tuple[HyperliquidExecutor, FakeExchange, FakeInfo]:
        factory, exch, info = _fake_factory_pair()
        kwargs = dict(
            private_key="0x" + "aa" * 32,
            testnet=True,
            db_path=tmp_db_path,
            clients_factory=factory,
        )
        kwargs.update(overrides)
        ex = HyperliquidExecutor(**kwargs)
        return ex, exch, info
    return _make


# ─────────────────────── hyperliquid_config ───────────────────────


def test_symbol_map_has_core_cryptos():
    assert hc.SYMBOL_MAP["BTC-USD"] == "BTC"
    assert hc.SYMBOL_MAP["ETH-USD"] == "ETH"
    assert hc.SYMBOL_MAP["SOL-USD"] == "SOL"


def test_to_hl_symbol_roundtrip():
    for internal, hl in hc.SYMBOL_MAP.items():
        assert hc.to_hl_symbol(internal) == hl
        assert hc.to_internal_symbol(hl) == internal


def test_to_hl_symbol_returns_none_for_unknown():
    assert hc.to_hl_symbol("SPY") is None
    assert hc.to_internal_symbol("DOGE") is None


def test_leverage_for_defaults_when_unset():
    assert hc.leverage_for("BTC-USD") == hc.DEFAULT_LEVERAGE


def test_leverage_for_honours_per_symbol_override(monkeypatch):
    monkeypatch.setitem(hc.LEVERAGE_BY_SYMBOL, "BTC-USD", 10)
    assert hc.leverage_for("BTC-USD") == 10


# ─────────────────────── Initialization ───────────────────────


def test_missing_private_key_raises(tmp_db_path, monkeypatch):
    monkeypatch.delenv(hc.PRIVATE_KEY_ENV_VAR, raising=False)
    factory, _, _ = _fake_factory_pair()
    with pytest.raises(HyperliquidExecutorError):
        HyperliquidExecutor(
            private_key=None, testnet=True, db_path=tmp_db_path, clients_factory=factory,
        )


def test_env_var_private_key_is_used(tmp_db_path, monkeypatch):
    monkeypatch.setenv(hc.PRIVATE_KEY_ENV_VAR, "0x" + "bb" * 32)
    factory, _, _ = _fake_factory_pair()
    HyperliquidExecutor(testnet=True, db_path=tmp_db_path, clients_factory=factory)
    assert factory.last_private_key == "0x" + "bb" * 32     # type: ignore[attr-defined]


def test_defaults_to_testnet(make_executor):
    ex, _, _ = make_executor()
    assert ex.testnet is True


def test_mainnet_requires_explicit_testnet_false(make_executor):
    ex, _, _ = make_executor(testnet=False)
    assert ex.testnet is False


def test_base_url_selection(tmp_db_path):
    """Real _build_clients maps testnet flag to the right constant."""
    factory_calls: List[bool] = []

    # Stub out Exchange/Info so we don't hit the network, but verify url selection.
    from hyperliquid_executor import _build_clients
    from hyperliquid.utils import constants
    from unittest.mock import patch

    with patch("hyperliquid_executor.Account") if False else patch("eth_account.Account") as acc, \
         patch("hyperliquid.info.Info") as Info, \
         patch("hyperliquid.exchange.Exchange") as Ex:
        acc.from_key.return_value = MagicMock(address="0xABC")
        Info.return_value = MagicMock()
        Ex.return_value = MagicMock()
        _build_clients("0x" + "cc" * 32, testnet=True)
        assert Info.call_args.kwargs["base_url"] == constants.TESTNET_API_URL
        _build_clients("0x" + "cc" * 32, testnet=False)
        assert Info.call_args.kwargs["base_url"] == constants.MAINNET_API_URL


def test_ensures_portfolios_for_all_markets(make_executor):
    from market_config import MARKETS
    ex, _, _ = make_executor()
    pfs = ex.get_all_portfolios()
    assert len(pfs) == len(MARKETS)


# ─────────────────────── Helpers ───────────────────────


def _signal(symbol="BTC-USD", direction="LONG", entry=100.0, stop=95.0, tp=115.0) -> Signal:
    return Signal(
        direction=direction, strength=0.8, strategy=StrategyType.MOMENTUM,
        symbol=symbol, timeframe="1h", entry_price=entry,
        stop_loss=stop, take_profit=tp, risk_reward_ratio=2.0,
        reasoning="test", metadata={},
    )


def _position(size=500.0, qty=5.0, stop=95.0, tp=115.0) -> PositionSizeResult:
    return PositionSizeResult(
        position_size_usd=size, quantity=qty, risk_per_trade_usd=size * 0.02,
        risk_pct=0.02, kelly_fraction=0.2, half_kelly=0.1,
        stop_loss=stop, take_profit=tp, reason="test",
    )


# ─────────────────────── execute_trade ───────────────────────


def test_execute_trade_places_entry_with_limit_order_type(make_executor):
    ex, exch, _ = make_executor()
    oid = ex.execute_trade(_signal(), _position())
    assert oid is not None
    # The first non-trigger order recorded is the entry.
    entries = [o for o in exch.orders if not o.get("reduce_only")]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "BTC"
    assert entry["is_buy"] is True
    assert entry["sz"] == 5.0
    assert entry["limit_px"] == 100.0
    assert entry["order_type"] == {"limit": {"tif": hc.LIMIT_TIF}}


def test_execute_trade_short_sends_sell(make_executor):
    ex, exch, _ = make_executor()
    ex.execute_trade(_signal(direction="SHORT", stop=105.0, tp=85.0), _position(stop=105.0, tp=85.0))
    entries = [o for o in exch.orders if not o.get("reduce_only")]
    assert entries[0]["is_buy"] is False


def test_execute_trade_places_sl_and_tp_triggers(make_executor):
    ex, exch, _ = make_executor()
    ex.execute_trade(_signal(), _position())
    triggers = [o for o in exch.orders if o.get("reduce_only")]
    # One SL + one TP
    assert len(triggers) == 2
    tpsls = {o["order_type"]["trigger"]["tpsl"] for o in triggers}
    assert tpsls == {"sl", "tp"}
    # Both triggers should be on the opposite side of a LONG entry — sell
    for t in triggers:
        assert t["is_buy"] is False


def test_execute_trade_sets_leverage_once_per_symbol(make_executor):
    ex, exch, info = make_executor()
    ex.execute_trade(_signal(), _position())
    # After the first entry, trying a second entry on the same symbol is
    # blocked by the "already have a position" check, so clear that:
    assert len(exch.leverage_calls) == 1

    # Add a second symbol — its leverage gets set the first time.
    ex.execute_trade(_signal(symbol="ETH-USD"), _position())
    assert len(exch.leverage_calls) == 2


def test_execute_trade_unsupported_symbol_returns_none(make_executor):
    ex, exch, _ = make_executor()
    sig = _signal(symbol="SPY")
    assert ex.execute_trade(sig, _position()) is None
    assert not exch.orders


def test_execute_trade_records_row_in_trades_table(make_executor, tmp_db_path):
    ex, _, _ = make_executor()
    oid = ex.execute_trade(_signal(), _position())
    assert oid is not None
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        rows = conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["symbol"] == "BTC-USD"
    assert row["direction"] == "LONG"
    assert row["status"] == "OPEN"
    # Metadata carries the Hyperliquid order id.
    import json
    meta = json.loads(row["decision_json"])
    assert meta["exchange"] == "hyperliquid"
    assert meta["network"] == "testnet"
    assert meta["hl_entry_oid"] is not None


def test_execute_trade_blocks_when_position_already_open(make_executor):
    ex, exch, info = make_executor()
    # Fake an existing on-exchange position for BTC.
    info.state["assetPositions"] = [{
        "position": {
            "coin": "BTC", "szi": "1.0", "entryPx": "100.0",
            "positionValue": "100.0", "unrealizedPnl": "0.0",
        }
    }]
    oid = ex.execute_trade(_signal(), _position())
    assert oid is None
    assert not exch.orders


# ─────────────────────── Safety limits ───────────────────────


def test_mainnet_rejects_oversized_position(make_executor):
    ex, exch, _ = make_executor(testnet=False, max_position_usd=50.0)
    oid = ex.execute_trade(_signal(), _position(size=100.0))
    assert oid is None
    assert not exch.orders


def test_testnet_allows_oversized_position(make_executor):
    # Safety cap exists but testnet is advisory.
    ex, exch, _ = make_executor(testnet=True, max_position_usd=50.0)
    oid = ex.execute_trade(_signal(), _position(size=100.0))
    assert oid is not None
    assert exch.orders


def test_mainnet_daily_loss_kill_switch(make_executor, tmp_db_path):
    ex, exch, _ = make_executor(testnet=False, max_daily_loss_usd=20.0)
    # Simulate realized daily loss beyond the cap by updating portfolios directly.
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        conn.execute("UPDATE portfolios SET daily_pnl = -25.0 WHERE symbol = 'BTC-USD'")
    oid = ex.execute_trade(_signal(), _position(size=10.0))
    assert oid is None
    assert not exch.orders


def test_zero_size_refused(make_executor):
    ex, exch, _ = make_executor()
    oid = ex.execute_trade(_signal(), _position(size=0.0, qty=0.0))
    assert oid is None
    assert not exch.orders


# ─────────────────────── close_trade / check_stops ───────────────────────


def test_close_trade_calls_market_close_and_finalizes(make_executor, tmp_db_path):
    ex, exch, info = make_executor()
    # Need to tell our open_positions check that the position exists first.
    info.state["assetPositions"] = []  # allow entry
    trade_id_str = ex.execute_trade(_signal(), _position())
    assert trade_id_str
    # Look up the SQLite row id.
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        row = conn.execute("SELECT id FROM trades WHERE status = 'OPEN'").fetchone()
    trade_id = int(row["id"])

    exch.close_avg_px = 110.0
    result = ex.close_trade(trade_id, exit_price=None, reason="manual")
    assert result is not None
    assert len(exch.market_closes) == 1
    assert exch.market_closes[0]["coin"] == "BTC"
    # 5 units LONG @ 100 closed @ 110 ⇒ +$50
    assert result["pnl"] == pytest.approx(50.0)
    assert result["status"] == "CLOSED"


def test_check_stops_finalizes_rows_whose_position_disappeared(make_executor, tmp_db_path):
    ex, exch, info = make_executor()
    # Create an open trade with SL/TP triggers on the exchange.
    ex.execute_trade(_signal(), _position())

    # The exchange side decides SL triggered — no more live positions.
    info.state["assetPositions"] = []
    info.fills = [{"coin": "BTC", "px": 95.0, "sz": 5.0, "side": "A"}]

    closed = ex.check_stops(current_prices={})
    assert len(closed) == 1
    trade = closed[0]
    # LONG 5 @ 100 closed at 95 ⇒ -$25
    assert trade["pnl"] == pytest.approx(-25.0)
    assert trade["status"] == "STOPPED"

    # The row is no longer OPEN in SQLite.
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'").fetchone()[0]
    assert remaining == 0


def test_check_stops_leaves_rows_alone_when_position_still_live(make_executor):
    ex, _, info = make_executor()
    ex.execute_trade(_signal(), _position())
    # Pretend the position is still alive on the exchange.
    info.state["assetPositions"] = [{
        "position": {
            "coin": "BTC", "szi": "5.0", "entryPx": "100.0",
            "positionValue": "500.0", "unrealizedPnl": "0.0",
        }
    }]
    closed = ex.check_stops(current_prices={"BTC-USD": 100.0})
    assert closed == []


# ─────────────────────── cancel / queries ───────────────────────


def test_cancel_all_orders_cancels_every_open(make_executor):
    ex, exch, info = make_executor()
    info.open_orders_list = [
        {"coin": "BTC", "oid": 111},
        {"coin": "BTC", "oid": 222},
        {"coin": "ETH", "oid": 333},
    ]
    attempts = ex.cancel_all_orders()
    assert attempts == 3
    assert {c["oid"] for c in exch.cancels} == {111, 222, 333}


def test_cancel_all_orders_filters_by_symbol(make_executor):
    ex, exch, info = make_executor()
    info.open_orders_list = [
        {"coin": "BTC", "oid": 1},
        {"coin": "ETH", "oid": 2},
    ]
    attempts = ex.cancel_all_orders(symbol="ETH-USD")
    assert attempts == 1
    assert exch.cancels[0]["name"] == "ETH"


def test_get_open_positions_translates_symbols(make_executor):
    ex, _, info = make_executor()
    info.state["assetPositions"] = [{
        "position": {
            "coin": "ETH", "szi": "-2.5", "entryPx": "2000.0",
            "positionValue": "5000.0", "unrealizedPnl": "10.0",
            "leverage": {"value": 3}, "marginUsed": "1666.0",
        }
    }]
    positions = ex.get_open_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p["symbol"] == "ETH-USD"
    assert p["hl_coin"] == "ETH"
    assert p["direction"] == "SHORT"
    assert p["quantity"] == 2.5
    assert p["leverage"] == 3


def test_get_position_filter_by_symbol(make_executor):
    ex, _, info = make_executor()
    info.state["assetPositions"] = [
        {"position": {"coin": "BTC", "szi": "1.0", "entryPx": "100.0", "positionValue": "100.0", "unrealizedPnl": "0.0"}},
        {"position": {"coin": "ETH", "szi": "2.0", "entryPx": "200.0", "positionValue": "400.0", "unrealizedPnl": "0.0"}},
    ]
    assert ex.get_position("BTC-USD")["hl_coin"] == "BTC"
    assert ex.get_position("ETH-USD")["hl_coin"] == "ETH"
    assert ex.get_position("SOL-USD") is None


def test_get_account_summary_parses_state(make_executor):
    ex, _, info = make_executor()
    info.state["marginSummary"] = {
        "accountValue": "1234.56", "totalMarginUsed": "10.0",
        "totalNtlPos": "100.0", "totalRawUsd": "1234.56",
    }
    info.state["withdrawable"] = "1200.0"
    summary = ex.get_account_summary()
    assert summary["account_value"] == pytest.approx(1234.56)
    assert summary["withdrawable"] == pytest.approx(1200.0)
    assert summary["testnet"] is True


def test_get_fills_truncates_to_limit(make_executor):
    ex, _, info = make_executor()
    info.fills = [{"coin": "BTC", "px": float(i)} for i in range(5)]
    assert len(ex.get_fills(limit=3)) == 3


# ─────────────────────── dry_run ───────────────────────


def test_dry_run_skips_exchange_submission(make_executor):
    ex, exch, _ = make_executor(dry_run=True)
    oid = ex.execute_trade(_signal(), _position())
    assert oid is not None and oid.startswith("dryrun-")
    # No orders went out.
    assert exch.orders == []
    # But the trade was still recorded locally for the dashboard.
    assert len(ex.get_all_portfolios()) > 0


# ─────────────────────── Error handling ───────────────────────


def test_rejected_entry_returns_none(make_executor):
    ex, exch, _ = make_executor()
    exch.fail_entry = True
    assert ex.execute_trade(_signal(), _position()) is None


def test_market_close_exception_is_caught(make_executor, tmp_db_path, monkeypatch):
    ex, exch, _ = make_executor()
    ex.execute_trade(_signal(), _position())
    from db_schema import get_connection
    with get_connection(tmp_db_path) as conn:
        tid = int(conn.execute("SELECT id FROM trades WHERE status = 'OPEN'").fetchone()["id"])

    def boom(*a, **kw):
        raise RuntimeError("exchange down")
    monkeypatch.setattr(exch, "market_close", boom)

    result = ex.close_trade(tid, exit_price=105.0)
    assert result is None


# ─────────────────────── run_live CLI smoke ───────────────────────


def test_run_live_filters_unsupported_symbols():
    from run_live import _filter_supported
    out = _filter_supported(["BTC-USD", "SPY", "DOGE-USD", "ETH-USD"])
    assert out == ["BTC-USD", "ETH-USD"]


def test_run_live_requires_private_key(monkeypatch, capsys):
    monkeypatch.delenv(hc.PRIVATE_KEY_ENV_VAR, raising=False)
    from run_live import main
    rc = main(["--symbols", "BTC-USD", "--once"])
    assert rc == 2
    err = capsys.readouterr().err
    assert hc.PRIVATE_KEY_ENV_VAR in err


def test_run_live_refuses_mainnet_without_confirmation(monkeypatch, capsys):
    monkeypatch.setenv(hc.PRIVATE_KEY_ENV_VAR, "0x" + "dd" * 32)
    # Simulate user typing the wrong thing at the confirmation prompt.
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "no thanks")
    from run_live import main
    rc = main(["--live", "--symbols", "BTC-USD", "--once"])
    assert rc == 1
