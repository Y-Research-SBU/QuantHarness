"""Unit tests for MiniMax provider integration in QuantAgent."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy native/incompatible dependencies before importing project modules
# TA-Lib requires a C library; langchain has pydantic v1/v2 conflicts
for mod_name in [
    "talib",
    "langchain_qwq",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from default_config import DEFAULT_CONFIG


class TestDefaultConfig(unittest.TestCase):
    """Tests for MiniMax fields in DEFAULT_CONFIG."""

    def test_minimax_api_key_field_exists(self):
        """DEFAULT_CONFIG should contain a minimax_api_key field."""
        self.assertIn("minimax_api_key", DEFAULT_CONFIG)

    def test_provider_comment_mentions_minimax(self):
        """Provider fields should accept 'minimax' as a valid value."""
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        self.assertEqual(config["agent_llm_provider"], "minimax")
        self.assertEqual(config["graph_llm_provider"], "minimax")


class TestTradingGraphGetApiKey(unittest.TestCase):
    """Tests for TradingGraph._get_api_key() with minimax provider."""

    def _make_graph(self, config):
        """Create a TradingGraph with mocked LLM creation."""
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_get_api_key_from_config(self):
        """Should return minimax_api_key from config."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-minimax-key-123"
        tg = self._make_graph(config)
        key = tg._get_api_key("minimax")
        self.assertEqual(key, "test-minimax-key-123")

    def test_get_api_key_from_env(self):
        """Should fall back to MINIMAX_API_KEY env var."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        tg = self._make_graph(config)
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-minimax-key"}):
            key = tg._get_api_key("minimax")
            self.assertEqual(key, "env-minimax-key")

    def test_get_api_key_missing_raises(self):
        """Should raise ValueError if no MiniMax API key is available."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        tg = self._make_graph(config)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                tg._get_api_key("minimax")
            self.assertIn("MiniMax", str(ctx.exception))

    def test_unsupported_provider_raises(self):
        """Should raise ValueError for unsupported provider."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with self.assertRaises(ValueError) as ctx:
            tg._get_api_key("unsupported_provider")
        self.assertIn("Unsupported provider", str(ctx.exception))
        self.assertIn("minimax", str(ctx.exception))


class TestTradingGraphCreateLlm(unittest.TestCase):
    """Tests for TradingGraph._create_llm() with minimax provider."""

    def _make_graph(self, config):
        """Create a TradingGraph with mocked LLM creation."""
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_uses_chatopenai(self, mock_openai):
        """MiniMax provider should create ChatOpenAI with custom base URL."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        result = tg._create_llm("minimax", "MiniMax-M2.7", 0.1)

        mock_openai.assert_called_once_with(
            model="MiniMax-M2.7",
            temperature=0.1,
            api_key="test-key",
            openai_api_base="https://api.minimax.io/v1",
        )

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_clamps_temperature(self, mock_openai):
        """MiniMax temperature should be clamped to (0.0, 1.0]."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("minimax", "MiniMax-M2.7", 0.0)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.01)

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_clamps_high_temperature(self, mock_openai):
        """MiniMax temperature > 1.0 should be clamped to 1.0."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("minimax", "MiniMax-M2.7", 1.5)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 1.0)

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_normal_temperature(self, mock_openai):
        """Normal temperature within range should be passed through."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("minimax", "MiniMax-M2.7", 0.5)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.5)


class TestTradingGraphUpdateApiKey(unittest.TestCase):
    """Tests for TradingGraph.update_api_key() with minimax provider."""

    def _make_graph(self, config):
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_update_api_key_minimax(self):
        """update_api_key('minimax') should update config and env var."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        config["agent_llm_model"] = "MiniMax-M2.7"
        config["graph_llm_model"] = "MiniMax-M2.7"
        tg = self._make_graph(config)

        with patch.object(tg, "refresh_llms"):
            tg.update_api_key("new-minimax-key", provider="minimax")

        self.assertEqual(tg.config["minimax_api_key"], "new-minimax-key")
        self.assertEqual(os.environ.get("MINIMAX_API_KEY"), "new-minimax-key")

    def test_update_api_key_unsupported_raises(self):
        """update_api_key() with unsupported provider should raise ValueError."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with self.assertRaises(ValueError) as ctx:
            tg.update_api_key("key", provider="unsupported")
        self.assertIn("minimax", str(ctx.exception))


class TestTradingGraphRefreshLlms(unittest.TestCase):
    """Tests for TradingGraph.refresh_llms() with minimax provider."""

    @patch("trading_graph.ChatOpenAI")
    @patch("trading_graph.ChatAnthropic")
    @patch("trading_graph.ChatQwen")
    def test_refresh_llms_minimax(self, mock_qwen, mock_anthropic, mock_openai):
        """refresh_llms() should recreate LLMs when provider is minimax."""
        from trading_graph import TradingGraph

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        config["agent_llm_model"] = "MiniMax-M2.7"
        config["graph_llm_model"] = "MiniMax-M2.7"
        config["minimax_api_key"] = "test-key"

        mock_openai.return_value = MagicMock()
        tg = TradingGraph(config=config)

        mock_openai.reset_mock()
        tg.refresh_llms()

        # ChatOpenAI should be called twice (agent_llm + graph_llm)
        self.assertEqual(mock_openai.call_count, 2)
        for call in mock_openai.call_args_list:
            self.assertEqual(call.kwargs["openai_api_base"], "https://api.minimax.io/v1")


class TestWebInterfaceProviderUpdate(unittest.TestCase):
    """Tests for web interface provider update with MiniMax."""

    @patch("web_interface.TradingGraph")
    def test_update_provider_minimax(self, mock_tg_class):
        """POST /api/update-provider with minimax should succeed."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "minimax"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(analyzer.config["agent_llm_model"], "MiniMax-M2.7")
        self.assertEqual(analyzer.config["graph_llm_model"], "MiniMax-M2.7")

    @patch("web_interface.TradingGraph")
    def test_update_provider_invalid(self, mock_tg_class):
        """POST /api/update-provider with invalid provider should fail."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "invalid"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertIn("error", data)

    @patch("web_interface.TradingGraph")
    def test_update_api_key_minimax(self, mock_tg_class):
        """POST /api/update-api-key with minimax should set env var."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-api-key",
            json={"api_key": "test-mm-key", "provider": "minimax"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(os.environ.get("MINIMAX_API_KEY"), "test-mm-key")

    @patch("web_interface.TradingGraph")
    def test_get_api_key_status_minimax(self, mock_tg_class):
        """GET /api/get-api-key-status?provider=minimax should work."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-minimax-key-12345"
        analyzer.config = config
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/api/get-api-key-status?provider=minimax")
        data = resp.get_json()
        self.assertTrue(data.get("has_key"))
        self.assertIn("masked_key", data)

    @patch("web_interface.TradingGraph")
    def test_get_api_key_status_minimax_missing(self, mock_tg_class):
        """GET /api/get-api-key-status?provider=minimax with no key."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        analyzer.config = config
        analyzer.trading_graph = mock_tg

        os.environ.pop("MINIMAX_API_KEY", None)

        client = app.test_client()
        resp = client.get("/api/get-api-key-status?provider=minimax")
        data = resp.get_json()
        self.assertFalse(data.get("has_key"))


class TestProviderSwitchBackToOpenAI(unittest.TestCase):
    """Test that switching from MiniMax back to OpenAI resets model names."""

    @patch("web_interface.TradingGraph")
    def test_switch_minimax_to_openai(self, mock_tg_class):
        """Switching from minimax to openai should reset model names."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_model"] = "MiniMax-M2.7"
        config["graph_llm_model"] = "MiniMax-M2.7"
        analyzer.config = config
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "openai"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(analyzer.config["agent_llm_model"], "gpt-4o-mini")
        self.assertEqual(analyzer.config["graph_llm_model"], "gpt-4o")


if __name__ == "__main__":
    unittest.main()
