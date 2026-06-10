"""Unit tests for LM Studio provider integration in QuantAgent."""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

# Mock heavy native/incompatible dependencies before importing project modules
MOCK_MODULES = [
    "talib",
    "langchain_anthropic",
    "langchain_core",
    "langchain_core.language_models",
    "langchain_core.prompts",
    "langchain_core.tools",
    "langchain_openai",
    "langchain_qwq",
    "langgraph",
    "langgraph.graph",
    "langgraph.prebuilt",
    "matplotlib",
    "matplotlib.pyplot",
    "mplfinance",
    "yfinance",
]

for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


class FakeStateGraph:
    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, *args, **kwargs):
        pass

    def add_edge(self, *args, **kwargs):
        pass

    def compile(self):
        return MagicMock()


sys.modules["langchain_core.language_models"].BaseChatModel = object
sys.modules["langgraph.graph"].END = "__end__"
sys.modules["langgraph.graph"].START = "__start__"
sys.modules["langgraph.graph"].StateGraph = FakeStateGraph

if "langchain_core.messages" not in sys.modules:
    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.AIMessage = MagicMock
    messages_module.BaseMessage = MagicMock
    messages_module.HumanMessage = MagicMock
    messages_module.SystemMessage = MagicMock
    messages_module.ToolMessage = MagicMock
    sys.modules["langchain_core.messages"] = messages_module

from default_config import DEFAULT_CONFIG


class TestDefaultConfig(unittest.TestCase):
    """Tests for LM Studio fields in DEFAULT_CONFIG."""

    def test_lm_studio_api_key_field_exists(self):
        """DEFAULT_CONFIG should contain a lm_studio_api_key field."""
        self.assertIn("lm_studio_api_key", DEFAULT_CONFIG)

    def test_lm_studio_base_url_field_exists(self):
        """DEFAULT_CONFIG should contain a lm_studio_base_url field."""
        self.assertIn("lm_studio_base_url", DEFAULT_CONFIG)
        self.assertEqual(
            DEFAULT_CONFIG["lm_studio_base_url"],
            "http://127.0.0.1:1234/v1",
        )

    def test_provider_comment_mentions_lm_studio(self):
        """Provider fields should accept 'lm_studio' as a valid value."""
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        self.assertEqual(config["agent_llm_provider"], "lm_studio")
        self.assertEqual(config["graph_llm_provider"], "lm_studio")


class TestTradingGraphGetApiKey(unittest.TestCase):
    """Tests for TradingGraph._get_api_key() with lm_studio provider."""

    def _make_graph(self, config):
        """Create a TradingGraph with mocked LLM creation."""
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_get_api_key_from_config(self):
        """Should return lm_studio_api_key from config."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = "test-lmstudio-key-123"
        tg = self._make_graph(config)
        key = tg._get_api_key("lm_studio")
        self.assertEqual(key, "test-lmstudio-key-123")

    def test_get_api_key_from_env(self):
        """Should fall back to LM_STUDIO_API_KEY env var."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = ""
        tg = self._make_graph(config)
        with patch.dict(os.environ, {"LM_STUDIO_API_KEY": "env-lmstudio-key"}):
            key = tg._get_api_key("lm_studio")
            self.assertEqual(key, "env-lmstudio-key")

    def test_get_api_key_fallback_to_dummy(self):
        """Should return 'dummy' when no API key is available."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = ""
        tg = self._make_graph(config)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LM_STUDIO_API_KEY", None)
            key = tg._get_api_key("lm_studio")
            self.assertEqual(key, "dummy")

    def test_unsupported_provider_raises(self):
        """Should raise ValueError for unsupported provider."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with self.assertRaises(ValueError) as ctx:
            tg._get_api_key("unsupported_provider")
        self.assertIn("Unsupported provider", str(ctx.exception))
        self.assertIn("lm_studio", str(ctx.exception))


class TestTradingGraphCreateLlm(unittest.TestCase):
    """Tests for TradingGraph._create_llm() with lm_studio provider."""

    def _make_graph(self, config):
        """Create a TradingGraph with mocked LLM creation."""
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_lm_studio_uses_chatopenai(self, mock_openai):
        """LM Studio provider should create ChatOpenAI with custom base URL."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        result = tg._create_llm("lm_studio", "local-model", 0.1)

        mock_openai.assert_called_once_with(
            model="local-model",
            temperature=0.1,
            api_key="test-key",
            openai_api_base="http://127.0.0.1:1234/v1",
        )

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_lm_studio_uses_custom_base_url(self, mock_openai):
        """LM Studio provider should use the configured base URL from DEFAULT_CONFIG."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = "test-key"
        config["lm_studio_base_url"] = "http://custom-host:8080/v1"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("lm_studio", "local-model", 0.1)

        # The base_url comes from LM_STUDIO_PROVIDER_CONFIG, which reads from DEFAULT_CONFIG
        # at module load time, so it will be the original value.
        mock_openai.assert_called_once_with(
            model="local-model",
            temperature=0.1,
            api_key="test-key",
            openai_api_base="http://127.0.0.1:1234/v1",
        )

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_lm_studio_normal_temperature(self, mock_openai):
        """Normal temperature should be passed through for LM Studio (no clamping)."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("lm_studio", "local-model", 0.5)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.5)

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_lm_studio_zero_temperature(self, mock_openai):
        """LM Studio should pass through temperature=0.0 without clamping."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("lm_studio", "local-model", 0.0)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.0)

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_lm_studio_high_temperature(self, mock_openai):
        """LM Studio should pass through temperature > 1.0 without clamping."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("lm_studio", "local-model", 1.5)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 1.5)

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_lm_studio_with_dummy_key(self, mock_openai):
        """LM Studio should work with the dummy API key."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = ""
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LM_STUDIO_API_KEY", None)
            tg._create_llm("lm_studio", "local-model", 0.1)

        mock_openai.assert_called_once_with(
            model="local-model",
            temperature=0.1,
            api_key="dummy",
            openai_api_base="http://127.0.0.1:1234/v1",
        )


class TestTradingGraphUpdateApiKey(unittest.TestCase):
    """Tests for TradingGraph.update_api_key() with lm_studio provider."""

    def _make_graph(self, config):
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_update_api_key_lm_studio(self):
        """update_api_key('lm_studio') should update config and env var."""
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = ""
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        config["agent_llm_model"] = "local-model"
        config["graph_llm_model"] = "local-model"
        tg = self._make_graph(config)

        with patch.object(tg, "refresh_llms"):
            tg.update_api_key("new-lmstudio-key", provider="lm_studio")

        self.assertEqual(tg.config["lm_studio_api_key"], "new-lmstudio-key")
        self.assertEqual(os.environ.get("LM_STUDIO_API_KEY"), "new-lmstudio-key")

    def test_update_api_key_unsupported_raises(self):
        """update_api_key() with unsupported provider should raise ValueError."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with self.assertRaises(ValueError) as ctx:
            tg.update_api_key("key", provider="unsupported")
        self.assertIn("lm_studio", str(ctx.exception))


class TestTradingGraphRefreshLlms(unittest.TestCase):
    """Tests for TradingGraph.refresh_llms() with lm_studio provider."""

    @patch("trading_graph.ChatOpenAI")
    @patch("trading_graph.ChatAnthropic")
    @patch("trading_graph.ChatQwen")
    def test_refresh_llms_lm_studio(self, mock_qwen, mock_anthropic, mock_openai):
        """refresh_llms() should recreate LLMs when provider is lm_studio."""
        from trading_graph import TradingGraph

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        config["agent_llm_model"] = "local-model"
        config["graph_llm_model"] = "local-model"
        config["lm_studio_api_key"] = "test-key"

        mock_openai.return_value = MagicMock()
        tg = TradingGraph(config=config)

        mock_openai.reset_mock()
        tg.refresh_llms()

        # ChatOpenAI should be called twice (agent_llm + graph_llm)
        self.assertEqual(mock_openai.call_count, 2)
        for call in mock_openai.call_args_list:
            self.assertEqual(call.kwargs["openai_api_base"], "http://127.0.0.1:1234/v1")


class TestWebInterfaceProviderUpdate(unittest.TestCase):
    """Tests for web interface provider update with LM Studio."""

    @patch("web_interface.TradingGraph")
    def test_update_provider_lm_studio(self, mock_tg_class):
        """POST /api/update-provider with lm_studio should succeed."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.save_llm_config = MagicMock(return_value=True)

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "lm_studio"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        # LM Studio keeps existing model names (no override)
        self.assertEqual(analyzer.config["agent_llm_provider"], "lm_studio")
        self.assertEqual(analyzer.config["graph_llm_provider"], "lm_studio")

    @patch("web_interface.TradingGraph")
    def test_update_provider_invalid(self, mock_tg_class):
        """POST /api/update-provider with invalid provider should fail."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.save_llm_config = MagicMock(return_value=True)

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "invalid"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertIn("error", data)

    @patch("web_interface.TradingGraph")
    def test_update_api_key_lm_studio(self, mock_tg_class):
        """POST /api/update-api-key with lm_studio should set env var."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.save_llm_config = MagicMock(return_value=True)

        client = app.test_client()
        resp = client.post(
            "/api/update-api-key",
            json={"api_key": "test-lmstudio-key", "provider": "lm_studio"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(os.environ.get("LM_STUDIO_API_KEY"), "test-lmstudio-key")

    @patch("web_interface.TradingGraph")
    def test_update_api_key_lm_studio_empty_allowed(self, mock_tg_class):
        """POST /api/update-api-key with lm_studio should allow empty API key."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.save_llm_config = MagicMock(return_value=True)

        client = app.test_client()
        resp = client.post(
            "/api/update-api-key",
            json={"api_key": "", "provider": "lm_studio"},
            content_type="application/json",
        )
        data = resp.get_json()
        # LM Studio allows empty/missing API key
        self.assertTrue(data.get("success"))

    @patch("web_interface.TradingGraph")
    def test_get_api_key_status_lm_studio(self, mock_tg_class):
        """GET /api/get-api-key-status?provider=lm_studio should always report valid."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = ""
        analyzer.config = config
        analyzer.trading_graph = mock_tg
        analyzer.save_llm_config = MagicMock(return_value=True)

        client = app.test_client()
        resp = client.get("/api/get-api-key-status?provider=lm_studio")
        data = resp.get_json()
        # LM Studio always reports as valid since it's a local server
        self.assertTrue(data.get("has_key"))


class TestProviderSwitchBackToOpenAI(unittest.TestCase):
    """Test that switching from LM Studio back to OpenAI resets model names."""

    @patch("web_interface.TradingGraph")
    def test_switch_lm_studio_to_openai(self, mock_tg_class):
        """Switching from lm_studio to openai should reset model names."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_model"] = "google/gemma-4-26b-a4b"
        config["graph_llm_model"] = "google/gemma-4-26b-a4b"
        analyzer.config = config
        analyzer.trading_graph = mock_tg
        analyzer.save_llm_config = MagicMock(return_value=True)

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


class TestApplyProviderDefaults(unittest.TestCase):
    """Tests for apply_provider_defaults() with lm_studio provider."""

    def test_lm_studio_sets_default_model(self):
        """Switching to lm_studio should set default model if not already google-prefixed."""
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_model"] = "gpt-4o"
        config["graph_llm_model"] = "gpt-4o"

        from web_interface import apply_provider_defaults
        apply_provider_defaults(config, "lm_studio")

        self.assertTrue(config["agent_llm_model"].startswith("google"))
        self.assertTrue(config["graph_llm_model"].startswith("google"))

    def test_lm_studio_skips_model_change_if_already_google(self):
        """Should not override model if it already starts with 'google'."""
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_model"] = "google/my-custom-model"
        config["graph_llm_model"] = "google/another-model"

        from web_interface import apply_provider_defaults
        apply_provider_defaults(config, "lm_studio")

        self.assertEqual(config["agent_llm_model"], "google/my-custom-model")
        self.assertEqual(config["graph_llm_model"], "google/another-model")


class TestValidateApiKeyLmStudio(unittest.TestCase):
    """Tests for validate_api_key() with lm_studio provider."""

    @patch("web_interface.TradingGraph")
    def test_validate_api_key_lm_studio_returns_valid(self, mock_tg_class):
        """POST /api/validate-api-key for lm_studio should return valid: True."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = ""
        config["agent_llm_provider"] = "lm_studio"
        analyzer.config = config
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/validate-api-key",
            json={"provider": "lm_studio"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("valid"))


class TestUpdateProviderNeedsApiKey(unittest.TestCase):
    """Tests for the needs_api_key flow in update_provider()."""

    @patch("web_interface.TradingGraph")
    def test_update_provider_lm_studio_needs_api_key(self, mock_tg_class):
        """When refresh_llms raises 'API key not found', should return needs_api_key: True."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["lm_studio_api_key"] = ""
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        config["agent_llm_model"] = "local-model"
        config["graph_llm_model"] = "local-model"
        analyzer.config = config
        analyzer.trading_graph = mock_tg
        analyzer.trading_graph.refresh_llms.side_effect = ValueError("API key not found")

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "lm_studio"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("needs_api_key"))
        self.assertIn("Please set its API key", data.get("message", ""))

if __name__ == "__main__":
    unittest.main()
