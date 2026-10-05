"""Unit tests for Cheaper Inference provider integration in QuantHarness."""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

# Mock heavy native/incompatible dependencies before importing project modules
# TA-Lib requires a C library; langchain has pydantic v1/v2 conflicts
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
    """Tests for Cheaper Inference fields in DEFAULT_CONFIG."""

    def test_cheaperinference_api_key_field_exists(self):
        """DEFAULT_CONFIG should contain a cheaperinference_api_key field."""
        self.assertIn("cheaperinference_api_key", DEFAULT_CONFIG)


class TestTradingGraphCheaperInference(unittest.TestCase):
    """Tests for TradingGraph with the cheaperinference provider."""

    def _make_graph(self, config):
        """Create a TradingGraph with mocked LLM creation."""
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_get_api_key_from_config(self):
        """Should return cheaperinference_api_key from config."""
        config = DEFAULT_CONFIG.copy()
        config["cheaperinference_api_key"] = "test-ci-key-123"
        tg = self._make_graph(config)
        self.assertEqual(tg._get_api_key("cheaperinference"), "test-ci-key-123")

    def test_get_api_key_from_env(self):
        """Should fall back to CHEAPER_INFERENCE_API_KEY env var."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with patch.dict(os.environ, {"CHEAPER_INFERENCE_API_KEY": "env-ci-key"}, clear=True):
            self.assertEqual(tg._get_api_key("cheaperinference"), "env-ci-key")

    def test_get_api_key_missing_raises(self):
        """Should raise ValueError if no Cheaper Inference API key is available."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                tg._get_api_key("cheaperinference")
            self.assertIn("Cheaper Inference", str(ctx.exception))

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_uses_chatopenai(self, mock_openai):
        """Cheaper Inference should create ChatOpenAI with its base URL."""
        config = DEFAULT_CONFIG.copy()
        config["cheaperinference_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        tg._create_llm("cheaperinference", "gpt-5.4-mini", 0.1)

        mock_openai.assert_called_once_with(
            model="gpt-5.4-mini",
            temperature=0.1,
            api_key="test-key",
            openai_api_base="https://api.cheaperinference.com/v1",
        )

    def test_update_api_key(self):
        """update_api_key('cheaperinference') should update config and env var."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(tg, "refresh_llms"):
                tg.update_api_key("new-ci-key", provider="cheaperinference")
            self.assertEqual(os.environ.get("CHEAPER_INFERENCE_API_KEY"), "new-ci-key")

        self.assertEqual(tg.config["cheaperinference_api_key"], "new-ci-key")


class TestWebInterfaceCheaperInference(unittest.TestCase):
    """Tests for web interface routes with the cheaperinference provider."""

    def _setup_analyzer(self, mock_tg_class, config=None):
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = config if config is not None else DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.save_llm_config = MagicMock(return_value=True)
        return app, analyzer

    @patch("web_interface.TradingGraph")
    def test_update_provider(self, mock_tg_class):
        """POST /api/update-provider with cheaperinference should set default models."""
        app, analyzer = self._setup_analyzer(mock_tg_class)

        resp = app.test_client().post(
            "/api/update-provider",
            json={"provider": "cheaperinference"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(analyzer.config["agent_llm_model"], "gpt-5.4-mini")
        self.assertEqual(analyzer.config["graph_llm_model"], "gpt-5.4")

    @patch("web_interface.TradingGraph")
    def test_update_api_key(self, mock_tg_class):
        """POST /api/update-api-key with cheaperinference should set env var and provider."""
        app, analyzer = self._setup_analyzer(mock_tg_class)

        with patch.dict(os.environ, {}, clear=True):
            resp = app.test_client().post(
                "/api/update-api-key",
                json={"api_key": "test-ci-key", "provider": "cheaperinference"},
                content_type="application/json",
            )
            data = resp.get_json()
            self.assertTrue(data.get("success"))
            self.assertEqual(os.environ.get("CHEAPER_INFERENCE_API_KEY"), "test-ci-key")
        self.assertEqual(analyzer.config["cheaperinference_api_key"], "test-ci-key")
        self.assertEqual(analyzer.config["agent_llm_provider"], "cheaperinference")

    @patch("web_interface.TradingGraph")
    def test_get_api_key_status(self, mock_tg_class):
        """GET /api/get-api-key-status?provider=cheaperinference should report the key."""
        config = DEFAULT_CONFIG.copy()
        config["cheaperinference_api_key"] = "test-ci-key-12345"
        app, _ = self._setup_analyzer(mock_tg_class, config)

        with patch.dict(os.environ, {}, clear=True):
            resp = app.test_client().get("/api/get-api-key-status?provider=cheaperinference")
        data = resp.get_json()
        self.assertTrue(data.get("has_key"))
        self.assertIn("masked_key", data)


if __name__ == "__main__":
    unittest.main()
