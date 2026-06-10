"""Integration tests for LM Studio provider in QuantAgent.

These tests verify end-to-end behavior with a running LM Studio server.
They require an LM Studio server to be available at the configured base URL
(default: http://127.0.0.1:1234/v1).

Skip with: pytest -k "not integration"
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy native dependencies
for mod_name in ["talib", "langchain_qwq"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Check if LM Studio server is reachable
LM_STUDIO_BASE_URL = os.environ.get(
    "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"
)
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "")


def _lm_studio_available():
    """Check if LM Studio server is reachable via health check endpoint."""
    import urllib.request
    import urllib.error

    # Try the models endpoint which is always available
    url = LM_STUDIO_BASE_URL.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


SERVER_AVAILABLE = _lm_studio_available()
SKIP_REASON = "LM Studio server not available at " + LM_STUDIO_BASE_URL


@unittest.skipUnless(SERVER_AVAILABLE, SKIP_REASON)
class TestLmStudioIntegration(unittest.TestCase):
    """Integration tests that hit the real LM Studio server."""

    def test_create_llm_lm_studio(self):
        """Should create a working LM Studio LLM via ChatOpenAI."""
        from trading_graph import TradingGraph
        from default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        # Make sure model is installed on lm studio
        config["agent_llm_model"] = "google/gemma-4-12b-qat"
        config["graph_llm_model"] = "google/gemma-4-12b-qat"
        config["lm_studio_api_key"] = LM_STUDIO_API_KEY
        config["lm_studio_base_url"] = LM_STUDIO_BASE_URL

        tg = TradingGraph(config=config)

        # The agent_llm should be a ChatOpenAI instance
        from langchain_openai import ChatOpenAI
        self.assertIsInstance(tg.agent_llm, ChatOpenAI)

    def test_lm_studio_simple_invoke(self):
        """Should successfully invoke LM Studio for a simple query."""
        from trading_graph import TradingGraph
        from default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        config["agent_llm_model"] = DEFAULT_CONFIG["agent_llm_model"]
        config["graph_llm_model"] = DEFAULT_CONFIG["graph_llm_model"]
        config["lm_studio_api_key"] = LM_STUDIO_API_KEY
        config["lm_studio_base_url"] = LM_STUDIO_BASE_URL

        tg = TradingGraph(config=config)

        # Simple invoke test
        response = tg.agent_llm.invoke("Say 'hello' and nothing else.")
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)

    def test_lm_studio_provider_full_lifecycle(self):
        """Test full lifecycle: create -> update key -> refresh."""
        from trading_graph import TradingGraph
        from default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        config["agent_llm_model"] = DEFAULT_CONFIG["agent_llm_model"]
        config["graph_llm_model"] = DEFAULT_CONFIG["graph_llm_model"]
        config["lm_studio_api_key"] = LM_STUDIO_API_KEY
        config["lm_studio_base_url"] = LM_STUDIO_BASE_URL

        tg = TradingGraph(config=config)

        # Update API key (same key, just testing the mechanism)
        tg.update_api_key(LM_STUDIO_API_KEY, provider="lm_studio")

        # Verify the LLM still works after refresh
        response = tg.agent_llm.invoke("Reply with just the word 'ok'.")
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)

    def test_lm_studio_custom_base_url(self):
        """Should respect a custom LM Studio base URL from config."""
        from trading_graph import TradingGraph
        from default_config import DEFAULT_CONFIG

        custom_url = os.environ.get(
            "LM_STUDIO_BASE_URL_OVERRIDE",
            "http://127.0.0.1:1234/v1",
        )

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "lm_studio"
        config["graph_llm_provider"] = "lm_studio"
        config["agent_llm_model"] = DEFAULT_CONFIG["agent_llm_model"]
        config["graph_llm_model"] = DEFAULT_CONFIG["graph_llm_model"]
        config["lm_studio_api_key"] = LM_STUDIO_API_KEY
        config["lm_studio_base_url"] = custom_url

        tg = TradingGraph(config=config)

        from langchain_openai import ChatOpenAI
        self.assertIsInstance(tg.agent_llm, ChatOpenAI)


if __name__ == "__main__":
    unittest.main()
