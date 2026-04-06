"""Integration tests for MiniMax provider in QuantAgent.

These tests verify end-to-end behavior with the actual MiniMax API.
They require the MINIMAX_API_KEY environment variable to be set.
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

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that hit the real MiniMax API."""

    def test_create_llm_minimax_m27(self):
        """Should create a working MiniMax M2.7 LLM via ChatOpenAI."""
        from trading_graph import TradingGraph
        from default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        config["agent_llm_model"] = "MiniMax-M2.7"
        config["graph_llm_model"] = "MiniMax-M2.7"
        config["minimax_api_key"] = MINIMAX_API_KEY

        tg = TradingGraph(config=config)

        # The agent_llm should be a ChatOpenAI instance
        from langchain_openai import ChatOpenAI
        self.assertIsInstance(tg.agent_llm, ChatOpenAI)

    def test_minimax_simple_invoke(self):
        """Should successfully invoke MiniMax M2.7-highspeed for a simple query."""
        from trading_graph import TradingGraph
        from default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        config["agent_llm_model"] = "MiniMax-M2.7-highspeed"
        config["graph_llm_model"] = "MiniMax-M2.7-highspeed"
        config["minimax_api_key"] = MINIMAX_API_KEY

        tg = TradingGraph(config=config)

        # Simple invoke test
        response = tg.agent_llm.invoke("Say 'hello' and nothing else.")
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)

    def test_minimax_provider_full_lifecycle(self):
        """Test full lifecycle: create -> update key -> refresh."""
        from trading_graph import TradingGraph
        from default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        config["agent_llm_model"] = "MiniMax-M2.7-highspeed"
        config["graph_llm_model"] = "MiniMax-M2.7-highspeed"
        config["minimax_api_key"] = MINIMAX_API_KEY

        tg = TradingGraph(config=config)

        # Update API key (same key, just testing the mechanism)
        tg.update_api_key(MINIMAX_API_KEY, provider="minimax")

        # Verify the LLM still works after refresh
        response = tg.agent_llm.invoke("Reply with just the word 'ok'.")
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)


if __name__ == "__main__":
    unittest.main()
