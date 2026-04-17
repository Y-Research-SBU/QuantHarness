"""
Anthropic Claude configuration for QuantAgent.
Uses claude-haiku-4-5 for sub-agents (cheap) and claude-sonnet-4 for decision agent (smart).
"""

ANTHROPIC_CONFIG = {
    "agent_llm_model": "claude-haiku-4-5-20251001",      # Sub-agents: indicator, pattern, trend
    "graph_llm_model": "claude-sonnet-4-20250514",     # Decision agent (smart)
    "agent_llm_provider": "anthropic",
    "graph_llm_provider": "anthropic",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
}
