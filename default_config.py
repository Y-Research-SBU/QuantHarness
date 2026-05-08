DEFAULT_CONFIG = {
    "agent_llm_model": "gpt-4o-mini",
    "graph_llm_model": "gpt-4o",
    "agent_llm_provider": "openai",  # "openai", "anthropic", "qwen", "minimax", or "minimax_cn"
    "graph_llm_provider": "openai",  # "openai", "anthropic", "qwen", "minimax", or "minimax_cn"
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "sk-",  # OpenAI API key
    "anthropic_api_key": "sk-",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
    "qwen_api_key": "sk-",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
    "minimax_api_key": "",  # MiniMax API key (optional, can also use MINIMAX_API_KEY env var)
    "minimax_cn_api_key": "",  # MiniMax CN API key (optional, can also use MINIMAX_CN_API_KEY or MINIMAX_API_KEY env var)
}
