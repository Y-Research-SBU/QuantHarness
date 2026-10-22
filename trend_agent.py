"""
Agent for trend analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to generate and interpret trendline charts for short-term prediction.
"""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from openai import RateLimitError


# --- Retry wrapper for LLM invocation ---
def invoke_with_retry(call_fn, *args, retries=3, wait_sec=4):
    """
    Retry a function call with exponential backoff for rate limits or errors.
    """
    for attempt in range(retries):
        try:
            result = call_fn(*args)
            return result
        except RateLimitError:
            print(
                f"Rate limit hit, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
            )
        except Exception as e:
            print(
                f"Other error: {e}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
            )
        # Only sleep if not the last attempt
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("Max retries exceeded")


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """
    Create a trend analysis agent node for HFT. The agent uses precomputed images from state or falls back to tool generation.
    """

    def trend_agent_node(state):
        # --- Tool definitions ---
        tools = [toolkit.generate_trend_image]
        time_frame = state["time_frame"]

        # --- Check for precomputed image in state ---
        trend_image_b64 = state.get("trend_image")

        messages = []

        # --- If no precomputed image, fall back to tool generation ---
        if not trend_image_b64:
            print("No precomputed trend image found in state, generating with tool...")

            # --- System prompt for LLM ---
            system_prompt = (
                "You are a K-line trend pattern recognition assistant operating in a high-frequency trading context. "
                "You must first call the tool `generate_trend_image` using the provided `kline_data`. "
                "Once the chart is generated, analyze the image for support/resistance trendlines and known candlestick patterns. "
                "Only then should you proceed to make a prediction about the short-term trend (upward, downward, or sideways). "
                "Do not make any predictions before generating and analyzing the image."
            )

            # --- Compose messages for the first round ---
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Here is the recent kline data:\n{json.dumps(state['kline_data'], indent=2)}"
                ),
            ]

            # --- Prepare tool chain ---
            chain = tool_llm.bind_tools(tools)

            # --- Step 1: Let LLM decide if it wants to call generate_trend_image ---
            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            # --- Step 2: Handle tool call (generate_trend_image) ---
            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    # Always provide kline_data
                    import copy

                    tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                    tool_fn = next(t for t in tools if t.name == tool_name)
                    tool_result = tool_fn.invoke(tool_args)
                    trend_image_b64 = tool_result.get("trend_image")
                    messages.append(
                        ToolMessage(
                            tool_call_id=call["id"], content=json.dumps(tool_result)
                        )
                    )
        else:
            print("Using precomputed trend image from state")

        # --- Step 3: Vision analysis with image (precomputed or generated) ---
        if trend_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"This candlestick ({time_frame} K-line) chart includes automated trendlines: the **blue line** is support, and the **red line** is resistance, both derived from recent closing prices.\n\n"
                        "Analyze how price interacts with these lines — are candles bouncing off, breaking through, or compressing between them?\n\n"
                        "Based on trendline slope, spacing, and recent K-line behavior, predict the likely short-term trend: **upward**, **downward**, or **sideways**. "
                        "Support your prediction with respect to prediction, reasoning, signals."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
                },
            ]

            # Create messages - ensure HumanMessage has valid content
            # For Anthropic, SystemMessage is extracted separately, but messages array must have at least one message
            human_msg = HumanMessage(content=image_prompt)
            
            # Verify HumanMessage content is valid
            if not human_msg.content:
                raise ValueError("HumanMessage content is empty")
            if isinstance(human_msg.content, list) and len(human_msg.content) == 0:
                raise ValueError("HumanMessage content list is empty")
            
            messages = [
                SystemMessage(
                    content="You are a K-line trend pattern recognition assistant operating in a high-frequency trading context. "
                    "Your task is to analyze candlestick charts annotated with support and resistance trendlines."
                ),
                human_msg,
            ]
            
            try:
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    messages,
                )
            except Exception as e:
                error_str = str(e)
                # Handle Anthropic's "at least one message is required" error
                # This can happen when SystemMessage extraction leaves empty messages array
                if "at least one message" in error_str.lower():
                    # Retry with only HumanMessage (SystemMessage will be lost but Anthropic should work)
                    print("Retrying with HumanMessage only due to Anthropic message conversion issue...")
                    final_response = invoke_with_retry(
                        graph_llm.invoke,
                        [human_msg],
                    )
                else:
                    raise
        else:
            # If no image was generated, fall back to reasoning with messages
            final_response = invoke_with_retry(chain.invoke, messages)

        return {
            "messages": messages + [final_response],
            # "trend_report": final_response.content,
            "trend_report": str(final_response.content),
            "trend_image": trend_image_b64,
            "trend_image_filename": "trend_graph.png",
            "trend_image_description": (
                "Trend-enhanced candlestick chart with support/resistance lines"
                if trend_image_b64
                else None
            ),
        }

    return trend_agent_node
