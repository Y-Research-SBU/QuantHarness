"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.
"""


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """

    def trade_decision_node(state) -> dict:
        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        kronos_forecast = state.get("kronos_forecast", "")
        kronos_data = state.get("kronos_forecast_data") or {}
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        # --- System prompt for LLM ---
        prompt = f"""You are a high-frequency quantitative trading (HFT) analyst operating on the current {time_frame} K-line chart for {stock_name}. Your task is to issue an **immediate execution order**: **LONG** or **SHORT**. ⚠️ HOLD is prohibited due to HFT constraints.

            Your decision should forecast the market move over the **next N candlesticks**, where:
            - For example: TIME_FRAME = 15min, N = 1 → Predict the next 15 minutes.
            - TIME_FRAME = 4hour, N = 1 → Predict the next 4 hours.

            Base your decision on the combined strength, alignment, and timing of the following three reports:

            ---

            ### 1. Technical Indicator Report:
            - Evaluate momentum (e.g., MACD, ROC) and oscillators (e.g., RSI, Stochastic, Williams %R).
            - Give **higher weight to strong directional signals** such as MACD crossovers, RSI divergence, extreme overbought/oversold levels.
            - **Ignore or down-weight neutral or mixed signals** unless they align across multiple indicators.

            ---

            ### 2. Pattern Report:
            - Only act on bullish or bearish patterns if:
            - The pattern is **clearly recognizable and mostly complete**, and
            - A **breakout or breakdown is already underway** or highly probable based on price and momentum (e.g., strong wick, volume spike, engulfing candle).
            - **Do NOT act** on early-stage or speculative patterns. Do not treat consolidating setups as tradable unless there is **breakout confirmation** from other reports.

            ---

            ### 3. Trend Report:
            - Analyze how price interacts with support and resistance:
            - An **upward sloping support line** suggests buying interest.
            - A **downward sloping resistance line** suggests selling pressure.
            - If price is compressing between trendlines:
            - Predict breakout **only when confluence exists with strong candles or indicator confirmation**.
            - **Do NOT assume breakout direction** from geometry alone.

            ---

            ### 4. Kronos Forecast Report:
            - The Kronos foundation model produces a numerical forecast of the next several candles, expressed as **direction (UP / DOWN / NEUTRAL)**, **predicted percent change vs. last close**, and a **confidence score** (0-1).
            - Treat Kronos as a **fourth independent vote**.
            - Weight high-confidence (≥ 0.6) Kronos signals on par with strong indicator/pattern signals; treat low-confidence signals (< 0.3) as noise.
            - If Kronos disagrees with the indicator/pattern/trend consensus, lower your conviction and prefer tighter risk-reward.
            - If Kronos aligns with the consensus, bias toward the wider end of the risk-reward range (1.5-1.8).

            ---

            ### ✅ Decision Strategy

            1. Only act on **confirmed** signals — avoid emerging, speculative, or conflicting signals.
            2. Prioritize decisions where **all four reports** (Indicator, Pattern, Trend, and Kronos) **align in the same direction**.
            3. Give more weight to:
            - Recent strong momentum (e.g., MACD crossover, RSI breakout)
            - Decisive price action (e.g., breakout candle, rejection wicks, support bounce)
            - High-confidence Kronos directional forecasts.
            4. If reports disagree:
            - Choose the direction with **stronger and more recent confirmation**
            - Prefer **momentum-backed signals** over weak oscillator hints.
            - When Kronos agrees with one camp, that camp wins ties.
            5. ⚖️ If the market is in consolidation or reports are mixed:
            - Default to the **dominant trendline slope** (e.g., SHORT in descending channel), then sanity-check against the Kronos direction.
            - Do not guess direction — choose the **more defensible** side.
            6. Suggest a reasonable **risk-reward ratio** between **1.2 and 1.8**, based on current volatility, trend strength, and Kronos confidence.

            ---
            ### 🧠 Output Format in json(for system parsing):

            ```
            {{
            "forecast_horizon": "Predicting next 3 candlestick (15 minutes, 1 hour, etc.)",
            "decision": "<LONG or SHORT>",
            "justification": "<Concise, confirmed reasoning based on reports>",
            "risk_reward_ratio": "<float between 1.2 and 1.8>",
            }}

            --------
            **Technical Indicator Report**
            {indicator_report}

            **Pattern Report**
            {pattern_report}

            **Trend Report**
            {trend_report}

            **Kronos Forecast Report**
            {kronos_forecast or "(no Kronos forecast available — proceed using the three reports above)"}

            **Kronos Structured Signal**
            direction={kronos_data.get("direction", "N/A")}, magnitude_pct={kronos_data.get("magnitude_pct", "N/A")}, confidence={kronos_data.get("confidence", "N/A")}, source={kronos_data.get("source", "N/A")}

        """

        # --- LLM call for decision ---
        response = llm.invoke(prompt)

        return {
            "final_trade_decision": response.content,
            "messages": [response],
            "decision_prompt": prompt,
        }

    return trade_decision_node
