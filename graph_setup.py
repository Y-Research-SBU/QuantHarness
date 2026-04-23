from typing import Dict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent_state import IndicatorAgentState
from decision_agent import create_final_trade_decider
from graph_util import TechnicalTools
from indicator_agent import create_indicator_agent
from kronos_agent import create_kronos_agent
from pattern_agent import create_pattern_agent
from trend_agent import create_trend_agent


class SetGraph:
    def __init__(
        self,
        agent_llm: ChatOpenAI,
        graph_llm: ChatOpenAI,
        toolkit: TechnicalTools,
        # tool_nodes: Dict[str, ToolNode],
    ):
        self.agent_llm = agent_llm
        self.graph_llm = graph_llm
        self.toolkit = toolkit
        # self.tool_nodes = tool_nodes

    def set_graph(self):
        # Create analyst nodes
        agent_nodes = {}
        tool_nodes = {}
        all_agents = ["indicator", "pattern", "trend", "kronos"]

        # create nodes for indicator agent
        agent_nodes["indicator"] = create_indicator_agent(self.graph_llm, self.toolkit)
        # tool_nodes["indicator"] = self.tool_nodes["indicator"]

        # create nodes for pattern agent
        agent_nodes["pattern"] = create_pattern_agent(
            self.agent_llm, self.graph_llm, self.toolkit
        )
        # tool_nodes["pattern"] = self.tool_nodes["pattern"]

        # create nodes for trend agent
        agent_nodes["trend"] = create_trend_agent(
            self.agent_llm, self.graph_llm, self.toolkit
        )
        # tool_nodes["trend"] = self.tool_nodes["trend"]

        # create node for Kronos forecaster (5th agent — purely numerical, no LLM)
        agent_nodes["kronos"] = create_kronos_agent()

        # create nodes for decision agent
        decision_agent_node = create_final_trade_decider(self.graph_llm)

        # create graph
        graph = StateGraph(IndicatorAgentState)

        # add agent nodes to graph
        for agent_type, cur_node in agent_nodes.items():
            graph.add_node(f"{agent_type.capitalize()} Agent", cur_node)

        # add rest of the nodes
        graph.add_node("Decision Maker", decision_agent_node)

        # set start of graph
        graph.add_edge(START, "Indicator Agent")

        # add edges to graph
        for i, agent_type in enumerate(all_agents):
            current_agent = f"{agent_type.capitalize()} Agent"

            if i == len(all_agents) - 1:
                graph.add_edge(current_agent, "Decision Maker")
            else:

                next_agent = f"{all_agents[i + 1].capitalize()} Agent"
                graph.add_edge(current_agent, next_agent)

        # Decision Maker Process
        graph.add_edge("Decision Maker", END)

        return graph.compile()
