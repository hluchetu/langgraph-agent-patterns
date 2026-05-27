from langgraph_react_agent.graph import agent_graph
from langgraph_react_agent.tools import calculator


def test_calculator_tool() -> None:
    import asyncio
    assert asyncio.run(calculator.ainvoke({"expression": "25 * 17"})) == "425"


def test_graph_contains_react_nodes() -> None:
    graph = agent_graph.get_graph()
    assert "call_model" in graph.nodes
    assert "tools" in graph.nodes
