from langgraph_react_agent.react.graph import agent_graph
from langgraph_react_agent.planning.graph import planning_graph
from langgraph_react_agent.tools import calculator


def test_calculator_tool() -> None:
    import asyncio
    assert asyncio.run(calculator.ainvoke({"expression": "25 * 17"})) == "425"


def test_react_graph_contains_nodes() -> None:
    graph = agent_graph.get_graph()
    assert "call_model" in graph.nodes
    assert "tools" in graph.nodes


def test_planning_graph_contains_nodes() -> None:
    graph = planning_graph.get_graph()
    assert "create_plan" in graph.nodes
    assert "execute_step" in graph.nodes
    assert "update_plan" in graph.nodes
