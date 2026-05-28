from langgraph_react_agent.routing.graph import route_request, routing_graph
from langgraph_react_agent.routing.state import State

from langchain_core.messages import HumanMessage


def _state(category: str) -> State:
    return State(messages=[HumanMessage(content="test")], category=category)


def test_route_request_math() -> None:
    assert route_request(_state("math")) == "math_agent"


def test_route_request_time() -> None:
    assert route_request(_state("time")) == "time_agent"


def test_route_request_general() -> None:
    assert route_request(_state("general")) == "general_agent"


def test_route_request_unknown_falls_back_to_general() -> None:
    assert route_request(_state("billing")) == "general_agent"


def test_routing_graph_contains_expected_nodes() -> None:
    graph = routing_graph.get_graph()
    assert "classify" in graph.nodes
    assert "math_agent" in graph.nodes
    assert "time_agent" in graph.nodes
    assert "general_agent" in graph.nodes
