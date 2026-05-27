from langgraph_react_agent.planning.graph import _parse_plan, planning_graph


def test_parse_plan() -> None:
    assert _parse_plan("1. Search for context\n2. Summarize findings") == [
        "Search for context",
        "Summarize findings",
    ]


def test_planning_graph_contains_expected_nodes() -> None:
    graph = planning_graph.get_graph()
    assert "create_plan" in graph.nodes
    assert "execute_step" in graph.nodes
    assert "update_plan" in graph.nodes
