from langgraph_react_agent.retrieval.graph import (
    _keyword_search,
    _DOCUMENTS,
    route_after_grading,
    retrieval_graph,
)
from langgraph_react_agent.retrieval.state import State


def test_keyword_search_returns_relevant_documents() -> None:
    results = _keyword_search("ReAct agent loop", _DOCUMENTS)
    assert len(results) > 0
    assert any("ReAct" in doc for doc in results)


def test_keyword_search_returns_empty_for_no_match() -> None:
    results = _keyword_search("xyzzy quux frob", _DOCUMENTS)
    assert results == []


def test_route_after_grading_generates_when_docs_present() -> None:
    state = State(question="what is ReAct?", documents=["some relevant doc"])
    assert route_after_grading(state) == "generate_answer"


def test_route_after_grading_rewrites_when_no_docs() -> None:
    state = State(question="what is ReAct?", documents=[], rewrite_count=0)
    assert route_after_grading(state) == "rewrite_query"


def test_route_after_grading_generates_after_max_rewrites() -> None:
    state = State(question="what is ReAct?", documents=[], rewrite_count=2)
    assert route_after_grading(state) == "generate_answer"


def test_retrieval_graph_contains_expected_nodes() -> None:
    graph = retrieval_graph.get_graph()
    assert "retrieve" in graph.nodes
    assert "grade_documents" in graph.nodes
    assert "generate_answer" in graph.nodes
    assert "rewrite_query" in graph.nodes
