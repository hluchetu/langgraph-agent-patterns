from __future__ import annotations

import re
from typing import Literal

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langgraph_react_agent.context import Configuration
from langgraph_react_agent.retrieval.state import InputState, State
from langgraph_react_agent.utils import load_chat_model


_DOCUMENTS = [
    "ReAct agents combine reasoning and acting in a loop: the model decides the next action, executes it, observes the result, and repeats until the task is complete.",
    "Planning agents decompose a goal into steps before executing. A planner creates a task list; an executor works through each step using a ReAct loop internally.",
    "Routing in agent systems directs requests to the most appropriate handler. A classifier node sets a category, then conditional edges send the request to the matching specialist.",
    "Vector stores enable semantic search by embedding documents as high-dimensional vectors. Similarity search finds the documents closest to the query embedding.",
    "Retrieval-augmented generation (RAG) grounds model responses in retrieved documents, reducing hallucination on knowledge-intensive tasks.",
    "LangGraph represents agent workflows as state machines. Nodes update state; edges determine which node runs next. Conditional edges enable branching logic.",
    "Tool calling allows language models to invoke external functions. The model outputs a tool name and arguments; the framework executes the call and returns the result as an observation.",
]

MAX_REWRITES = 2

GRADE_PROMPT = """You are grading whether a document is relevant to a question.

Question: {question}
Document: {document}

Is this document relevant? Respond with only "yes" or "no"."""

GENERATE_PROMPT = """Answer the question using only the provided context. Be concise.

Question: {question}

Context:
{context}

Answer:"""

REWRITE_PROMPT = """Rewrite this search query to improve retrieval results.
Return only the rewritten query, nothing else.

Original query: {question}"""


def _keyword_search(query: str, documents: list[str], k: int = 3) -> list[str]:
    query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    scored = [
        (len(query_words & set(re.sub(r"[^\w\s]", "", doc.lower()).split())), doc)
        for doc in documents
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:k] if score > 0]


async def retrieve(state: State, runtime: Runtime[Configuration]) -> dict:
    documents = _keyword_search(state.question, _DOCUMENTS)
    return {"documents": documents}


async def grade_documents(state: State, runtime: Runtime[Configuration]) -> dict:
    model = load_chat_model(runtime.context.model)
    relevant = []
    for doc in state.documents:
        response = await model.ainvoke(
            GRADE_PROMPT.format(question=state.question, document=doc)
        )
        if str(response.content).strip().lower().startswith("yes"):
            relevant.append(doc)
    return {"documents": relevant}


async def generate_answer(state: State, runtime: Runtime[Configuration]) -> dict:
    model = load_chat_model(runtime.context.model)
    context = (
        "\n\n".join(state.documents)
        if state.documents
        else "No relevant documents found."
    )
    response = await model.ainvoke(
        GENERATE_PROMPT.format(question=state.question, context=context)
    )
    return {"generation": str(response.content).strip()}


async def rewrite_query(state: State, runtime: Runtime[Configuration]) -> dict:
    model = load_chat_model(runtime.context.model)
    response = await model.ainvoke(
        REWRITE_PROMPT.format(question=state.question)
    )
    return {
        "question": str(response.content).strip(),
        "rewrite_count": state.rewrite_count + 1,
    }


def route_after_grading(
    state: State,
) -> Literal["generate_answer", "rewrite_query"]:
    if state.documents:
        return "generate_answer"
    if state.rewrite_count >= MAX_REWRITES:
        return "generate_answer"
    return "rewrite_query"


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node(retrieve)
builder.add_node(grade_documents)
builder.add_node(generate_answer)
builder.add_node(rewrite_query)

builder.add_edge("__start__", "retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges("grade_documents", route_after_grading)
builder.add_edge("generate_answer", "__end__")
builder.add_edge("rewrite_query", "retrieve")

retrieval_graph = builder.compile(name="Retrieval Agent")
