from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langgraph_react_agent.context import Configuration
from langgraph_react_agent.react.graph import agent_graph
from langgraph_react_agent.routing.state import InputState, State
from langgraph_react_agent.utils import load_chat_model


CLASSIFY_PROMPT = """Classify the user's request into exactly one category.

Categories:
- math: arithmetic, calculations, or number problems
- time: questions about the current date or time
- general: anything else

Respond with only the category name, nothing else.

Request: {message}
"""


async def classify(state: State, runtime: Runtime[Configuration]) -> dict:
    model = load_chat_model(runtime.context.model)
    last_message = state.messages[-1]
    response = await model.ainvoke(
        CLASSIFY_PROMPT.format(message=last_message.content)
    )
    category = str(response.content).strip().lower()
    if category not in ("math", "time"):
        category = "general"
    return {"category": category}


async def math_agent(state: State, runtime: Runtime[Configuration]) -> dict:
    result = await agent_graph.ainvoke(
        {"messages": state.messages},
        context=Configuration(
            model=runtime.context.model,
            system_prompt=(
                "You are a math specialist. "
                "Use the calculator tool to solve arithmetic problems. "
                "Show your work clearly."
            ),
        ),
    )
    return {"messages": result["messages"]}


async def time_agent(state: State, runtime: Runtime[Configuration]) -> dict:
    result = await agent_graph.ainvoke(
        {"messages": state.messages},
        context=Configuration(
            model=runtime.context.model,
            system_prompt=(
                "You are a time specialist. "
                "Use the current_time tool to answer date and time questions."
            ),
        ),
    )
    return {"messages": result["messages"]}


async def general_agent(state: State, runtime: Runtime[Configuration]) -> dict:
    result = await agent_graph.ainvoke(
        {"messages": state.messages},
        context=runtime.context,
    )
    return {"messages": result["messages"]}


def route_request(
    state: State,
) -> Literal["math_agent", "time_agent", "general_agent"]:
    if state.category == "math":
        return "math_agent"
    if state.category == "time":
        return "time_agent"
    return "general_agent"


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node(classify)
builder.add_node(math_agent)
builder.add_node(time_agent)
builder.add_node(general_agent)

builder.add_edge("__start__", "classify")
builder.add_conditional_edges("classify", route_request)
builder.add_edge("math_agent", "__end__")
builder.add_edge("time_agent", "__end__")
builder.add_edge("general_agent", "__end__")

routing_graph = builder.compile(name="Routing Agent")
