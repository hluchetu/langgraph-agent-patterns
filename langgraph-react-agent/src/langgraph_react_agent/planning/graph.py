from __future__ import annotations

import re
from typing import Dict, List, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langgraph_react_agent.context import Configuration
from langgraph_react_agent.planning.state import InputState, State
from langgraph_react_agent.react.graph import agent_graph
from langgraph_react_agent.utils import load_chat_model


PLAN_PROMPT = """Create a short plan for this objective.

Objective:
{objective}

Return only the plan as numbered steps.
"""

REPLAN_PROMPT = """You are updating a plan after one or more steps have been executed.

Objective:
{objective}

Original remaining plan:
{plan}

Completed steps and observations:
{past_steps}

If the objective is complete, respond exactly like this:
FINAL ANSWER: <answer>

Otherwise, respond exactly like this:
PLAN:
1. <next step>
2. <next step>
"""


def _parse_plan(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[\).\s-]*|[-*]\s*)", "", line).strip()
        if cleaned and not cleaned.upper().startswith("PLAN"):
            steps.append(cleaned)
    return steps


def _format_steps(steps: list[str]) -> str:
    if not steps:
        return "No remaining steps."
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))


def _format_past_steps(past_steps: list[tuple[str, str]]) -> str:
    if not past_steps:
        return "No completed steps yet."
    return "\n\n".join(
        f"Step: {step}\nObservation: {observation}"
        for step, observation in past_steps
    )


async def create_plan(
    state: State, runtime: Runtime[Configuration]
) -> Dict[str, List[str]]:
    model = load_chat_model(runtime.context.model)
    response = await model.ainvoke(
        PLAN_PROMPT.format(objective=state.objective)
    )
    return {"plan": _parse_plan(str(response.content))}


async def execute_step(
    state: State, runtime: Runtime[Configuration]
) -> Dict[str, List[tuple[str, str]]]:
    step = state.plan[0]
    task = (
        f"Objective: {state.objective}\n\n"
        f"Current step to execute: {step}\n\n"
        "Complete only this step. Use tools if they help."
    )
    result = await agent_graph.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        context=runtime.context,
    )
    observation = result["messages"][-1].content
    return {"past_steps": [(step, observation)]}


async def update_plan(state: State, runtime: Runtime[Configuration]) -> dict:
    model = load_chat_model(runtime.context.model)
    remaining_plan = state.plan[1:]
    response = await model.ainvoke(
        REPLAN_PROMPT.format(
            objective=state.objective,
            plan=_format_steps(remaining_plan),
            past_steps=_format_past_steps(state.past_steps),
        )
    )
    content = str(response.content).strip()

    if content.upper().startswith("FINAL ANSWER:"):
        return {"response": content.split(":", maxsplit=1)[1].strip(), "plan": []}

    next_plan = _parse_plan(content)
    if not next_plan and not remaining_plan:
        return {"response": state.past_steps[-1][1], "plan": []}
    return {"plan": next_plan or remaining_plan}


def route_after_update(state: State) -> Literal["execute_step", "__end__"]:
    if state.response:
        return "__end__"
    if not state.plan:
        return "__end__"
    return "execute_step"


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node(create_plan)
builder.add_node(execute_step)
builder.add_node(update_plan)

builder.add_edge("__start__", "create_plan")
builder.add_edge("create_plan", "execute_step")
builder.add_edge("execute_step", "update_plan")
builder.add_conditional_edges("update_plan", route_after_update)

planning_graph = builder.compile(name="Plan And Execute Agent")
