from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep


@dataclass
class InputState:
    """Public input schema — the only fields callers need to supply."""

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


@dataclass
class State(InputState):
    """Full internal state; extends InputState with LangGraph-managed fields."""

    is_last_step: IsLastStep = field(default=False)
