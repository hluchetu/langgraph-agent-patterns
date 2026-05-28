from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep


@dataclass
class InputState:
    """Public input schema."""

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


@dataclass
class State(InputState):
    """Internal state; extends InputState with the routing category."""

    category: str = ""
    is_last_step: IsLastStep = field(default=False)
