from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated


@dataclass
class InputState:
    """Public input for the plan-and-execute graph."""

    objective: str = ""


@dataclass
class State(InputState):
    """Internal state for planning, execution, and replanning."""

    plan: list[str] = field(default_factory=list)
    past_steps: Annotated[list[tuple[str, str]], operator.add] = field(
        default_factory=list
    )
    response: str | None = None
