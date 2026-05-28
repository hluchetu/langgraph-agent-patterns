from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InputState:
    """Public input schema."""

    question: str = ""


@dataclass
class State(InputState):
    """Internal state for the retrieve-grade-generate loop."""

    documents: list[str] = field(default_factory=list)
    generation: str = ""
    rewrite_count: int = 0
