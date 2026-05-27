from __future__ import annotations

import os
from dataclasses import dataclass, field

from langgraph_react_agent.prompts import SYSTEM_PROMPT


@dataclass
class Configuration:
    """Runtime configuration injected into each graph run via LangGraph's Runtime."""

    model: str = "ollama/gemma4:e4b-mlx"
    system_prompt: str = SYSTEM_PROMPT
    max_search_results: int = 10

    def __post_init__(self) -> None:
        for attr in ("model", "system_prompt", "max_search_results"):
            env_val = os.environ.get(attr.upper())
            if env_val is not None:
                setattr(self, attr, type(getattr(self, attr))(env_val))
