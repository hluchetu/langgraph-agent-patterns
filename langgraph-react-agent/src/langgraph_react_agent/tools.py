from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, List

from langchain_core.tools import tool


@tool
async def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if any(char not in allowed for char in expression):
        return "Error: calculator only accepts numbers and basic arithmetic."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"


@tool
async def current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().isoformat(timespec="seconds")


TOOLS: List[Callable[..., Any]] = [calculator, current_time]
