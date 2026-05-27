from __future__ import annotations

import argparse
import asyncio
import os

from langgraph_react_agent.context import Configuration
from langgraph_react_agent.planning.graph import planning_graph


async def run(objective: str, model: str) -> None:
    result = await planning_graph.ainvoke(
        {"objective": objective},
        context=Configuration(model=model),
    )
    print(result.get("response") or "No final response was produced.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("objective", nargs="*", help="Objective for the agent")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "ollama/gemma4:e4b-mlx"),
        help="Model string in 'provider/model' format",
    )
    args = parser.parse_args()

    objective = " ".join(args.objective).strip()
    if not objective:
        objective = input("Objective: ").strip()

    asyncio.run(run(objective, args.model))


if __name__ == "__main__":
    main()
