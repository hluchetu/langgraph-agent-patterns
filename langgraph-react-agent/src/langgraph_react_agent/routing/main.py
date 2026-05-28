from __future__ import annotations

import argparse
import asyncio
import os

from langchain_core.messages import HumanMessage

from langgraph_react_agent.context import Configuration
from langgraph_react_agent.routing.graph import routing_graph


async def run(message: str, model: str) -> None:
    result = await routing_graph.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        context=Configuration(model=model),
    )
    category = result.get("category", "unknown")
    last_message = result["messages"][-1]
    print(f"[routed to: {category}]\n{last_message.content}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="*", help="Message to route")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "ollama/gemma4:e4b-mlx"),
        help="Model string in 'provider/model' format",
    )
    args = parser.parse_args()

    message = " ".join(args.message).strip()
    if not message:
        message = input("Message: ").strip()

    asyncio.run(run(message, args.model))


if __name__ == "__main__":
    main()
