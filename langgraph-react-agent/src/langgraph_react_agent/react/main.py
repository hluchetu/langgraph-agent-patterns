from __future__ import annotations

import argparse
import asyncio
import os

from langchain_core.messages import HumanMessage

from langgraph_react_agent.context import Configuration
from langgraph_react_agent.react.graph import agent_graph


async def run(question: str, model: str) -> None:
    result = await agent_graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        context=Configuration(model=model),
    )
    print(result["messages"][-1].content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Question for the agent")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "ollama/gemma4:e4b-mlx"),
        help="Model string in 'provider/model' format",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()

    asyncio.run(run(question, args.model))


if __name__ == "__main__":
    main()
