from __future__ import annotations

import argparse
import asyncio
import os

from langgraph_react_agent.context import Configuration
from langgraph_react_agent.retrieval.graph import retrieval_graph


async def run(question: str, model: str) -> None:
    result = await retrieval_graph.ainvoke(
        {"question": question},
        context=Configuration(model=model),
    )
    rewrites = result.get("rewrite_count", 0)
    docs_used = len(result.get("documents", []))
    print(f"[rewrites: {rewrites}, documents used: {docs_used}]\n{result['generation']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Question to answer")
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
