# LangGraph ReAct Agent

A minimal but production-patterned **ReAct agent** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com), structured after the official [langchain-ai/react-agent](https://github.com/langchain-ai/react-agent) reference implementation.

ReAct (Reasoning + Acting) lets a model loop between thinking, calling tools, observing results, and thinking again — instead of answering in one shot.

## How it works

```
        +-----------+
        | __start__ |
        +-----------+
               *
               *
        +------------+
        | call_model |
        +------------+
          .         .
        ..           ..
       .               .
+---------+         +-------+
| __end__ |         | tools |
+---------+         +-------+
```

1. `call_model` sends the conversation + system prompt to the LLM
2. If the model returns tool calls → `tools` runs them and appends results
3. Loop back to `call_model` until the model gives a plain answer → `__end__`

## Example

```
$ uv run react-agent "what is 1234 * 5678"

The product of 1234 and 5678 is 7,006,652.
```

What actually happened under the hood:

```
HumanMessage  → "what is 1234 * 5678"
AIMessage     → (empty content, calls calculator("1234 * 5678"))
ToolMessage   → "7006652"
AIMessage     → "The product of 1234 and 5678 is 7,006,652."
```

The model never computed the math — it delegated to the tool, observed the result, then answered.

## Project structure

```
src/langgraph_react_agent/
├── state.py      # InputState (public) and State (internal + is_last_step)
├── context.py    # Configuration dataclass, reads overrides from env vars
├── prompts.py    # SYSTEM_PROMPT constant with {system_time} placeholder
├── utils.py      # load_chat_model — provider-agnostic via init_chat_model
├── tools.py      # calculator and current_time tools
├── graph.py      # nodes, edges, routing logic, compiled agent_graph
└── main.py       # CLI entry point
```

## Setup

Requires [Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull gemma4:e4b-mlx
```

Install dependencies:

```bash
uv sync
```

## Run

```bash
uv run react-agent "what is 1234 * 5678"
uv run react-agent "what time is it right now"
uv run react-agent "what is the capital of France"
```

Use a different model:

```bash
uv run react-agent "what is 99 * 99" --model ollama/llama3
```

Override via environment variable:

```bash
cp .env.example .env
# edit .env, then:
MODEL=ollama/llama3 uv run react-agent "hello"
```

Switch to any LangChain-supported provider by changing the model string:

```bash
# Anthropic (requires ANTHROPIC_API_KEY)
uv run react-agent "hello" --model anthropic/claude-opus-4-7

# OpenAI (requires OPENAI_API_KEY)
uv run react-agent "hello" --model openai/gpt-4o
```

## Run tests

```bash
uv run pytest
```

## Design notes

See [ARCHITECTURE.md](ARCHITECTURE.md) for the reasoning behind key design decisions.
