# LangGraph Agent Patterns

A small collection of agent architecture patterns built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com).

It currently includes:

- **ReAct**: the model reasons, calls tools, observes results, and repeats.
- **Plan-and-execute**: the model creates a plan, executes one step, observes, updates the plan, and continues or finishes.

## ReAct

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

## Plan-and-execute

The planning graph adds a higher-level loop around execution:

1. `create_plan` breaks the objective into steps
2. `execute_step` runs the next step using the ReAct graph as the executor
3. `update_plan` revises the remaining plan or produces a final answer
4. Loop back to `execute_step` until the objective is complete

```bash
uv run planning-agent "Research what ReAct agents are and summarize the idea"
```

## Project structure

```
src/langgraph_react_agent/
├── context.py    # Configuration dataclass, reads overrides from env vars
├── graph.py      # ReAct nodes, edges, routing logic, compiled agent_graph
├── main.py       # ReAct CLI entry point
├── planning_graph.py   # plan-and-execute graph
├── planning_main.py    # planning CLI entry point
├── planning_state.py   # planning graph state
├── prompts.py    # prompt constants
├── state.py      # ReAct graph state
├── tools.py      # calculator and current_time tools
└── utils.py      # load_chat_model — provider-agnostic via init_chat_model
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
uv run planning-agent "Create a two-step plan to answer what 1234 * 5678 is, then answer"
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
