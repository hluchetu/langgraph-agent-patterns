# Architecture

## Why two state classes?

```python
@dataclass
class InputState:
    messages: Annotated[list[AnyMessage], add_messages]

@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
```

`InputState` is the public contract — what callers provide. `State` is the internal shape the graph uses, extending `InputState` with `is_last_step`, a field managed entirely by LangGraph that signals when the recursion limit is one step away.

Separating them keeps the public API clean. Callers never need to know about `is_last_step`.

## Why `is_last_step`?

Without it, hitting the recursion limit raises `GraphRecursionError` — an unhandled crash. With it, `call_model` can detect the situation and return a graceful message instead:

```python
if state.is_last_step and response.tool_calls:
    return {"messages": [AIMessage(content="Sorry, I could not find an answer...")]}
```

LangGraph sets `is_last_step = True` automatically on the second-to-last step.

## Why a custom router instead of `tools_condition`?

LangGraph ships `tools_condition` as a prebuilt router. We use a custom `route_model_output` instead because it:

- Returns `Literal["__end__", "tools"]` — a type-checked string, not a magic value
- Validates that the last message is actually an `AIMessage` and raises clearly if not
- Is explicit and readable — the logic is visible in your own codebase, not hidden in a dependency

## Why `init_chat_model` instead of `ChatOllama` directly?

```python
def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)
```

Hardcoding `ChatOllama` ties the project to one provider. `init_chat_model` resolves the right class from a `"provider/model"` string — switching from `"ollama/gemma4:e4b-mlx"` to `"anthropic/claude-opus-4-7"` requires no code changes, only a config change.

## Why `Configuration.__post_init__` reads env vars?

```python
def __post_init__(self) -> None:
    for attr in ("model", "system_prompt", "max_search_results"):
        env_val = os.environ.get(attr.upper())
        if env_val is not None:
            setattr(self, attr, type(getattr(self, attr))(env_val))
```

This gives three levels of configuration without any framework:

```
--model flag  →  MODEL env var  →  hardcoded default
```

The `type(getattr(self, attr))(env_val)` pattern coerces the env var string to the field's original type — so `MAX_SEARCH_RESULTS=5` becomes the integer `5`, not the string `"5"`.

## Why async?

`call_model` uses `await model.ainvoke(...)` because the HTTP call to Ollama is I/O-bound. Async lets the event loop handle other work while waiting for the response, rather than blocking the thread. `ToolNode` also handles async tools natively.

## Why `context_schema` on the graph?

```python
builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)
```

`context_schema` tells LangGraph the shape of the runtime context. This allows LangGraph to inject it into any node that declares `runtime: Runtime[Configuration]` in its signature — without you having to pass it manually through state. Config travels a separate channel from state, keeping them cleanly separated.
