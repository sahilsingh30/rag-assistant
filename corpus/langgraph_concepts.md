# LangGraph Core Concepts

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains or agents in a graph-based workflow where each node represents a computation step and edges define the control flow.

LangGraph is designed for production use cases that require:
- Long-running, multi-step workflows
- Cycles and conditional branching
- Persistent state across steps
- Human-in-the-loop checkpoints

## StateGraph

The core abstraction in LangGraph is the `StateGraph`. You define:
1. A **state schema** (a TypedDict or Pydantic model) that holds all data flowing through the graph
2. **Nodes** — Python functions that receive the state and return partial updates
3. **Edges** — connections between nodes (fixed or conditional)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    question: str
    answer: str

graph = StateGraph(MyState)
graph.add_node("my_node", my_function)
graph.set_entry_point("my_node")
graph.add_edge("my_node", END)
app = graph.compile()
```

## Nodes

A node is a plain Python function. It receives the current state and returns a dictionary with only the keys it wants to update.

```python
def my_node(state: MyState) -> dict:
    # state is the full current state
    return {"answer": "computed answer"}
```

Nodes run synchronously by default. Async nodes are supported with `async def`.

## Edges

### Fixed Edges
```python
graph.add_edge("node_a", "node_b")
```
Always routes from node_a to node_b.

### Conditional Edges
```python
def route(state: MyState) -> str:
    if state["score"] > 0.5:
        return "good_path"
    return "bad_path"

graph.add_conditional_edges(
    "grading_node",
    route,
    {"good_path": "generate", "bad_path": "fallback"}
)
```
The routing function returns a string key that maps to the next node name.

## Compiling and Running

```python
app = graph.compile()

# Synchronous invocation
result = app.invoke({"question": "What is LangGraph?"})

# Streaming
for chunk in app.stream({"question": "What is LangGraph?"}):
    print(chunk)
```

## Checkpointing and State Persistence

LangGraph supports checkpointers to persist state between graph steps:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
result = app.invoke(state, config={"configurable": {"thread_id": "abc123"}})
```

## Common Patterns

### Retry Loop
Track retries in state:
```python
class State(TypedDict):
    retry_count: int

def route_after_check(state: State) -> str:
    if state["retry_count"] >= 3:
        return "give_up"
    return "try_again"
```

### Parallel Execution
Use `Send` for fan-out:
```python
from langgraph.types import Send

def fan_out(state):
    return [Send("worker", {"item": x}) for x in state["items"]]
```
