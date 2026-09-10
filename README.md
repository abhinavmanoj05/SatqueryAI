# SatQuery AI — Agentic Orchestrator (Layer 2)

A LangGraph `StateGraph` that parses a natural-language remote-sensing
query, validates uploaded imagery against the parsed intent, routes to a
specialist model, and returns a combined answer with visual evidence,
confidence, and an auditable execution trace.

All 32 tests pass (`python -m pytest tests/ -v`), and `python
run_orchestrator.py` runs a live end-to-end smoke test of the fusion path.

## Files

| File | Purpose |
|---|---|
| `state_schema.py` | Graph state (`SatQueryState`) + `IntentSchema` for validated intent |
| `intent_classifier.py` | Query → intent: rule-based (default, offline) + optional LLM path |
| `agentic_orchestrator.py` | All 4 graph nodes + 5 specialist nodes + `build_graph()` |
| `tool_registry.py` | LangChain `@tool` wrappers around each specialist backend |
| `run_orchestrator.py` | `run_satquery()` entry point with timing + trace augmentation |
| `internvl2_tool.py`, `paligemma_tool.py`, `qwen2vl_tool.py`, `resnet_tool.py` | Specialist model backends (**mock/stub** — see below) |
| `tests/test_orchestrator.py` | 32 unit + integration tests |
| `graph.mmd.md` | Mermaid diagram of the compiled graph |

## ⚠️ Mock model backends

`internvl2_tool.py`, `paligemma_tool.py`, `qwen2vl_tool.py`, and
`resnet_tool.py` are **deterministic mock implementations**, not real
model weights or inference calls — there's no InternVL2-8B, PaliGemma-3B,
Qwen2-VL-7B, or ResNet-18 running anywhere in this sandbox. They exist so
the graph is fully runnable and unit-testable offline. Each file has a
`STATUS` note in its docstring; swap the method bodies for real inference
calls (an HTTP call to a vLLM/TGI endpoint, a local `transformers`
pipeline, etc.) and keep the method signatures — the rest of the
orchestrator (`agentic_orchestrator.py`, `tool_registry.py`) needs no
changes.

## Fixes applied vs. the original design sketch

The original prompt's code had a few things that would not actually run;
these are fixed and documented inline at each site:

1. **State schema mismatch** (`state_schema.py`): the sketch subclassed
   `MessagesState` (a `TypedDict`) but assigned pydantic-style default
   values (`= None`) to fields — `TypedDict` doesn't support that and it
   raises `TypeError` at class-definition time. Fixed by declaring
   `SatQueryState` as `TypedDict(total=False)`, keeping `messages` wired
   to the same `add_messages` reducer `MessagesState` uses.
2. **`validation_gate`'s `both`-modality branch was a no-op** (`pass`) —
   it silently didn't check anything when fusion/land-cover analysis was
   requested. Fixed to require ≥1 optical **and** ≥1 SAR file, with a
   specific error message for whichever is missing.
3. **`task_router`'s conditional-edge map had no `"error"` key**, so a
   defensive `routing_decision = "error"` return would have raised
   LangGraph's `InvalidUpdateError` for an unmapped branch. Added an
   explicit `"error": END` mapping.
4. **Silent generic fallback on specialist errors**: `change_detection_node`
   / `fusion_node` can return `{"error": ...}` without populating
   `tool_outputs` (e.g. missing second image, missing SAR file), and the
   original `output_combinator` had no way to surface that — it would've
   fallen through to a generic "Unable to process the query" message.
   Fixed: `output_combinator` now checks `state.get("error")` first.
5. **Unvalidated LLM JSON output**: the sketch's `query_parser` did
   `json.loads(response.content)` with a bare fallback to a hardcoded VQA
   intent. Replaced with `IntentSchema` (pydantic) validation, plus a
   deterministic, offline rule-based classifier as the default backend
   (see `intent_classifier.py`) so the whole graph works without an LLM
   API key; a real LLM can be injected via `build_graph(llm=...)`.

## Running

```bash
pip install -r requirements.txt
python run_orchestrator.py          # smoke test (fusion path)
python -m pytest tests/ -v          # full test suite
```

## Using a real LLM for intent parsing

```python
from langchain_openai import ChatOpenAI  # or ChatAnthropic, etc.
from run_orchestrator import run_satquery

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
result = run_satquery(user_query="...", uploaded_files=[...], llm=llm)
```

`llm` just needs to support `.with_structured_output(IntentSchema)`
(all current LangChain chat model integrations do).

## Section 9 (Command routing) — not wired in

The original design's Section 9 describes `Command(update=..., goto=...)`
for dynamic multi-agent handoffs, with a warning about re-routing loops if
a node inspects *all* messages instead of only new ones. This graph's
specialist step is a single fixed fan-out/fan-in
(`task_router` → one specialist → `output_combinator`), which doesn't
need that flexibility, so it isn't used here. If you extend a specialist
node to conditionally hand off to another specialist, switch that node's
return type to `Command[Literal[...]]` and keep that warning in mind.
