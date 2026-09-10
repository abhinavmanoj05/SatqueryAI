"""
state_schema.py
================
State schema for the SatQuery AI LangGraph orchestrator (Layer 2).

NOTE ON A FIX vs. THE ORIGINAL SPEC
------------------------------------
The original design sketch declared the state as:

    class SatQueryState(MessagesState):
        user_query: str
        intent: Optional[Dict[str, Any]] = None
        ...

That does not work at runtime: `MessagesState` is a `TypedDict`
(`langgraph.graph.MessagesState.__mro__ == (MessagesState, dict, object)`),
and Python's `TypedDict` does not support class-level default values the
way `pydantic.BaseModel` or `dataclasses` do -- assigning `= None` on a
`TypedDict` field raises a `TypeError` at class-definition time.

LangGraph's `StateGraph` is schema-agnostic (it accepts `TypedDict`,
`dataclass`, or a `pydantic.BaseModel`), but node functions in this project
return *partial* dict updates (e.g. `return {"intent": intent}`), which is
the idiomatic reducer-merge pattern for `TypedDict`/dict-based state, not
for `BaseModel` state (where you'd return a whole new model instance or use
`Annotated` reducers explicitly). To keep the partial-update pattern
described throughout the spec, this file defines `SatQueryState` as a
`TypedDict` (`total=False` for fields that are only populated partway
through the graph), while still keeping `messages` wired to LangGraph's
built-in `add_messages` reducer -- i.e., the same contract as
`MessagesState`, just declared in a way that actually type-checks and runs.

If you would prefer strict runtime validation of the *intent* payload
specifically (the one field populated by an LLM and therefore the most
likely to be malformed), see `IntentSchema` below and
`agentic_orchestrator.query_parser`, which validates LLM output against it
before writing it into state.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerated task types supported by the orchestrator.
# ---------------------------------------------------------------------------
PrimaryTask = Literal[
    "vqa",
    "captioning",
    "grounding",
    "change_detection",
    "fusion",
    "land_cover_analysis",
    "conversational",
]

Modality = Literal["optical", "sar", "both", "none"]


class IntentSchema(BaseModel):
    """
    Strict schema for the structured intent produced by `query_parser`.

    Using a pydantic model here (rather than trusting raw `json.loads`
    output from an LLM) gives us:
      - type coercion / validation with a clear error message on failure
      - a single source of truth for the four intent fields
      - an easy target for `ChatModel.with_structured_output(IntentSchema)`
        if/when a real LLM backend is wired in.
    """

    primary_task: PrimaryTask = "vqa"
    input_count: int = Field(default=1, ge=0, le=8)
    requires_spatial_output: bool = False
    expected_modality: Modality = "optical"

    @field_validator("primary_task", mode="before")
    @classmethod
    def _normalize_task(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("expected_modality", mode="before")
    @classmethod
    def _normalize_modality(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class UploadedFile(TypedDict, total=False):
    """Shape expected for each entry in `uploaded_files`."""

    path: str
    modality: str  # "optical" | "sar"
    format: str  # "geotiff" | "tiff" | "tif" | "png" | "jpeg" | "jpg"
    bands: List[str]
    metadata: Dict[str, Any]


class ValidationResult(TypedDict, total=False):
    is_valid: bool
    error_message: Optional[str]
    validated_files: List[UploadedFile]


class ExecutionTrace(TypedDict, total=False):
    task: str
    models_used: List[str]
    parameters: Dict[str, Any]
    input_count: int
    timestamp: str
    duration_ms: float
    thinking: Optional[str]
    confidence_label: Optional[str]


class SatQueryState(TypedDict, total=False):
    """
    Graph-wide state for the SatQuery AI orchestrator.

    `total=False` lets nodes return partial updates (LangGraph merges dict
    updates returned by a node into the running state); only `user_query`
    and `uploaded_files` are expected to be present at invocation time.
    """

    # Conversational history (optional; wired to LangGraph's message reducer
    # exactly like `MessagesState.messages`).
    messages: Annotated[List[AnyMessage], add_messages]

    # Input fields
    user_query: str
    uploaded_files: List[UploadedFile]
    api_key: Optional[str]

    # Parsed intent (validated against IntentSchema, stored as plain dict
    # for JSON-serializability / checkpointing)
    intent: Optional[Dict[str, Any]]
    thinking: Optional[str]
    conversational_answer: Optional[str]

    # Validation
    validation_result: Optional[ValidationResult]

    # Routing
    routing_decision: Optional[str]

    # Tool outputs (populated by specialist nodes; each node merges its own
    # key in rather than overwriting the whole dict)
    tool_outputs: Dict[str, Any]

    # Final outputs
    final_answer: Optional[str]
    visual_evidence: Optional[Any]
    confidence: Optional[float]
    execution_trace: Optional[ExecutionTrace]

    # Error handling
    error: Optional[str]
