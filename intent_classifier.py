"""
intent_classifier.py
=====================
Turns a raw natural-language query (+ file count) into a structured
`IntentSchema` (see state_schema.py).

Two backends are provided:

1. `classify_intent_rule_based` -- a deterministic, dependency-free
   keyword classifier. This is the **default** used by
   `agentic_orchestrator.query_parser` so the whole graph runs and is unit
   -testable offline, with no LLM API key required.

2. `classify_intent_llm` -- the LLM-based path described in the original
   design (an instruction-following model classifies the query into the
   same schema). It uses `ChatModel.with_structured_output(IntentSchema)`
   so the result is validated the same way regardless of backend, and
   falls back to the rule-based classifier if the LLM call fails or
   returns something that doesn't validate.

`agentic_orchestrator.build_graph()` accepts an optional `llm` argument;
when provided, `query_parser` uses `classify_intent_llm`, otherwise it uses
the rule-based classifier. This makes the production LLM swap-in a
one-line change at graph-construction time rather than a code change deep
inside the node.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from state_schema import IntentSchema

# ---------------------------------------------------------------------------
# 1. Rule-based classifier (default, offline, deterministic)
# ---------------------------------------------------------------------------

_CHANGE_KEYWORDS = (
    "change", "before and after", "compare these two", "difference between",
    "what changed", "over time", "bi-temporal", "between these two dates",
)
_FUSION_KEYWORDS = (
    "sar", "radar", "optical and sar", "fuse", "fusion", "both images together",
    "combine the images", "together to identify",
)
_GROUNDING_KEYWORDS = (
    "highlight", "locate", "point out", "circle", "outline", "mark the",
    "bounding box", "where is", "find the",
)
_CAPTIONING_KEYWORDS = (
    "describe", "caption", "summarize the scene", "what does this image show",
    "give a description",
)
_LAND_COVER_KEYWORDS = (
    "land cover", "land-cover", "classify the land", "land use", "landuse",
)


def classify_intent_rule_based(query: str, num_files: int) -> IntentSchema:
    """Deterministic keyword-based intent classifier.

    Priority order matters: change_detection and fusion are checked before
    the more generic grounding/captioning/vqa buckets since their keyword
    sets can overlap (e.g. "compare" could loosely relate to VQA).
    """
    q = query.lower()

    def has_any(keywords: tuple) -> bool:
        return any(kw in q for kw in keywords)

    if has_any(_CHANGE_KEYWORDS) or num_files >= 2 and re.search(r"\b(then|now|previously)\b", q):
        return IntentSchema(
            primary_task="change_detection",
            input_count=2,
            requires_spatial_output=True,
            expected_modality="optical",
        )

    if has_any(_FUSION_KEYWORDS):
        return IntentSchema(
            primary_task="fusion",
            input_count=2,
            requires_spatial_output=True,
            expected_modality="both",
        )

    if has_any(_LAND_COVER_KEYWORDS):
        return IntentSchema(
            primary_task="land_cover_analysis",
            input_count=max(1, min(num_files, 2)),
            requires_spatial_output=True,
            expected_modality="both" if num_files >= 2 else "optical",
        )

    if has_any(_GROUNDING_KEYWORDS):
        return IntentSchema(
            primary_task="grounding",
            input_count=1,
            requires_spatial_output=True,
            expected_modality="optical",
        )

    if has_any(_CAPTIONING_KEYWORDS):
        return IntentSchema(
            primary_task="captioning",
            input_count=1,
            requires_spatial_output=False,
            expected_modality="optical",
        )

    # Default fallback: plain VQA.
    return IntentSchema(
        primary_task="vqa",
        input_count=max(1, min(num_files, 1)),
        requires_spatial_output=False,
        expected_modality="optical",
    )


# ---------------------------------------------------------------------------
# 2. Optional LLM-based classifier (production path)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a remote-sensing query parser.
Classify the user query into exactly one of these primary tasks:
- "vqa": single-image visual question answering
- "captioning": single-image scene description
- "grounding": text-guided region grounding (e.g., "highlight the water")
- "change_detection": bi-temporal change understanding
- "fusion": optical-SAR joint analysis
- "land_cover_analysis": general land-cover classification

Respond with a structured object matching the required schema."""


def classify_intent_llm(llm: Any, query: str, num_files: int) -> IntentSchema:
    """Classify intent using a LangChain chat model.

    `llm` must be a LangChain `BaseChatModel` (e.g. `ChatOpenAI`,
    `ChatAnthropic`, or any local model wrapper that implements
    `with_structured_output`). Falls back to the rule-based classifier if
    the call fails or returns invalid output, so a flaky/unavailable LLM
    never takes down the graph.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        structured_llm = llm.with_structured_output(IntentSchema)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}\nUploaded files: {num_files}"),
        ]
        result: Optional[IntentSchema] = structured_llm.invoke(messages)
        if isinstance(result, IntentSchema):
            return result
        # Some backends return a dict-like object instead of the pydantic
        # model instance; validate it explicitly.
        return IntentSchema.model_validate(result)
    except Exception:
        return classify_intent_rule_based(query, num_files)
