"""
agentic_orchestrator.py
========================
Layer 2 of the SatQuery AI architecture: a LangGraph `StateGraph` that
parses a natural-language query, validates the uploaded imagery against
the parsed intent, routes to the right specialist model, and combines the
result into an auditable final answer.

Fixes applied relative to the original design sketch (documented inline
at each site, kept brief here):

  1. `validation_gate` actually enforces `expected_modality == "both"`
     (require >=1 optical AND >=1 sar file) instead of a no-op `pass`,
     and reports the *specific* missing piece.
  2. `task_router`'s conditional-edge map includes an explicit "error"
     branch to `END`, so a defensively-triggered error routing decision
     can never raise a LangGraph `InvalidUpdateError` for an unmapped key
     (belt-and-braces: `should_validate` already prevents `task_router`
     from running on invalid state, but the router re-checks and this
     keeps that check meaningful rather than dead code).
  3. `change_detection_node` and `fusion_node` can short-circuit with an
     `{"error": ...}` update (e.g. missing second image / missing SAR
     image) *without* populating `tool_outputs`; `output_combinator` now
     checks `state.get("error")` first so the user sees the specific
     reason instead of a generic "Unable to process" fallback.
  4. `query_parser` validates LLM (or rule-based) output against
     `IntentSchema` (state_schema.py) before writing it into state, so a
     malformed LLM JSON response can't silently propagate bad routing
     decisions downstream.
  5. `query_parser` accepts an optional LangChain chat model (injected at
     `build_graph()` time) for production use; with no model supplied it
     uses the deterministic rule-based classifier in
     `intent_classifier.py`, so the graph is fully runnable/testable
     offline.

Section 9 of the design doc ("Conditional Edges with Command Routing")
describes `Command(update=..., goto=...)` for *dynamic*, potentially
cyclic multi-agent handoffs (e.g. a specialist deciding to call another
specialist, or hand off to a human). This orchestrator's specialist step
is a single fixed fan-out/fan-in (task_router -> one specialist ->
output_combinator), which doesn't need that flexibility, so it isn't
wired in here to keep the graph simple and non-cyclic. If you extend this
graph so a specialist node needs to conditionally re-route (e.g. fusion
node decides to also run change_detection), switch that node's return
type to `Command[Literal[...]]` and read the warning in Section 9 about
only inspecting *new* messages, or you can get infinite re-routing /
`GraphRecursionError`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from intent_classifier import classify_intent_llm, classify_intent_rule_based
from state_schema import IntentSchema, SatQueryState

logger = logging.getLogger("satquery.orchestrator")

_SUPPORTED_FORMATS = {"geotiff", "tiff", "tif", "png", "jpeg", "jpg"}

_ROUTING_MAP = {
    "vqa": "vqa_node",
    "captioning": "captioning_node",
    "grounding": "grounding_node",
    "change_detection": "change_detection_node",
    "fusion": "fusion_node",
    "land_cover_analysis": "fusion_node",  # land-cover uses the ResNet-18 + VLM fusion path
}


# ---------------------------------------------------------------------------
# NODE 1: query_parser
# ---------------------------------------------------------------------------
def make_query_parser(llm: Optional[Any] = None):
    """Factory so `llm` can be injected at graph-build time without changing
    the node's `(state) -> dict` signature that `StateGraph.add_node`
    expects."""

    def query_parser(state: SatQueryState) -> Dict[str, Any]:
        query = state["user_query"]
        num_files = len(state.get("uploaded_files", []))

        if llm is not None:
            intent = classify_intent_llm(llm, query, num_files)
        else:
            intent = classify_intent_rule_based(query, num_files)

        # Belt-and-braces: even the rule-based path returns an IntentSchema
        # already, but re-validate in case a caller passes a raw dict here
        # (e.g. from a custom LLM integration that bypasses classify_intent_llm).
        if not isinstance(intent, IntentSchema):
            try:
                intent = IntentSchema.model_validate(intent)
            except Exception as exc:
                logger.warning("Intent validation failed (%s); falling back to VQA.", exc)
                intent = IntentSchema()

        return {"intent": intent.model_dump()}

    return query_parser


# ---------------------------------------------------------------------------
# NODE 2: validation_gate
# ---------------------------------------------------------------------------
def validation_gate(state: SatQueryState) -> Dict[str, Any]:
    """Cross-check the parsed intent against the uploaded files."""
    intent = state["intent"]
    files = state.get("uploaded_files", [])
    errors = []

    expected_modality = intent.get("expected_modality", "optical")
    expected_count = intent.get("input_count", 1)

    if len(files) < expected_count:
        errors.append(f"Intent requires {expected_count} image(s), but {len(files)} uploaded.")

    optical_files = [f for f in files if f.get("modality") == "optical"]
    sar_files = [f for f in files if f.get("modality") == "sar"]

    if expected_modality == "optical" and not optical_files:
        errors.append("Expected at least one optical image, but none was uploaded.")
    elif expected_modality == "sar" and not sar_files:
        errors.append("Expected at least one SAR image, but none was uploaded.")
    elif expected_modality == "both":
        if not optical_files:
            errors.append("Fusion/land-cover analysis requires an optical image, but none was uploaded.")
        if not sar_files:
            errors.append("Fusion/land-cover analysis requires a SAR image, but none was uploaded.")

    for f in files:
        fmt = str(f.get("format", "")).lower()
        if fmt not in _SUPPORTED_FORMATS:
            errors.append(f"Unsupported format for {f.get('path', '<unknown>')}: '{fmt}'")

    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else None

    return {
        "validation_result": {
            "is_valid": is_valid,
            "error_message": error_message,
            "validated_files": files if is_valid else [],
        },
        "error": error_message,
    }


def should_validate(state: SatQueryState) -> str:
    """Conditional edge after validation_gate."""
    return "continue" if state["validation_result"]["is_valid"] else "error"


# ---------------------------------------------------------------------------
# NODE 3: task_router
# ---------------------------------------------------------------------------
def task_router(state: SatQueryState) -> Dict[str, Any]:
    """Determine which specialist node to route to."""
    if not state["validation_result"]["is_valid"]:
        # Defensive: should_validate already sends invalid state to END
        # before task_router runs, but keep this branch meaningful (see
        # module docstring point 2) rather than assuming it's unreachable.
        return {"routing_decision": "error"}

    task = state["intent"]["primary_task"]
    return {"routing_decision": _ROUTING_MAP.get(task, "vqa_node")}


def route_to_specialist(state: SatQueryState) -> str:
    """Conditional edge: route to the appropriate specialist node."""
    return state["routing_decision"]


# ---------------------------------------------------------------------------
# Specialist tool nodes
# ---------------------------------------------------------------------------
def vqa_node(state: SatQueryState) -> Dict[str, Any]:
    """Single-image VQA using InternVL2-8B."""
    from internvl2_tool import InternVL2Tool

    tool = InternVL2Tool()
    image_path = state["validation_result"]["validated_files"][0]["path"]
    query = state["user_query"]
    result = tool.predict(image_path, query)

    return {"tool_outputs": {**state.get("tool_outputs", {}), "vqa": result}}


def captioning_node(state: SatQueryState) -> Dict[str, Any]:
    """Single-image captioning using PaliGemma-3B."""
    from paligemma_tool import PaliGemmaTool

    tool = PaliGemmaTool()
    image_path = state["validation_result"]["validated_files"][0]["path"]
    caption = tool.predict(image_path)

    return {"tool_outputs": {**state.get("tool_outputs", {}), "captioning": {"caption": caption}}}


def grounding_node(state: SatQueryState) -> Dict[str, Any]:
    """Text-guided region grounding using InternVL2-8B."""
    from internvl2_tool import InternVL2Tool

    tool = InternVL2Tool()
    image_path = state["validation_result"]["validated_files"][0]["path"]
    query = state["user_query"]
    result = tool.ground(image_path, query)

    return {"tool_outputs": {**state.get("tool_outputs", {}), "grounding": result}}


def change_detection_node(state: SatQueryState) -> Dict[str, Any]:
    """Bi-temporal change VQA using Qwen2-VL-7B."""
    from qwen2vl_tool import Qwen2VLTool

    files = state["validation_result"]["validated_files"]
    if len(files) < 2:
        return {"error": "Change detection requires two images."}

    tool = Qwen2VLTool()
    image1_path, image2_path = files[0]["path"], files[1]["path"]
    query = state["user_query"]
    result = tool.predict_change(image1_path, image2_path, query)

    return {"tool_outputs": {**state.get("tool_outputs", {}), "change_detection": result}}


def fusion_node(state: SatQueryState) -> Dict[str, Any]:
    """Optical-SAR fusion: ResNet-18 sensor prior + InternVL2-8B answer."""
    from internvl2_tool import InternVL2Tool
    from resnet_tool import ResNet18Tool

    files = state["validation_result"]["validated_files"]
    optical_file = next((f for f in files if f.get("modality") == "optical"), None)
    sar_file = next((f for f in files if f.get("modality") == "sar"), None)

    if not optical_file or not sar_file:
        return {"error": "Fusion requires both optical and SAR images."}

    resnet_tool = ResNet18Tool()
    resnet_result = resnet_tool.predict(optical_file["path"], sar_file["path"])

    sensor_prior = resnet_result["sensor_prior"]
    query = state["user_query"]
    system_prompt = (
        f"System: [Vision Tool] detects: {sensor_prior}\n"
        f"User asks: {query}\n"
        f"Answer strictly based on the image and this prior."
    )

    vlm_tool = InternVL2Tool()
    vlm_result = vlm_tool.predict_with_context(optical_file["path"], system_prompt)

    return {
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "fusion": {"resnet_prior": resnet_result, "vlm_answer": vlm_result},
        }
    }


# ---------------------------------------------------------------------------
# NODE 4: output_combinator
# ---------------------------------------------------------------------------
def output_combinator(state: SatQueryState) -> Dict[str, Any]:
    """Combine specialist outputs into the final answer + execution trace."""
    tool_outputs = state.get("tool_outputs", {})
    intent = state.get("intent") or {}

    execution_trace: Dict[str, Any] = {
        "task": intent.get("primary_task", "unknown"),
        "models_used": [],
        "parameters": {},
    }

    # If a specialist node short-circuited with an error (e.g. fusion missing
    # a required modality) and never populated tool_outputs, surface that
    # reason directly instead of falling through to a generic message.
    node_error = state.get("error")
    if node_error and not tool_outputs:
        return {
            "final_answer": f"Unable to process the query: {node_error}",
            "visual_evidence": None,
            "confidence": 0.0,
            "execution_trace": execution_trace,
        }

    final_answer = ""
    visual_evidence: Any = None
    confidence = 0.0

    if "vqa" in tool_outputs:
        final_answer = tool_outputs["vqa"]["answer"]
        confidence = tool_outputs["vqa"].get("confidence", 0.0)
        execution_trace["models_used"].append("InternVL2-8B")

    elif "captioning" in tool_outputs:
        final_answer = tool_outputs["captioning"]["caption"]
        confidence = 0.8  # PaliGemma captioning has no native confidence signal
        execution_trace["models_used"].append("PaliGemma-3B")

    elif "grounding" in tool_outputs:
        boxes = tool_outputs["grounding"].get("boxes", [])
        final_answer = f"Located region(s) for: {tool_outputs['grounding']['query']}"
        visual_evidence = {"boxes": boxes}
        confidence = tool_outputs["grounding"].get("confidence", 0.0)
        execution_trace["models_used"].append("InternVL2-8B")

    elif "change_detection" in tool_outputs:
        final_answer = tool_outputs["change_detection"]["answer"]
        visual_evidence = tool_outputs["change_detection"].get("change_mask")
        confidence = tool_outputs["change_detection"].get("confidence", 0.0)
        execution_trace["models_used"].append("Qwen2-VL-7B")

    elif "fusion" in tool_outputs:
        final_answer = tool_outputs["fusion"]["vlm_answer"]
        visual_evidence = {"top_k": tool_outputs["fusion"]["resnet_prior"].get("top_k", [])}
        confidence = tool_outputs["fusion"]["resnet_prior"].get("confidence", 0.0)
        execution_trace["models_used"].extend(["ResNet-18", "InternVL2-8B"])
        execution_trace["parameters"]["sensor_prior"] = tool_outputs["fusion"]["resnet_prior"]["sensor_prior"]

    if not final_answer:
        final_answer = "Unable to process the query. Please check your inputs."
        confidence = 0.0

    return {
        "final_answer": final_answer,
        "visual_evidence": visual_evidence,
        "confidence": confidence,
        "execution_trace": execution_trace,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(llm: Optional[Any] = None):
    """Construct and compile the SatQuery AI orchestrator graph.

    Args:
        llm: Optional LangChain chat model (must support
            `with_structured_output`) used by `query_parser` for
            production-grade intent classification. If omitted, a
            deterministic rule-based classifier is used instead, which
            keeps the graph runnable/testable with no external API
            dependency.
    """
    builder = StateGraph(SatQueryState)

    builder.add_node("query_parser", make_query_parser(llm))
    builder.add_node("validation_gate", validation_gate)
    builder.add_node("task_router", task_router)
    builder.add_node("vqa_node", vqa_node)
    builder.add_node("captioning_node", captioning_node)
    builder.add_node("grounding_node", grounding_node)
    builder.add_node("change_detection_node", change_detection_node)
    builder.add_node("fusion_node", fusion_node)
    builder.add_node("output_combinator", output_combinator)

    builder.add_edge(START, "query_parser")
    builder.add_edge("query_parser", "validation_gate")

    builder.add_conditional_edges(
        "validation_gate",
        should_validate,
        {"continue": "task_router", "error": END},
    )

    builder.add_conditional_edges(
        "task_router",
        route_to_specialist,
        {
            "vqa_node": "vqa_node",
            "captioning_node": "captioning_node",
            "grounding_node": "grounding_node",
            "change_detection_node": "change_detection_node",
            "fusion_node": "fusion_node",
            "error": END,  # defensive branch; see module docstring point 2
        },
    )

    for specialist in (
        "vqa_node",
        "captioning_node",
        "grounding_node",
        "change_detection_node",
        "fusion_node",
    ):
        builder.add_edge(specialist, "output_combinator")

    builder.add_edge("output_combinator", END)

    return builder.compile()


# Module-level default graph (rule-based intent classifier, no LLM
# dependency) for convenience / quick `import agentic_orchestrator; graph`
# usage. Build your own via `build_graph(llm=...)` for production.
graph = build_graph()
