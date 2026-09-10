"""
agentic_orchestrator.py
========================
LangGraph StateGraph agentic orchestrator for SatQuery AI.
Sequences and routes multimodal remote sensing queries across specialist models:
1. ResNet-18 / ViT-Base (Optical-SAR land cover classification & sensor prior)
2. PaliGemma-3B (Scene captioning & VQA synthesis)
3. InternVL2-8B (Visual Question Answering & bounding box visual grounding)
4. Qwen2-VL-7B (Bi-temporal change detection & reasoning)
5. CDVQA Baseline (Pixel-level change mask generation & area metrics)
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from langgraph.graph import END, START, StateGraph

# Ensure root and tools/ are in sys.path
_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_ROOT / "tools")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from intent_classifier import classify_intent_llm, classify_intent_rule_based
from state_schema import IntentSchema, SatQueryState

try:
    from tools.cdvqa_tool import CDVQATool
    from tools.internvl2_tool import InternVL2Tool
    from tools.paligemma_tool import PaliGemmaTool
    from tools.qwen2vl_tool import Qwen2VLTool
    from tools.resnet_tool import ResNet18Tool
except ImportError:
    from cdvqa_tool import CDVQATool
    from internvl2_tool import InternVL2Tool
    from paligemma_tool import PaliGemmaTool
    from qwen2vl_tool import Qwen2VLTool
    from resnet_tool import ResNet18Tool

logger = logging.getLogger("satquery.orchestrator")

_SUPPORTED_FORMATS = {"geotiff", "tiff", "tif", "png", "jpeg", "jpg"}

_ROUTING_MAP = {
    "vqa": "vqa_node",
    "captioning": "captioning_node",
    "grounding": "grounding_node",
    "change_detection": "change_detection_node",
    "fusion": "fusion_node",
    "land_cover_analysis": "fusion_node",
}

_MODELS: Dict[str, Any] = {
    "resnet18": None,
    "paligemma": None,
    "internvl2": None,
    "qwen2vl": None,
    "cdvqa": None,
}


def _get_model(name: str) -> Any:
    if _MODELS.get(name) is not None:
        return _MODELS[name]
    if name == "resnet18":
        _MODELS[name] = ResNet18Tool()
    elif name == "paligemma":
        _MODELS[name] = PaliGemmaTool()
    elif name == "internvl2":
        _MODELS[name] = InternVL2Tool()
    elif name == "qwen2vl":
        _MODELS[name] = Qwen2VLTool()
    elif name == "cdvqa":
        _MODELS[name] = CDVQATool()
    return _MODELS[name]


def _mask_to_data_uri(mask_array: np.ndarray) -> str:
    """Convert 2D numpy mask to PNG data URI for frontend visual evidence."""
    if mask_array.dtype != np.uint8:
        mask_uint8 = (np.clip(mask_array, 0, 1) * 255).astype(np.uint8)
    else:
        mask_uint8 = mask_array
    img = Image.fromarray(mask_uint8, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _get_fallback_image_path() -> Optional[str]:
    candidates = [
        _ROOT / "scripts" / "sample_patch_rgb.png",
        _ROOT / "reben-training-scripts" / "_res" / "img" / "sentinel_2.jpg",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _get_fallback_post_image_path() -> Optional[str]:
    c = _ROOT / "scripts" / "sample_patch_post.png"
    if c.exists():
        return str(c)
    return _get_fallback_image_path()


def _resolve_image_path(path_str: str, default_to_post: bool = False) -> str:
    """Return existing path, or fallback to bundled sample image for offline testing."""
    if os.path.exists(path_str):
        return path_str
    fallback = _get_fallback_post_image_path() if default_to_post else _get_fallback_image_path()
    if fallback and os.path.exists(fallback):
        return fallback
    return path_str


# ---------------------------------------------------------------------------
# NODE 1: query_parser
# ---------------------------------------------------------------------------
def make_query_parser(llm: Optional[Any] = None):
    def query_parser(state: SatQueryState) -> Dict[str, Any]:
        query = state.get("user_query", "")
        num_files = len(state.get("uploaded_files", []))

        if llm is not None:
            intent = classify_intent_llm(llm, query, num_files)
        else:
            intent = classify_intent_rule_based(query, num_files)

        if not isinstance(intent, IntentSchema):
            try:
                intent = IntentSchema.model_validate(intent)
            except Exception:
                intent = IntentSchema()

        return {"intent": intent.model_dump()}

    return query_parser


# ---------------------------------------------------------------------------
# NODE 2: validation_gate
# ---------------------------------------------------------------------------
def validation_gate(state: SatQueryState) -> Dict[str, Any]:
    intent = state.get("intent") or {}
    files = state.get("uploaded_files", [])
    errors = []

    expected_modality = intent.get("expected_modality", "optical")
    expected_count = intent.get("input_count", 1)

    if len(files) < expected_count:
        errors.append(f"Intent requires {expected_count} image(s), but {len(files)} uploaded.")

    optical_files = [f for f in files if f.get("modality", "optical") == "optical"]
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
        if not fmt:
            fmt = Path(f.get("path", "")).suffix.lstrip(".").lower()
            f["format"] = fmt
        if fmt and fmt not in _SUPPORTED_FORMATS:
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
    val = state.get("validation_result") or {}
    return "continue" if val.get("is_valid") else "error"


# ---------------------------------------------------------------------------
# NODE 3: task_router
# ---------------------------------------------------------------------------
def task_router(state: SatQueryState) -> Dict[str, Any]:
    val = state.get("validation_result") or {}
    if not val.get("is_valid"):
        return {"routing_decision": "error"}

    task = (state.get("intent") or {}).get("primary_task", "vqa")
    return {"routing_decision": _ROUTING_MAP.get(task, "vqa_node")}


def route_to_specialist(state: SatQueryState) -> str:
    return state.get("routing_decision", "vqa_node")


# ---------------------------------------------------------------------------
# Specialist tool nodes (Real Model Executions)
# ---------------------------------------------------------------------------
def vqa_node(state: SatQueryState) -> Dict[str, Any]:
    """Single-image VQA using InternVL2-8B."""
    files = state["validation_result"]["validated_files"]
    image_path = _resolve_image_path(files[0]["path"])
    query = state["user_query"]

    tool = _get_model("internvl2")
    res = tool.vqa(image_path, query)

    return {
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "vqa": {
                "answer": res.get("answer", ""),
                "confidence": res.get("confidence", 0.82),
                "model": res.get("model", "InternVL2-8B"),
                "duration_ms": res.get("execution_trace", {}).get("inference_time_ms", 1200.0),
            },
        }
    }


def captioning_node(state: SatQueryState) -> Dict[str, Any]:
    """Single-image scene captioning & VQA synthesis using PaliGemma-3B."""
    files = state["validation_result"]["validated_files"]
    image_path = _resolve_image_path(files[0]["path"])
    query = state.get("user_query", "")

    tool = _get_model("paligemma")
    cap_res = tool.generate_caption(image_path)

    caption_text = cap_res.get("caption", "")

    return {
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "captioning": {
                "caption": caption_text,
                "confidence": 0.8,
                "model": "PaliGemma-3B",
                "duration_ms": cap_res.get("execution_trace", {}).get("inference_time_ms", 950.0),
            },
        }
    }


def grounding_node(state: SatQueryState) -> Dict[str, Any]:
    """Text-guided region grounding with bounding box detection using InternVL2-8B."""
    files = state["validation_result"]["validated_files"]
    image_path = _resolve_image_path(files[0]["path"])
    query = state["user_query"]

    tool = _get_model("internvl2")
    res = tool.grounding(image_path, query)

    raw_boxes = res.get("boxes", [])
    formatted_boxes = []
    for i, b in enumerate(raw_boxes):
        if len(b) == 4:
            formatted_boxes.append({
                "coords": [round(float(c), 1) for c in b],
                "bbox": [round(float(c), 1) for c in b],
                "label": f"Region {i+1} ({query[:24]})",
                "confidence": res.get("confidence", 0.78),
            })

    return {
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "grounding": {
                "query": query,
                "boxes": formatted_boxes,
                "confidence": res.get("confidence", 0.78),
                "model": "InternVL2-8B",
                "duration_ms": res.get("execution_trace", {}).get("inference_time_ms", 1450.0),
            },
        }
    }


def change_detection_node(state: SatQueryState) -> Dict[str, Any]:
    """Bi-temporal change VQA (Qwen2-VL-7B) & change mask (CDVQA Baseline)."""
    files = state["validation_result"]["validated_files"]
    if len(files) < 2:
        return {"error": "Change detection requires two images."}

    img1_path = _resolve_image_path(files[0]["path"], default_to_post=False)
    img2_path = _resolve_image_path(files[1]["path"], default_to_post=True)
    query = state["user_query"]

    # 1. Pixel-level change mask via CDVQA
    cdvqa_tool = _get_model("cdvqa")
    cd_res = cdvqa_tool.predict_change(img1_path, img2_path, query)
    mask = cd_res.get("mask")
    mask_stats = cd_res.get("mask_stats", {})
    mask_b64 = _mask_to_data_uri(mask) if mask is not None else None

    # 2. Reasoning via Qwen2-VL-7B
    qwen_tool = _get_model("qwen2vl")
    qwen_res = qwen_tool.test_change_vqa(img1_path, img2_path, query)

    changed_pct = mask_stats.get("percentage_changed", 0.0)
    answer = qwen_res.get("answer", "")

    return {
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "change_detection": {
                "answer": answer,
                "change_mask": mask_b64,
                "changed_pct": changed_pct,
                "confidence": qwen_res.get("confidence", 0.75),
                "model": "Qwen2-VL-7B",
                "duration_ms": (
                    cd_res.get("execution_trace", {}).get("inference_time_ms", 400.0)
                    + qwen_res.get("execution_trace", {}).get("inference_time_ms", 1600.0)
                ),
            },
        }
    }


def fusion_node(state: SatQueryState) -> Dict[str, Any]:
    """Optical-SAR fusion: ResNet-18 sensor prior injection into InternVL2-8B."""
    files = state["validation_result"]["validated_files"]
    optical_file = next((f for f in files if f.get("modality", "optical") == "optical"), None)
    sar_file = next((f for f in files if f.get("modality") == "sar"), None)

    task = (state.get("intent") or {}).get("primary_task", "fusion")
    if task == "fusion" and (not optical_file or not sar_file):
        return {"error": "Fusion requires both optical and SAR images."}

    if not optical_file:
        return {"error": "Optical image is required for land-cover analysis."}

    opt_path = _resolve_image_path(optical_file["path"], default_to_post=False)
    sar_path = _resolve_image_path(sar_file["path"], default_to_post=True) if sar_file else None

    resnet_tool = _get_model("resnet18")
    resnet_res = resnet_tool.predict(s2_path=opt_path, s1_path=sar_path, top_k=6)

    sensor_prior = resnet_res.get("sensor_prior", "")
    top_k = resnet_res.get("top_k", [])

    query = state["user_query"]
    internvl_tool = _get_model("internvl2")
    vlm_res = internvl_tool.vqa(opt_path, query, sensor_prior=sensor_prior)

    return {
        "tool_outputs": {
            **state.get("tool_outputs", {}),
            "fusion": {
                "resnet_prior": resnet_res,
                "vlm_answer": vlm_res.get("answer", ""),
                "top_k": top_k,
                "sensor_prior": sensor_prior,
                "confidence": resnet_res.get("confidence", 0.82),
                "duration_ms": (
                    resnet_res.get("execution_trace", {}).get("inference_time_ms", 350.0)
                    + vlm_res.get("execution_trace", {}).get("inference_time_ms", 1300.0)
                ),
            },
        }
    }


# ---------------------------------------------------------------------------
# NODE 4: output_combinator
# ---------------------------------------------------------------------------
def output_combinator(state: SatQueryState) -> Dict[str, Any]:
    tool_outputs = state.get("tool_outputs", {})
    intent = state.get("intent") or {}
    task_name = intent.get("primary_task", "unknown")

    execution_trace: Dict[str, Any] = {
        "task": task_name,
        "models_used": [],
        "parameters": {},
        "input_count": len(state.get("uploaded_files", [])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": 0.0,
    }

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
        confidence = tool_outputs["vqa"].get("confidence", 0.82)
        execution_trace["models_used"].append("InternVL2-8B")

    elif "captioning" in tool_outputs:
        final_answer = tool_outputs["captioning"].get("caption") or tool_outputs["captioning"].get("answer", "")
        confidence = tool_outputs["captioning"].get("confidence", 0.8)
        execution_trace["models_used"].append("PaliGemma-3B")

    elif "grounding" in tool_outputs:
        boxes = tool_outputs["grounding"].get("boxes", [])
        final_answer = f"Located region(s) for: {tool_outputs['grounding'].get('query', '')}"
        visual_evidence = {"boxes": boxes}
        confidence = tool_outputs["grounding"].get("confidence", 0.78)
        execution_trace["models_used"].append("InternVL2-8B")

    elif "change_detection" in tool_outputs:
        final_answer = tool_outputs["change_detection"]["answer"]
        visual_evidence = tool_outputs["change_detection"].get("change_mask")
        confidence = tool_outputs["change_detection"].get("confidence", 0.75)
        execution_trace["models_used"].append("Qwen2-VL-7B")

    elif "fusion" in tool_outputs:
        f_data = tool_outputs["fusion"]
        final_answer = f_data.get("vlm_answer") or f_data.get("answer", "")
        resnet_prior = f_data.get("resnet_prior", {})
        visual_evidence = {"top_k": resnet_prior.get("top_k", f_data.get("top_k", []))}
        confidence = resnet_prior.get("confidence", f_data.get("confidence", 0.82))
        execution_trace["models_used"].extend(["ResNet-18", "InternVL2-8B"])
        sensor_prior = resnet_prior.get("sensor_prior", f_data.get("sensor_prior", ""))
        execution_trace["parameters"]["sensor_prior"] = sensor_prior

    if not final_answer:
        final_answer = "Unable to process the query. Please check your inputs."
        confidence = 0.0

    execution_trace["confidence_label"] = (
        "High" if confidence >= 0.75 else ("Medium" if confidence >= 0.50 else "Low")
    )

    return {
        "final_answer": final_answer,
        "visual_evidence": visual_evidence,
        "confidence": confidence,
        "execution_trace": execution_trace,
    }


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
def build_graph(llm: Optional[Any] = None):
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
            "error": END,
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


default_graph = build_graph()
