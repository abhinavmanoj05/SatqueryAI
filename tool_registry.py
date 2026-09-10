"""
tool_registry.py
=================
LangChain `@tool`-decorated wrappers around each specialist model, so the
specialist models can be invoked uniformly (e.g. via `llm.bind_tools([...])`
in a ReAct-style agent, or via LangGraph's prebuilt `ToolNode` /
`tools_condition`) in addition to being called directly from the
deterministic specialist nodes in `agentic_orchestrator.py`.

These wrappers are the "Tool Registry" referenced in the architecture doc
(Section 5 / Section 11): a single, discoverable place that maps a callable
tool name to a specialist backend, independent of how the tool ends up
being invoked (deterministic node call vs. LLM tool-call).
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import tool

from internvl2_tool import InternVL2Tool
from paligemma_tool import PaliGemmaTool
from qwen2vl_tool import Qwen2VLTool
from resnet_tool import ResNet18Tool

# Lazily-instantiated singletons so repeated tool calls in the same process
# don't reload a model backend (or reopen a client connection) every time.
_internvl2: InternVL2Tool | None = None
_paligemma: PaliGemmaTool | None = None
_qwen2vl: Qwen2VLTool | None = None
_resnet18: ResNet18Tool | None = None


def _get_internvl2() -> InternVL2Tool:
    global _internvl2
    if _internvl2 is None:
        _internvl2 = InternVL2Tool()
    return _internvl2


def _get_paligemma() -> PaliGemmaTool:
    global _paligemma
    if _paligemma is None:
        _paligemma = PaliGemmaTool()
    return _paligemma


def _get_qwen2vl() -> Qwen2VLTool:
    global _qwen2vl
    if _qwen2vl is None:
        _qwen2vl = Qwen2VLTool()
    return _qwen2vl


def _get_resnet18() -> ResNet18Tool:
    global _resnet18
    if _resnet18 is None:
        _resnet18 = ResNet18Tool()
    return _resnet18


@tool
def internvl2_vqa(image_path: str, query: str) -> Dict[str, Any]:
    """Run InternVL2-8B for visual question answering on a single image.

    Args:
        image_path: Path to the (optical) satellite image.
        query: The user's natural-language question about the image.

    Returns:
        A dict with keys "answer" (str) and "confidence" (float, 0-1).
    """
    return _get_internvl2().predict(image_path, query)


@tool
def internvl2_ground(image_path: str, query: str) -> Dict[str, Any]:
    """Run InternVL2-8B for text-guided region grounding on a single image.

    Args:
        image_path: Path to the (optical) satellite image.
        query: A phrase describing the region to localize, e.g.
            "highlight the water body".

    Returns:
        A dict with keys "query", "boxes" (list of {label, bbox, score}),
        and "confidence" (float, 0-1).
    """
    return _get_internvl2().ground(image_path, query)


@tool
def paligemma_caption(image_path: str) -> Dict[str, str]:
    """Run PaliGemma-3B to generate a natural-language caption for a
    single satellite image.

    Args:
        image_path: Path to the (optical) satellite image.

    Returns:
        A dict with a single key "caption" (str).
    """
    return {"caption": _get_paligemma().predict(image_path)}


@tool
def qwen2vl_change(image1_path: str, image2_path: str, query: str) -> Dict[str, Any]:
    """Run Qwen2-VL-7B for bi-temporal (two-image) change understanding.

    Args:
        image1_path: Path to the earlier-dated image.
        image2_path: Path to the later-dated image.
        query: The user's natural-language question about what changed.

    Returns:
        A dict with keys "answer" (str), "change_mask" (dict placeholder
        for a rendered change raster/region), and "confidence" (float, 0-1).
    """
    return _get_qwen2vl().predict_change(image1_path, image2_path, query)


@tool
def resnet18_classify(optical_path: str, sar_path: str) -> Dict[str, Any]:
    """Run ResNet-18 on an optical+SAR image pair to obtain a fast
    land-cover "sensor prior" (top-k classes with probabilities).

    Args:
        optical_path: Path to the optical image.
        sar_path: Path to the SAR (radar) image.

    Returns:
        A dict with keys "sensor_prior" (str summary), "top_k"
        (list of {class, probability}), and "confidence" (float, 0-1).
    """
    return _get_resnet18().predict(optical_path, sar_path)


# All tools, keyed by name, for convenient `bind_tools()` / `ToolNode` use.
ALL_TOOLS: List[Any] = [
    internvl2_vqa,
    internvl2_ground,
    paligemma_caption,
    qwen2vl_change,
    resnet18_classify,
]

TOOLS_BY_NAME: Dict[str, Any] = {t.name: t for t in ALL_TOOLS}
