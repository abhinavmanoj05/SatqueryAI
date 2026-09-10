"""
tool_registry.py
=================
LangChain @tool-decorated wrappers around real specialist remote sensing models:
- ResNet-18 (BIFOLD BigEarthNet Optical-SAR land cover classification)
- PaliGemma-3B (Satellite scene captioning & VQA synthesis)
- InternVL2-8B (VQA & text-guided bounding box visual grounding)
- Qwen2-VL-7B (Bi-temporal change understanding)
- CDVQA Baseline (Pixel-level change mask detection)

Enables uniform invocation via tool-calling agents or LangGraph ToolNode.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_ROOT / "tools")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

_internvl2: Optional[InternVL2Tool] = None
_paligemma: Optional[PaliGemmaTool] = None
_qwen2vl: Optional[Qwen2VLTool] = None
_resnet18: Optional[ResNet18Tool] = None
_cdvqa: Optional[CDVQATool] = None


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


def _get_cdvqa() -> CDVQATool:
    global _cdvqa
    if _cdvqa is None:
        _cdvqa = CDVQATool()
    return _cdvqa


@tool
def internvl2_vqa(image_path: str, query: str) -> Dict[str, Any]:
    """Run InternVL2-8B for visual question answering on a single satellite image."""
    return _get_internvl2().vqa(image_path, query)


@tool
def internvl2_ground(image_path: str, query: str) -> Dict[str, Any]:
    """Run InternVL2-8B for text-guided region grounding and bounding box localization."""
    return _get_internvl2().grounding(image_path, query)


@tool
def paligemma_caption(image_path: str) -> Dict[str, str]:
    """Run PaliGemma-3B to generate a remote sensing scene caption."""
    return _get_paligemma().generate_caption(image_path)


@tool
def qwen2vl_change(image1_path: str, image2_path: str, query: str) -> Dict[str, Any]:
    """Run Qwen2-VL-7B for bi-temporal satellite change understanding."""
    return _get_qwen2vl().test_change_vqa(image1_path, image2_path, query)


@tool
def resnet18_classify(optical_path: str, sar_path: Optional[str] = None) -> Dict[str, Any]:
    """Run ResNet-18 on an optical or optical+SAR image to obtain a BigEarthNet land-cover sensor prior."""
    return _get_resnet18().predict(optical_path, sar_path)


ALL_TOOLS: List[Any] = [
    internvl2_vqa,
    internvl2_ground,
    paligemma_caption,
    qwen2vl_change,
    resnet18_classify,
]

TOOLS_BY_NAME: Dict[str, Any] = {t.name: t for t in ALL_TOOLS}
