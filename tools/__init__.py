"""
SatQuery AI - Specialist Remote Sensing Models & Tools
======================================================
This package contains dedicated wrapper tools for remote-sensing foundation models:
- ResNet18Tool: 12-channel Sentinel-1/2 land-cover classification and sensor prior generation
- PaliGemmaTool: Scene captioning and optical feature explanation
- InternVL2Tool: Visual question answering and bounding box object grounding
- Qwen2VLTool: Bi-temporal multitemporal change understanding and VQA
- CDVQATool: Bi-temporal change detection baseline & change mask generation
"""

from .resnet_tool import ResNet18Tool, CLASS_NAMES, BAND_ORDER
from .paligemma_tool import PaliGemmaTool
from .internvl2_tool import InternVL2Tool
from .qwen2vl_tool import Qwen2VLTool
from .cdvqa_tool import CDVQATool

__all__ = [
    "ResNet18Tool",
    "CLASS_NAMES",
    "BAND_ORDER",
    "PaliGemmaTool",
    "InternVL2Tool",
    "Qwen2VLTool",
    "CDVQATool",
]
