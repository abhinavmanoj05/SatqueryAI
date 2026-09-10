"""
resnet_tool.py
==============
Backend wrapper for a ResNet-18 classifier used as a fast, lightweight
"sensor prior" over optical+SAR image pairs (top-k land-cover classes with
probabilities), which is then injected into a VLM prompt by the fusion node.

STATUS: Mock/stub backend -- see internvl2_tool.py for rationale. Replace
`predict()`'s body with a real ResNet-18 inference call (e.g. a local
`torchvision` model or a small inference microservice); keep the return
shape (`sensor_prior`, `top_k`, `confidence`) unchanged since
`fusion_node`/`output_combinator` in agentic_orchestrator.py read those keys.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List

_CLASSES = [
    "Urban fabric",
    "Inland waters",
    "Arable land",
    "Forest",
    "Bare soil",
    "Wetlands",
]


def _pseudo_scores(*parts: str, n: int = len(_CLASSES)) -> List[float]:
    """Deterministic pseudo-random softmax-like distribution over classes,
    seeded by inputs, so mock outputs are reproducible in tests."""
    raw = []
    for i in range(n):
        digest = hashlib.sha256(f"{'|'.join(parts)}::{i}".encode("utf-8")).hexdigest()
        raw.append(int(digest[:8], 16) / 0xFFFFFFFF)
    total = sum(raw)
    return [round(v / total, 4) for v in raw]


class ResNet18Tool:
    """Thin client for the ResNet-18 sensor-prior classifier."""

    model_name = "ResNet-18"

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or os.environ.get("RESNET18_ENDPOINT", "mock://resnet18")

    def predict(self, optical_path: str, sar_path: str) -> Dict[str, Any]:
        """Classify an optical+SAR pair into land-cover classes.

        Returns:
            {
              "sensor_prior": "<human-readable summary string>",
              "top_k": [{"class": str, "probability": float}, ...],  # sorted desc
              "confidence": float,  # top-1 probability
            }
        """
        scores = _pseudo_scores(optical_path, sar_path)
        ranked = sorted(zip(_CLASSES, scores), key=lambda kv: kv[1], reverse=True)
        top_k = [{"class": cls, "probability": prob} for cls, prob in ranked[:3]]

        summary = ", ".join(f"{c['class']} ({c['probability']:.0%})" for c in top_k)
        sensor_prior = f"ResNet-18 detects: {summary}"

        return {
            "sensor_prior": sensor_prior,
            "top_k": top_k,
            "confidence": top_k[0]["probability"],
        }
