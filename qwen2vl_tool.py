"""
qwen2vl_tool.py
===============
Backend wrapper for Qwen2-VL-7B, used for bi-temporal (two-image) change
understanding / change VQA.

STATUS: Mock/stub backend -- see internvl2_tool.py for rationale. Replace
`predict_change()`'s body with a real Qwen2-VL-7B inference call; keep the
signature unchanged.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict


def _stable_pseudo_confidence(*parts: str, low: float = 0.55, high: float = 0.97) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return round(low + frac * (high - low), 3)


class Qwen2VLTool:
    """Thin client for the Qwen2-VL-7B change-detection model."""

    model_name = "Qwen2-VL-7B"

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or os.environ.get("QWEN2VL_ENDPOINT", "mock://qwen2vl")

    def predict_change(self, image1_path: str, image2_path: str, query: str) -> Dict[str, Any]:
        """Bi-temporal change VQA.

        Returns {"answer": str, "change_mask": {...}, "confidence": float}.
        `change_mask` here is a lightweight placeholder descriptor (e.g. a
        bounding region + change ratio) rather than a full raster mask --
        swap in a real path/URL to a rendered mask raster in production.
        """
        confidence = _stable_pseudo_confidence(image1_path, image2_path, query)
        answer = (
            f"[mock Qwen2-VL-7B] Comparing {os.path.basename(image1_path)} and "
            f"{os.path.basename(image2_path)} in response to '{query}': "
            f"noticeable change is present in the central region of the scene."
        )
        change_mask = {
            "type": "bbox_change_summary",
            "bbox": [0.30, 0.30, 0.70, 0.70],
            "change_ratio": round(0.05 + (confidence - 0.55) * 0.5, 3),
        }
        return {"answer": answer, "change_mask": change_mask, "confidence": confidence}
