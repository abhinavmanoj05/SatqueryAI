"""
internvl2_tool.py
=================
Backend wrapper for InternVL2-8B, used for:
  - single-image VQA (`predict`)
  - text-guided region grounding (`ground`)
  - VLM answer generation with an injected sensor prior (`predict_with_context`,
    used by the fusion node)

STATUS: This is a **mock/stub backend**. It returns deterministic,
plausible-looking outputs so the orchestrator graph is runnable and
testable end-to-end without GPU access or model weights. Replace the
method bodies with real calls to your InternVL2-8B deployment (e.g. an
HTTP call to a vLLM / TGI endpoint, or a local `transformers` pipeline)
without changing the public method signatures, and the rest of the graph
(`agentic_orchestrator.py`) requires no changes.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List


def _stable_pseudo_confidence(*parts: str, low: float = 0.55, high: float = 0.97) -> float:
    """Deterministic pseudo-random confidence in [low, high], seeded by inputs.

    Used only so mock outputs are reproducible in tests instead of being
    randomly flaky.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return round(low + frac * (high - low), 3)


class InternVL2Tool:
    """Thin client for the InternVL2-8B specialist model."""

    model_name = "InternVL2-8B"

    def __init__(self, endpoint: str | None = None):
        # In production, point this at a real model-serving endpoint, e.g.:
        #   self.endpoint = endpoint or os.environ["INTERNVL2_ENDPOINT"]
        self.endpoint = endpoint or os.environ.get("INTERNVL2_ENDPOINT", "mock://internvl2")

    def predict(self, image_path: str, query: str) -> Dict[str, Any]:
        """Single-image VQA. Returns {"answer": str, "confidence": float}."""
        confidence = _stable_pseudo_confidence(image_path, query)
        answer = (
            f"[mock InternVL2-8B] Based on {os.path.basename(image_path)}, "
            f"in response to '{query}': the scene shows a plausible answer "
            f"consistent with visible land-cover features."
        )
        return {"answer": answer, "confidence": confidence}

    def ground(self, image_path: str, query: str) -> Dict[str, Any]:
        """Text-guided region grounding.

        Returns {"query": str, "boxes": [{"label", "bbox": [x0,y0,x1,y1], "score"}], "confidence": float}.
        Bounding boxes are in normalized [0, 1] image coordinates.
        """
        confidence = _stable_pseudo_confidence(image_path, query, "ground")
        boxes: List[Dict[str, Any]] = [
            {
                "label": query,
                "bbox": [0.32, 0.41, 0.58, 0.69],
                "score": confidence,
            }
        ]
        return {"query": query, "boxes": boxes, "confidence": confidence}

    def predict_with_context(self, image_path: str, system_prompt: str) -> str:
        """Answer generation with an injected system-level context/prior
        (used by the fusion node to combine the ResNet-18 sensor prior with
        a natural-language answer)."""
        confidence_hint = _stable_pseudo_confidence(image_path, system_prompt)
        return (
            f"[mock InternVL2-8B] Given the injected sensor prior, the most "
            f"likely land-cover interpretation for {os.path.basename(image_path)} "
            f"is derived from the detector's top classes "
            f"(pseudo-confidence {confidence_hint})."
        )
