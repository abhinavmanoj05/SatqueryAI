"""
paligemma_tool.py
=================
Backend wrapper for PaliGemma-3B, used for single-image scene captioning.

STATUS: Mock/stub backend -- see internvl2_tool.py for rationale. Replace
`predict()`'s body with a real PaliGemma-3B inference call; keep the
signature (`image_path: str -> str`) unchanged.
"""

from __future__ import annotations

import os


class PaliGemmaTool:
    """Thin client for the PaliGemma-3B captioning model."""

    model_name = "PaliGemma-3B"

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or os.environ.get("PALIGEMMA_ENDPOINT", "mock://paligemma")

    def predict(self, image_path: str) -> str:
        """Return a short natural-language caption for the image."""
        return (
            f"[mock PaliGemma-3B] A satellite view of "
            f"{os.path.basename(image_path)} showing a mixed landscape "
            f"with visible terrain and land-cover boundaries."
        )
