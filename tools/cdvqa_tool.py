"""
CDVQA Baseline Tool for Multi-Temporal Change Detection & VQA
=============================================================
Specialist tool for bi-temporal satellite change mask generation and change VQA.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch


class CDVQATool:
    """
    CDVQA change detection and multi-temporal VQA specialist tool.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = "cuda",
    ) -> None:
        self.weights_path = weights_path
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path: Optional[str]) -> Any:
        try:
            from models import CDVQAModel
            model = CDVQAModel.from_pretrained(weights_path)
            model.to(self.device)
            model.eval()
            return model
        except Exception:
            return None

    def load_multispectral(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load multi-spectral or RGB GeoTIFF/image array (C, H, W)."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        if p.suffix.lower() in [".tif", ".tiff"]:
            with rasterio.Env():
                with rasterio.open(p) as src:
                    arr = src.read().astype(np.float32)
                    return arr
        else:
            from PIL import Image
            img = Image.open(p).convert("RGB")
            arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            return arr

    def compute_change_mask_simple(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.25) -> np.ndarray:
        """Compute difference change mask if standalone CDVQA model weights not attached."""
        # Align shapes if needed
        c = min(img1.shape[0], img2.shape[0])
        diff = np.abs(img1[:c] - img2[:c]).mean(axis=0)
        mask = (diff > threshold).astype(np.float32)
        return mask

    def predict_change(
        self,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path],
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate change mask and multi-temporal change answer.
        """
        start_time = time.perf_counter()
        img1 = self.load_multispectral(image1_path)
        img2 = self.load_multispectral(image2_path)

        if self.model is not None:
            t1 = torch.from_numpy(img1).unsqueeze(0).to(self.device)
            t2 = torch.from_numpy(img2).unsqueeze(0).to(self.device)
            with torch.no_grad():
                change_features = self.model.encode_change(t1, t2)
                change_mask = self.model.generate_mask(change_features).squeeze().cpu().numpy()
                answer = self.model.answer_question(change_features, question or "describe changes")
        else:
            change_mask = self.compute_change_mask_simple(img1, img2)
            pct = float(change_mask.mean() * 100)
            if pct > 15:
                answer = f"Significant structural/land-cover changes detected across ~{pct:.1f}% of the observed region."
            elif pct > 2:
                answer = f"Moderate localized changes observed covering {pct:.1f}% of the area."
            else:
                answer = "Minimal or no substantial change detected between the two time steps."

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "mask": change_mask,
            "answer": answer,
            "mask_stats": {
                "total_changed_pixels": int(change_mask.sum()),
                "percentage_changed": round(float(change_mask.mean() * 100), 2),
            },
            "execution_trace": {
                "inference_time_ms": elapsed_ms,
                "device": str(self.device),
            },
        }


if __name__ == "__main__":
    print("CDVQATool defined.")
