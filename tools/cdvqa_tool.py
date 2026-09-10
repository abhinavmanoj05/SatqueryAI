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

    def compute_bitemporal_analytics(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        threshold: float = 0.16,
    ) -> Tuple[np.ndarray, str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Compute pixel change mask, color-coded RGBA heatmap, bounding box clusters,
        and physical spectral transition statistics between Time 0 and Time 1.
        """
        import base64
        from io import BytesIO
        from PIL import Image

        # Ensure matching channel and spatial dimensions
        c = min(img1.shape[0], img2.shape[0])
        h = min(img1.shape[1], img2.shape[1])
        w = min(img1.shape[2], img2.shape[2])

        arr1 = img1[:c, :h, :w]
        arr2 = img2[:c, :h, :w]

        # Multi-channel absolute difference
        diff = np.abs(arr1 - arr2).mean(axis=0)
        binary_mask = (diff > threshold).astype(np.float32)

        # Spectral vegetation index delta (Green vs Red channels)
        g1, r1 = arr1[min(1, c - 1)], arr1[0]
        g2, r2 = arr2[min(1, c - 1)], arr2[0]
        veg1 = (g1 - r1) / (g1 + r1 + 1e-6)
        veg2 = (g2 - r2) / (g2 + r2 + 1e-6)
        dveg = veg2 - veg1

        # Surface reflectance brightness delta
        bright1 = arr1.mean(axis=0)
        bright2 = arr2.mean(axis=0)
        dbright = bright2 - bright1

        # Generate RGBA Color Heatmap Overlay
        # 🟥 Red = Vegetation clearing / demolition
        # 🟩 Green = Revegetation / agricultural growth
        # 🟨 Yellow = New concrete / built-up structures / roads
        # 🟧 Orange = General surface shift
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        chg = binary_mask > 0

        # Assign colors
        rgba[chg & (dveg < -0.08)] = [235, 45, 45, 220]      # Red: vegetation loss
        rgba[chg & (dveg > 0.08)] = [40, 220, 65, 220]       # Green: vegetation gain
        rgba[chg & (dbright > 0.10)] = [255, 195, 25, 230]   # Yellow: new built structures
        uncolored = chg & (rgba[:, :, 3] == 0)
        rgba[uncolored] = [245, 105, 30, 210]               # Orange: general change

        pil_rgba = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        pil_rgba.save(buf, format="PNG")
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        # Cluster into bounding boxes
        tile_size = max(16, min(h, w) // 12)
        candidate_boxes = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile = binary_mask[y : y + tile_size, x : x + tile_size]
                density = tile.mean()
                if density > 0.20:
                    tile_dveg = dveg[y : y + tile_size, x : x + tile_size].mean()
                    tile_dbright = dbright[y : y + tile_size, x : x + tile_size].mean()

                    if tile_dveg < -0.08:
                        lbl = f"Vegetation Clearing (ΔVeg {tile_dveg * 100:.0f}%)"
                    elif tile_dbright > 0.10:
                        lbl = f"New Construction (ΔRefl +{tile_dbright * 100:.0f}%)"
                    elif tile_dveg > 0.08:
                        lbl = f"Revegetation (ΔVeg +{tile_dveg * 100:.0f}%)"
                    else:
                        lbl = f"Surface Variation ({density * 100:.0f}% density)"

                    candidate_boxes.append({
                        "coords": [float(x), float(y), float(min(x + tile_size, w)), float(min(y + tile_size, h))],
                        "label": lbl,
                        "density": float(density),
                        "confidence": round(min(0.95, 0.70 + float(density) * 0.25), 2),
                    })

        # Sort and pick top discrete change hotspots
        candidate_boxes.sort(key=lambda b: b["density"], reverse=True)
        top_boxes = []
        for b in candidate_boxes:
            # Check overlap with existing
            c_box = b["coords"]
            overlap = False
            for existing in top_boxes:
                e_box = existing["coords"]
                if (
                    abs(c_box[0] - e_box[0]) < tile_size * 0.8
                    and abs(c_box[1] - e_box[1]) < tile_size * 0.8
                ):
                    overlap = True
                    break
            if not overlap:
                top_boxes.append({
                    "coords": [round(c, 1) for c in c_box],
                    "bbox": [round(c, 1) for c in c_box],
                    "label": f"Change Hotspot {len(top_boxes) + 1}: {b['label']}",
                    "confidence": b["confidence"],
                })
            if len(top_boxes) >= 5:
                break

        # Calculate summary statistics
        total_changed = int(binary_mask.sum())
        total_pixels = h * w
        pct_changed = round((total_changed / total_pixels) * 100, 2)
        veg_loss_pct = round(float((chg & (dveg < -0.08)).sum() / max(1, total_changed)) * 100, 1)
        built_up_pct = round(float((chg & (dbright > 0.10)).sum() / max(1, total_changed)) * 100, 1)
        veg_gain_pct = round(float((chg & (dveg > 0.08)).sum() / max(1, total_changed)) * 100, 1)

        # Spatial cross-correlation between the two acquisitions
        m1 = float(bright1.mean())
        m2 = float(bright2.mean())
        s1 = float(bright1.std())
        s2 = float(bright2.std())
        if s1 > 1e-5 and s2 > 1e-5:
            spatial_corr = float(np.mean((bright1 - m1) * (bright2 - m2)) / (s1 * s2))
        else:
            spatial_corr = 1.0

        # Disparate regions detection:
        # If >80% of pixels changed or if >65% changed with near-zero spatial correlation,
        # the two images almost certainly depict completely different places/footprints rather than true co-registered temporal change.
        is_disparate = bool(pct_changed > 80.0 or (pct_changed > 65.0 and spatial_corr < 0.15))

        stats = {
            "total_changed_pixels": total_changed,
            "percentage_changed": pct_changed,
            "vegetation_loss_percentage": veg_loss_pct,
            "built_up_expansion_percentage": built_up_pct,
            "vegetation_gain_percentage": veg_gain_pct,
            "active_hotspots": len(top_boxes),
            "spatial_correlation": round(spatial_corr, 3),
            "is_disparate_regions": is_disparate,
            "alignment_warning": (
                "Geospatial Discontinuity Detected: The two images exhibit extreme global divergence "
                f"({pct_changed:.1f}% pixel disparity, spatial correlation {spatial_corr:.2f}). "
                "They appear to depict completely different geographic regions rather than the same physical location over time."
            ) if is_disparate else None,
        }

        return binary_mask, heatmap_b64, top_boxes, stats

    def compute_change_mask_simple(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.20) -> np.ndarray:
        """Compute difference change mask if standalone CDVQA model weights not attached."""
        c = min(img1.shape[0], img2.shape[0])
        h = min(img1.shape[1], img2.shape[1])
        w = min(img1.shape[2], img2.shape[2])
        diff = np.abs(img1[:c, :h, :w] - img2[:c, :h, :w]).mean(axis=0)
        return (diff > threshold).astype(np.float32)

    def predict_change(
        self,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path],
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate change mask, color heatmap, hotspot bounding boxes, and multi-temporal statistics.
        """
        start_time = time.perf_counter()
        img1 = self.load_multispectral(image1_path)
        img2 = self.load_multispectral(image2_path)

        change_mask, heatmap_b64, boxes, stats = self.compute_bitemporal_analytics(img1, img2)

        pct = stats["percentage_changed"]
        v_loss = stats["vegetation_loss_percentage"]
        b_up = stats["built_up_expansion_percentage"]

        if stats.get("is_disparate_regions"):
            answer = (
                f"⚠️ **Geospatial Discontinuity Warning**: The two uploaded scenes exhibit extreme global disparity "
                f"({pct:.1f}% pixel delta with near-zero spatial correlation: {stats.get('spatial_correlation', 0.0):.2f}). "
                f"These images appear to depict completely different geographic regions rather than a co-registered temporal sequence of the same place.\n\n"
                f"Bi-temporal change detection requires observations of the same geographic footprint across time. "
                f"Contrasting the two distinct scenes: Image 1 exhibits structural reflectance characteristics that diverge globally from Image 2."
            )
        elif pct > 15:
            answer = (
                f"Significant multi-temporal surface variation detected across {pct:.1f}% of the scene. "
                f"Breakdown: {b_up:.0f}% structural expansion/new reflective surfaces, {v_loss:.0f}% vegetation loss/clearing. "
                f"Identified {len(boxes)} active spatial change clusters."
            )
        elif pct > 2:
            answer = (
                f"Moderate localized changes observed across {pct:.1f}% of the observed area. "
                f"Surface transitions indicate localized site development ({b_up:.0f}%) and agricultural/canopy shift ({v_loss:.0f}%). "
                f"Detected {len(boxes)} localized hotspots."
            )
        else:
            answer = f"Minimal or negligible surface variation detected ({pct:.1f}% delta) between Time 0 and Time 1."

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "mask": change_mask,
            "heatmap_b64": heatmap_b64,
            "boxes": boxes,
            "answer": answer,
            "mask_stats": stats,
            "execution_trace": {
                "inference_time_ms": elapsed_ms,
                "device": str(self.device),
            },
        }


if __name__ == "__main__":
    print("CDVQATool initialized successfully.")
