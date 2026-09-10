"""
SatQuery AI Master Benchmark Suite (test_all_models.py)
======================================================
Unified evaluation harness across all SatQuery AI models:
- ResNet-18 (BigEarthNet v2.0 Optical-SAR Classification)
- PaliGemma-3B (VRSBench Captioning)
- Qwen2-VL-7B (CDVQA Change VQA)
- InternVL2-8B (VRSBench Grounding)
- CDVQA Baseline (Change Mask & Analysis)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root and tools/ directory are in sys.path
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "tools")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from tools.resnet_tool import ResNet18Tool
except ImportError:
    from resnet_tool import ResNet18Tool


def _find_sample_img() -> Path:
    candidates = [
        _ROOT / "scripts" / "sample_patch_rgb.png",
        _ROOT / "sample_patch_rgb.png",
        Path(__file__).resolve().parent / "sample_patch_rgb.png",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


class SatQueryAITester:
    """Master evaluator coordinating all vision-language and remote sensing tools."""

    def __init__(self, output_file: Optional[str] = None) -> None:
        if output_file is None:
            self.output_file = str(Path(__file__).resolve().parent / "benchmark_results.json")
        else:
            self.output_file = output_file
        self.results: Dict[str, Any] = {}

    def test_resnet18(self, data_root: Optional[str] = None) -> Dict[str, Any]:
        """Test ResNet-18 on BigEarthNet v2.0."""
        print("[1/5] Evaluating ResNet-18 Optical-SAR Fusion Tool...")
        tool = ResNet18Tool()

        if data_root is None:
            candidates = [
                _ROOT / "reben-training-scripts" / "scripts" / "data",
                Path(__file__).resolve().parent / "reben-training-scripts" / "scripts" / "data",
            ]
            data_dir = next((c for c in candidates if c.exists()), candidates[0])
            s1_path = data_dir / "S1"
            s2_path = data_dir / "S2"
        else:
            s1_path = Path(data_root) / "S1"
            s2_path = Path(data_root) / "S2"

        if s1_path.exists() and s2_path.exists():
            pred = tool.predict(s2_path=str(s2_path), s1_path=str(s1_path))
            return {
                "status": "success",
                "top_predictions": pred["top_k"][:3],
                "confidence": pred["confidence"],
                "sensor_prior": pred["sensor_prior"],
                "execution_trace": pred["execution_trace"],
            }
        return {"status": "skipped", "reason": f"Data not found at {data_root}"}

    def test_vit(self, data_root: Optional[str] = None) -> Dict[str, Any]:
        """Test ViT-Base on BigEarthNet v2.0."""
        print("[2/6] Evaluating ViT-Base Optical-SAR Fusion Tool...")
        tool = ResNet18Tool(model_name="BIFOLD-BigEarthNetv2-0/vit_base_patch8_224-all-v0.2.0")

        if data_root is None:
            candidates = [
                _ROOT / "reben-training-scripts" / "scripts" / "data",
                Path(__file__).resolve().parent / "reben-training-scripts" / "scripts" / "data",
            ]
            data_dir = next((c for c in candidates if c.exists()), candidates[0])
            s1_path = data_dir / "S1"
            s2_path = data_dir / "S2"
        else:
            s1_path = Path(data_root) / "S1"
            s2_path = Path(data_root) / "S2"

        if s1_path.exists() and s2_path.exists():
            pred = tool.predict(s2_path=str(s2_path), s1_path=str(s1_path))
            return {
                "status": "success",
                "top_predictions": pred["top_k"][:3],
                "confidence": pred["confidence"],
                "sensor_prior": pred["sensor_prior"],
                "execution_trace": pred["execution_trace"],
            }
        return {"status": "skipped", "reason": f"Data not found at {data_root}"}

    def test_paligemma(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Test PaliGemma on VRSBench caption split."""
        print("[2/5] Evaluating PaliGemma-3B Captioning Tool...")
        try:
            try:
                from tools.paligemma_tool import PaliGemmaTool
            except ImportError:
                from paligemma_tool import PaliGemmaTool
            tool = PaliGemmaTool()
            sample_img = _find_sample_img()
            if sample_img.exists():
                res = tool.generate_caption(sample_img, prefix="caption")
                return {
                    "status": "success",
                    "module": "paligemma_tool.PaliGemmaTool",
                    "caption": res["caption"],
                    "execution_trace": res["execution_trace"],
                }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "skipped", "reason": "Sample image not found"}

    def test_qwen2vl(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Test Qwen2-VL on CDVQA."""
        print("[3/5] Evaluating Qwen2-VL-7B Change VQA Tool...")
        try:
            try:
                from tools.qwen2vl_tool import Qwen2VLTool
            except ImportError:
                from qwen2vl_tool import Qwen2VLTool
            tool = Qwen2VLTool()
            sample_img = _find_sample_img()
            if sample_img.exists():
                res = tool.test_change_vqa(sample_img, sample_img, "Describe changes between Date 1 and Date 2.")
                return {
                    "status": "success",
                    "module": "qwen2vl_tool.Qwen2VLTool",
                    "answer": res["answer"],
                    "execution_trace": res["execution_trace"],
                }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "skipped", "reason": "Sample image not found"}

    def test_internvl2(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Test InternVL2 on VRSBench grounding split."""
        print("[4/5] Evaluating InternVL2-8B Visual Grounding Tool...")
        try:
            try:
                from tools.internvl2_tool import InternVL2Tool
            except ImportError:
                from internvl2_tool import InternVL2Tool
            tool = InternVL2Tool()
            sample_img = _find_sample_img()
            if sample_img.exists():
                res = tool.grounding(sample_img, "water body")
                return {
                    "status": "success",
                    "module": "internvl2_tool.InternVL2Tool",
                    "detected_boxes": res["boxes"],
                    "execution_trace": res["execution_trace"],
                }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "skipped", "reason": "Sample image not found"}

    def test_cdvqa(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Test CDVQA Baseline Change Detection Tool."""
        print("[5/5] Evaluating CDVQA Baseline Tool...")
        try:
            try:
                from tools.cdvqa_tool import CDVQATool
            except ImportError:
                from cdvqa_tool import CDVQATool
            tool = CDVQATool()
            sample_img = _find_sample_img()
            if sample_img.exists():
                res = tool.predict_change(sample_img, sample_img, "What changed?")
                return {
                    "status": "success",
                    "answer": res["answer"],
                    "mask_stats": res["mask_stats"],
                }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "ready"}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and export benchmark_results.json."""
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resnet18": self.test_resnet18(),
            "vit_base": self.test_vit(),
            "paligemma": self.test_paligemma(),
            "qwen2vl": self.test_qwen2vl(),
            "internvl2": self.test_internvl2(),
            "cdvqa_baseline": self.test_cdvqa(),
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nAll tests completed. Results saved to '{self.output_file}'.")
        return self.results


if __name__ == "__main__":
    tester = SatQueryAITester()
    results = tester.run_all_tests()
    print(json.dumps(results, indent=2))
