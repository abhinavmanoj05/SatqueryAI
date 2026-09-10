"""
SatQuery AI - Interactive Specialist Model Evaluation UI
========================================================
Interactive Gradio Web UI integrating all specialist models in the SatQuery AI pipeline:
1. ResNet-18 (Optical-SAR 12-band fusion classification & sensor prior generator)
2. PaliGemma-3B (Satellite image captioning & prompt-injected VQA)
3. InternVL2-8B (VQA & visual grounding with bounding box overlays)
4. Qwen2-VL-7B (Bi-temporal satellite change understanding & VQA)
5. CDVQA Baseline (Dual-temporal change mask generation & area statistics)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

import rasterio
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root and tools/ directory are in sys.path
_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_ROOT / "tools")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import specialist model tools
try:
    from tools.resnet_tool import ResNet18Tool
except ImportError:
    from resnet_tool import ResNet18Tool

# Global lazy model store
MODELS: Dict[str, Any] = {
    "ResNet-18": None,
    "ViT-Base": None,
    "ResNet-50": None,
    "PaliGemma": None,
    "InternVL2": None,
    "Qwen2-VL": None,
    "CDVQA": None,
}


def get_device() -> str:
    """Return available device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(name: str) -> Any:
    """Lazily load models upon first user request."""
    if MODELS.get(name) is not None:
        return MODELS[name]

    device = get_device()

    if name == "ResNet-18":
        MODELS[name] = ResNet18Tool(model_name="BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0", device=device)
    elif name == "ViT-Base":
        MODELS[name] = ResNet18Tool(model_name="BIFOLD-BigEarthNetv2-0/vit_base_patch8_224-all-v0.2.0", device=device)
    elif name == "ResNet-50":
        MODELS[name] = ResNet18Tool(model_name="BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0", device=device)
    elif name == "PaliGemma":
        try:
            from tools.paligemma_tool import PaliGemmaTool
        except ImportError:
            from paligemma_tool import PaliGemmaTool
        MODELS[name] = PaliGemmaTool(device=device)
    elif name == "InternVL2":
        try:
            from tools.internvl2_tool import InternVL2Tool
        except ImportError:
            from internvl2_tool import InternVL2Tool
        MODELS[name] = InternVL2Tool(device=device)
    elif name == "Qwen2-VL":
        try:
            from tools.qwen2vl_tool import Qwen2VLTool
        except ImportError:
            from qwen2vl_tool import Qwen2VLTool
        MODELS[name] = Qwen2VLTool(device=device)
    elif name == "CDVQA":
        try:
            from tools.cdvqa_tool import CDVQATool
        except ImportError:
            from cdvqa_tool import CDVQATool
        MODELS[name] = CDVQATool(device=device)

    return MODELS[name]


def _extract_file_path(file_obj: Any) -> Optional[Any]:
    """Safely extract local filesystem path from Gradio file input object."""
    if file_obj is None:
        return None
    if isinstance(file_obj, (list, tuple)):
        extracted = [_extract_file_path(f) for f in file_obj]
        valid = [p for p in extracted if p is not None]
        return valid if valid else None
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def plot_probabilities(top_k_items: List[Dict[str, Any]]) -> plt.Figure:
    """Create a styled horizontal bar chart for class probabilities."""
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(top_k_items) * 0.45)), dpi=120)

    classes = [item["class"] for item in reversed(top_k_items)]
    probs = [item["probability"] * 100 for item in reversed(top_k_items)]

    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(classes)))
    bars = ax.barh(classes, probs, color=colors, height=0.65)

    ax.set_xlim(0, max(100, max(probs) * 1.15 if probs else 100))
    ax.set_xlabel("Probability (%)", fontsize=11, fontweight="bold")
    ax.set_title("Land Cover Class Probabilities", fontsize=13, fontweight="bold", pad=12)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="semibold",
            color="#222222",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1. Optical-SAR Land Cover Classification
# ---------------------------------------------------------------------------
def predict_resnet18(
    s2_file: Any = None,
    s1_file: Any = None,
    top_k: int = 10,
    model_name: str = "ResNet-18",
) -> Tuple[str, Optional[plt.Figure], str]:
    """Execute optical-SAR classification with selectable specialist model."""
    try:
        s2_path = _extract_file_path(s2_file)
        s1_path = _extract_file_path(s1_file)

        tool = get_model(model_name)
        result = tool.predict(s2_path=s2_path, s1_path=s1_path, top_k=int(top_k))

        summary_text = (
            f"### 🛰️ Sensor Prior Output ({model_name})\n\n"
            f"> **{result['sensor_prior']}**\n\n"
            f"**Max Confidence:** `{result['confidence'] * 100:.2f}%`\n"
            f"**Model ID:** `{result['execution_trace']['model']}`\n"
            f"**Inference Time:** `{result['execution_trace']['inference_time_ms']} ms`\n"
            f"**Execution Device:** `{result['execution_trace']['device']}`"
        )

        fig = plot_probabilities(result["top_k"])
        json_trace = f"```json\n{torch.tensor(result['probabilities']).numpy().round(4).tolist()}\n```"

        return summary_text, fig, json_trace

    except Exception as exc:
        traceback.print_exc()
        return f"❌ Execution Failed: {str(exc)}", None, ""


# ---------------------------------------------------------------------------
# 2. PaliGemma Inference
# ---------------------------------------------------------------------------
def predict_paligemma(
    image_file: Any,
    prefix_prompt: str,
    sensor_prior: str,
    api_key: Optional[str] = None,
) -> str:
    """Execute PaliGemma image captioning or VQA."""
    try:
        img_path = _extract_file_path(image_file)
        if not img_path:
            return "Error: Please upload an image."

        tool = get_model("PaliGemma")
        key_clean = api_key.strip() if api_key and api_key.strip() else None
        res = tool.generate_caption(
            image=img_path,
            prefix=prefix_prompt.strip() or "caption",
            sensor_prior=sensor_prior.strip() if sensor_prior.strip() else None,
            api_key=key_clean,
        )

        output = (
            f"### 📝 Generated Caption\n\n"
            f"{res['caption']}\n\n"
            f"---\n"
            f"**Engine / Model:** `{res['execution_trace']['model']}`\n"
            f"**Prompt Used:** `{res['prompt']}`\n"
            f"**Inference Time:** `{res['execution_trace']['inference_time_ms']} ms`"
        )
        return output

    except Exception as exc:
        traceback.print_exc()
        return f"❌ Execution Failed: {str(exc)}"


# ---------------------------------------------------------------------------
# 3. InternVL2 Inference
# ---------------------------------------------------------------------------
def predict_internvl2(
    image_file: Any,
    query: str,
    task_mode: str,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[Image.Image]]:
    """Execute InternVL2 for VQA or Grounding."""
    try:
        img_path = _extract_file_path(image_file)
        if not img_path:
            return "Error: Please upload an image.", None

        tool = get_model("InternVL2")
        pil_img = Image.open(img_path).convert("RGB")
        key_clean = api_key.strip() if api_key and api_key.strip() else None

        if "Grounding" in task_mode:
            res = tool.grounding(img_path, query.strip() or "detect objects", api_key=key_clean)
            boxes = res["boxes"]

            # Draw bounding boxes on PIL image
            annotated = pil_img.copy()
            draw = ImageDraw.Draw(annotated)
            w, h = annotated.size

            for box in boxes:
                # Handle normalized vs absolute coords
                x1, y1, x2, y2 = box
                if x2 <= 1.0 and y2 <= 1.0:
                    x1, x2 = x1 * w, x2 * w
                    y1, y2 = y1 * h, y2 * h
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1 + 3, max(0, y1 - 12)), query, fill="red")

            text_out = (
                f"### 🎯 Visual Grounding Results\n\n"
                f"**Query:** `{query}`\n"
                f"**Engine / Model:** `{res['execution_trace']['model']}`\n"
                f"**Detected Bounding Boxes:** `{len(boxes)}` found\n"
                f"**Coordinates:** `{boxes}`\n"
                f"**Inference Time:** `{res['execution_trace']['inference_time_ms']} ms`"
            )
            return text_out, annotated
        else:
            res = tool.vqa(img_path, query.strip() or "Describe this satellite image.", api_key=key_clean)
            text_out = (
                f"### 💬 VQA Answer\n\n"
                f"{res['answer']}\n\n"
                f"**Engine / Model:** `{res['execution_trace']['model']}`\n"
                f"**Question:** `{query}`\n"
                f"**Inference Time:** `{res['execution_trace']['inference_time_ms']} ms`"
            )
            return text_out, pil_img

    except Exception as exc:
        traceback.print_exc()
        return f"❌ Execution Failed: {str(exc)}", None


# ---------------------------------------------------------------------------
# 4. Qwen2-VL Inference
# ---------------------------------------------------------------------------
def predict_qwen2vl(
    img_pre_file: Any,
    img_post_file: Any,
    query: str,
    sensor_prior: str,
    api_key: Optional[str] = None,
) -> str:
    """Execute Qwen2-VL bi-temporal change analysis."""
    try:
        pre_path = _extract_file_path(img_pre_file)
        post_path = _extract_file_path(img_post_file)

        if not pre_path or not post_path:
            return "Error: Please upload both Pre-event (Date 1) and Post-event (Date 2) images."

        tool = get_model("Qwen2-VL")
        key_clean = api_key.strip() if api_key and api_key.strip() else None
        res = tool.test_change_vqa(
            image_pre=pre_path,
            image_post=post_path,
            question=query.strip() or "Describe the visual and structural changes between Date 1 and Date 2.",
            sensor_prior=sensor_prior.strip() if sensor_prior.strip() else None,
            api_key=key_clean,
        )

        output = (
            f"### 🔄 Change Analysis & VQA\n\n"
            f"{res['answer']}\n\n"
            f"---\n"
            f"**Engine / Model:** `{res['execution_trace']['model']}`\n"
            f"**Question:** `{query}`\n"
            f"**Inference Time:** `{res['execution_trace']['inference_time_ms']} ms`"
        )
        return output

    except Exception as exc:
        traceback.print_exc()
        return f"❌ Execution Failed: {str(exc)}"


# ---------------------------------------------------------------------------
# 5. CDVQA Baseline Inference
# ---------------------------------------------------------------------------
def predict_cdvqa(
    img1_file: Any,
    img2_file: Any,
    query: str,
) -> Tuple[str, Optional[Image.Image]]:
    """Execute CDVQA dual-temporal change detection & mask generation."""
    try:
        p1 = _extract_file_path(img1_file)
        p2 = _extract_file_path(img2_file)

        if not p1 or not p2:
            return "Error: Please upload both Time 1 and Time 2 images.", None

        tool = get_model("CDVQA")
        res = tool.predict_change(p1, p2, query.strip() or "describe changes")

        mask = res["mask"]
        # Convert mask to colored RGBA overlay
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[..., 0] = 255  # Red channel
        mask_rgba[..., 3] = (mask * 180).astype(np.uint8)  # Alpha transparency
        mask_img = Image.fromarray(mask_rgba, mode="RGBA")

        stats = res["mask_stats"]
        text_out = (
            f"### 🔍 Multi-Temporal Change Detection\n\n"
            f"**Answer:** {res['answer']}\n\n"
            f"**Total Changed Pixels:** `{stats['total_changed_pixels']}`\n"
            f"**Percentage Area Changed:** `{stats['percentage_changed']}%`\n"
            f"**Inference Time:** `{res['execution_trace']['inference_time_ms']} ms`"
        )
        return text_out, mask_img

    except Exception as exc:
        traceback.print_exc()
        return f"❌ Execution Failed: {str(exc)}", None


# ---------------------------------------------------------------------------
# 6. Agentic Orchestrator (Requirement 5 from Requirement.pdf)
# ---------------------------------------------------------------------------
def run_agentic_orchestrator(
    img_primary: Any,
    img_secondary: Any,
    user_query: str,
    prior_model: str = "ResNet-18",
    api_key: Optional[str] = None,
) -> Tuple[str, str, Optional[Image.Image], str]:
    """
    Autonomous multi-model agent that evaluates satellite inputs & user query,
    determines the optimal toolchain, executes tools sequentially, passes
    sensor priors forward, and synthesizes a comprehensive final response.
    """
    try:
        start_time = time.perf_counter()
        p1 = _extract_file_path(img_primary)
        p2 = _extract_file_path(img_secondary)
        query = user_query.strip() if user_query else "Analyze the land cover, features, and any detectable changes."
        key_clean = api_key.strip() if api_key and api_key.strip() else None

        if not p1:
            sample_p = Path(__file__).resolve().parent / "sample_patch_rgb.png"
            if sample_p.exists():
                p1 = str(sample_p)
            else:
                return "❌ Please upload at least one satellite image.", "", None, ""

        plan_steps = []
        tools_executed = []
        pil_p1 = Image.open(p1).convert("RGB")
        annotated_image = pil_p1.copy()
        sensor_prior = ""

        # Step 1: Decide if Land Cover / SAR Prior is needed
        plan_steps.append(f"1. **Analyze Spectral Characteristics & Prior**: Invoke `{prior_model}` Optical-SAR classifier.")
        prior_tool = get_model(prior_model)
        try:
            res_pred = prior_tool.predict(s2_path=p1, top_k=3)
            sensor_prior = res_pred["sensor_prior"]
            tools_executed.append({
                "step": 1,
                "tool": f"{prior_model} Optical-SAR Classifier",
                "purpose": "Generate land cover priors & confidence scores",
                "output": sensor_prior,
                "latency_ms": res_pred["execution_trace"]["inference_time_ms"]
            })
        except Exception as e:
            tools_executed.append({
                "step": 1,
                "tool": prior_model,
                "error": str(e)
            })

        # Step 2: Multi-Temporal / Bi-Temporal Reasoning (if second image provided or query asks about change)
        change_info = ""
        is_bitemporal = (p2 is not None) or any(k in query.lower() for k in ["change", "difference", "delta", "temporal", "date 1", "time"])
        if is_bitemporal:
            target_p2 = p2 if p2 else p1
            plan_steps.append("2. **Bi-Temporal Analysis**: Invoke `CDVQA Tool` (change mask) & `Qwen2-VL` (temporal reasoning).")
            
            cdvqa_tool = get_model("CDVQA")
            cd_res = cdvqa_tool.predict_change(p1, target_p2, query)
            mask = cd_res["mask"]
            stats = cd_res["mask_stats"]
            
            qwen_tool = get_model("Qwen2-VL")
            qwen_res = qwen_tool.test_change_vqa(p1, target_p2, query, sensor_prior=sensor_prior, api_key=key_clean)
            change_info = qwen_res["answer"]
            
            if stats["total_changed_pixels"] > 0:
                mask_uint8 = (mask * 255).astype(np.uint8)
                w, h = annotated_image.size
                mask_resized = Image.fromarray(mask_uint8).resize((w, h), Image.NEAREST)
                overlay = Image.new("RGBA", annotated_image.size, (255, 0, 0, 100))
                annotated_image = Image.composite(overlay, annotated_image.convert("RGBA"), mask_resized).convert("RGB")

            tools_executed.append({
                "step": 2,
                "tool": "CDVQA Baseline + Qwen2-VL-7B",
                "purpose": "Dual-temporal pixel delta mask & change narrative",
                "output": f"Changed Area: {stats['percentage_changed']}%. {change_info}",
                "latency_ms": cd_res["execution_trace"]["inference_time_ms"] + qwen_res["execution_trace"]["inference_time_ms"]
            })

        # Step 3: Semantic Grounding / Target Localization with InternVL2
        grounding_info = ""
        needs_grounding = any(k in query.lower() for k in ["ground", "box", "detect", "locate", "where", "find", "water", "forest", "vegetation", "agriculture", "crop"])
        if needs_grounding:
            plan_steps.append("3. **Visual Grounding**: Invoke `InternVL2-8B` to extract localized bounding boxes.")
            internvl_tool = get_model("InternVL2")
            g_target = "water body" if "water" in query.lower() else ("forest" if "forest" in query.lower() else "vegetation / parcel")
            g_res = internvl_tool.grounding(p1, g_target, api_key=key_clean)
            boxes = g_res["boxes"]
            draw = ImageDraw.Draw(annotated_image)
            w, h = annotated_image.size
            for box in boxes:
                x1, y1, x2, y2 = box
                if x2 <= 1.0 and y2 <= 1.0:
                    x1, x2 = x1 * w, x2 * w
                    y1, y2 = y1 * h, y2 * h
                draw.rectangle([x1, y1, x2, y2], outline="#00FF66", width=4)
                draw.text((x1 + 4, max(0, y1 - 14)), f"Target: {g_target}", fill="#00FF66")
            grounding_info = f"Localized {len(boxes)} regions for query target '{g_target}' at coordinates: `{boxes}`"
            tools_executed.append({
                "step": 3,
                "tool": "InternVL2-8B Visual Grounding",
                "purpose": f"Localize '{g_target}' coordinates",
                "output": grounding_info,
                "latency_ms": g_res["execution_trace"]["inference_time_ms"]
            })

        # Step 4: Captioning & High-level Semantic VQA with PaliGemma
        plan_steps.append(f"4. **Semantic Synthesis**: Invoke `PaliGemma-3B` with {prior_model} sensor prior injection.")
        pali_tool = get_model("PaliGemma")
        pali_res = pali_tool.vqa(p1, query, sensor_prior=sensor_prior, api_key=key_clean)
        caption_res = pali_tool.generate_caption(p1, sensor_prior=sensor_prior, api_key=key_clean)
        tools_executed.append({
            "step": 4,
            "tool": "PaliGemma-3B VLM",
            "purpose": "Prompt-injected satellite captioning & VQA synthesis",
            "output": pali_res["answer"],
            "latency_ms": pali_res["execution_trace"]["inference_time_ms"]
        })

        # Synthesize final executive answer
        total_time = round((time.perf_counter() - start_time) * 1000, 2)
        exec_plan_md = "### 🧠 Agent Execution Plan & Routing\n\n" + "\n".join(plan_steps) + "\n\n"
        
        synthesis_md = (
            f"### 📋 Executive Intelligence Report\n\n"
            f"**Query:** *\"{query}\"*\n\n"
            f"#### 1. Multi-Sensor Land Cover Identification ({prior_model})\n"
            f"> {sensor_prior}\n\n"
            f"#### 2. Scene Description & Visual Question Answering (PaliGemma-3B)\n"
            f"- **Observation:** {caption_res['caption']}\n"
            f"- **VQA Response:** {pali_res['answer']}\n\n"
        )
        if is_bitemporal:
            synthesis_md += (
                f"#### 3. Bi-Temporal Change Detection (CDVQA + Qwen2-VL-7B)\n"
                f"- **Change Assessment:** {change_info}\n\n"
            )
        if grounding_info:
            synthesis_md += (
                f"#### 4. Spatial Localization & Grounding (InternVL2-8B)\n"
                f"- **Grounding Status:** {grounding_info}\n\n"
            )

        synthesis_md += (
            f"---\n"
            f"⚡ **Agent Workflow Completed in {total_time} ms** across `{len(tools_executed)}` specialist tools."
        )

        trace_json = json.dumps(tools_executed, indent=2)

        return synthesis_md, exec_plan_md, annotated_image, f"```json\n{trace_json}\n```"

    except Exception as exc:
        traceback.print_exc()
        return f"❌ Agent Orchestration Failed: {str(exc)}", "", None, ""


# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
    )

    with gr.Blocks(title="SatQuery AI - Specialist Model Tester", theme=theme) as demo:
        gr.Markdown(
            """
            # 🛰️ SatQuery AI - Specialist Model Evaluation Suite
            Interactive multi-modal workbench for Optical-SAR classification, Captioning, VQA, Visual Grounding, and Change Detection.
            """
        )

        with gr.Accordion("🔑 Live Vision-Language API Setup (Optional: Free 72B / Flash Inference)", open=False):
            gr.Markdown(
                "💡 **Connect to Live Free VLMs:** Paste your free **OpenRouter API key** (`sk-or-...`) or **Google AI Studio Gemini API key** (`AIza...`) "
                "to enable real-time 72B parameter VQA, Grounding, and Captioning. Leave blank to run in fast local simulation mode."
            )
            api_key_input = gr.Textbox(
                label="Free API Key (OpenRouter or Google Gemini)",
                placeholder="sk-or-... or AIza... (optional)",
                type="password",
            )

        with gr.Tabs():

            # -------------------------------------------------------------------
            # TAB 1: Optical-SAR Land Cover Specialists
            # -------------------------------------------------------------------
            with gr.TabItem("📡 1. Optical-SAR Land Cover Specialists"):
                gr.Markdown(
                    "### 12-Channel Sentinel-1 (SAR) + Sentinel-2 (Optical) Land Cover Classification\n"
                    "Select an Optical-SAR classifier to generate frozen sensor priors for VLM prompt injection.\n\n"
                    "💡 **Available Specialists:**\n"
                    "- **ResNet-18**: Native Optical-SAR 12-channel convolutional prior generator.\n"
                    "- **ViT-Base**: Vision Transformer (`vit_base_patch8_224`) capturing multi-head attention across patches.\n"
                    "- **ResNet-50**: Deep convolutional optical model (`resnet50-s2-v0.2.0`)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        sar_model_choice = gr.Radio(
                            choices=["ResNet-18", "ViT-Base", "ResNet-50"],
                            value="ResNet-18",
                            label="Select Specialist Model Architecture",
                        )
                        s2_file = gr.File(
                            label="Sentinel-2 Optical GeoTIFF (.tif / .tiff / image)",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                        )
                        s1_file = gr.File(
                            label="Sentinel-1 Radar SAR GeoTIFF (.tif / .tiff)",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            file_count="multiple",
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=19,
                            value=10,
                            step=1,
                            label="Top-K Land Cover Classes",
                        )
                        with gr.Row():
                            resnet_btn = gr.Button("🚀 Run Classification Prior Generator", variant="primary")
                            load_sample_btn = gr.Button("⚡ Auto-Load Bundled Optical + SAR Patch", variant="secondary")

                    with gr.Column(scale=1):
                        resnet_text = gr.Markdown(label="Sensor Prior Output")
                        resnet_plot = gr.Plot(label="Top-K Probabilities")
                        resnet_raw = gr.Markdown(label="Raw Probabilities Vector")

                resnet_btn.click(
                    fn=predict_resnet18,
                    inputs=[s2_file, s1_file, top_k_slider, sar_model_choice],
                    outputs=[resnet_text, resnet_plot, resnet_raw],
                )
                load_sample_btn.click(
                    fn=lambda k, m: predict_resnet18(None, None, k, m),
                    inputs=[top_k_slider, sar_model_choice],
                    outputs=[resnet_text, resnet_plot, resnet_raw],
                )

            # -------------------------------------------------------------------
            # TAB 2: PaliGemma-3B
            # -------------------------------------------------------------------
            with gr.TabItem("📝 2. PaliGemma-3B (Captioning & VQA)"):
                gr.Markdown("### Satellite Image Captioning & Prompt-Injected VQA")
                with gr.Row():
                    with gr.Column(scale=1):
                        paligemma_img = gr.Image(label="Satellite Image", type="filepath")
                        paligemma_prefix = gr.Textbox(
                            label="Task Prompt / Prefix",
                            value="caption",
                            placeholder="e.g. caption, or answer en what is in this image?",
                        )
                        paligemma_prior = gr.Textbox(
                            label="Optional Sensor Prior (from ResNet-18 / ViT-Base)",
                            placeholder="e.g. Broad-leaved forest (67%), Inland wetlands (37%)",
                        )
                        paligemma_btn = gr.Button("🚀 Generate Caption / Answer", variant="primary")

                    with gr.Column(scale=1):
                        paligemma_out = gr.Markdown(label="Output")

                paligemma_btn.click(
                    fn=predict_paligemma,
                    inputs=[paligemma_img, paligemma_prefix, paligemma_prior, api_key_input],
                    outputs=[paligemma_out],
                )

            # -------------------------------------------------------------------
            # TAB 3: InternVL2-8B
            # -------------------------------------------------------------------
            with gr.TabItem("🎯 3. InternVL2-8B (VQA & Grounding)"):
                gr.Markdown("### Remote Sensing VQA & Object Visual Grounding")
                with gr.Row():
                    with gr.Column(scale=1):
                        internvl_img = gr.Image(label="Satellite Image", type="filepath")
                        internvl_mode = gr.Radio(
                            choices=["Visual Question Answering (VQA)", "Visual Grounding (Bounding Boxes)"],
                            value="Visual Grounding (Bounding Boxes)",
                            label="Execution Mode",
                        )
                        internvl_query = gr.Textbox(
                            label="Query / Object Target",
                            value="water body",
                            placeholder="e.g. 'water body', 'airport runway', 'storage tanks'",
                        )
                        internvl_btn = gr.Button("🚀 Run InternVL2", variant="primary")

                    with gr.Column(scale=1):
                        internvl_text = gr.Markdown(label="Answer / Details")
                        internvl_annotated = gr.Image(label="Grounding Visualization")

                internvl_btn.click(
                    fn=predict_internvl2,
                    inputs=[internvl_img, internvl_query, internvl_mode, api_key_input],
                    outputs=[internvl_text, internvl_annotated],
                )

            # -------------------------------------------------------------------
            # TAB 4: Qwen2-VL-7B
            # -------------------------------------------------------------------
            with gr.TabItem("🔄 4. Qwen2-VL-7B (Change VQA)"):
                gr.Markdown("### Bi-Temporal Satellite Change Understanding & VQA")
                with gr.Row():
                    with gr.Column(scale=1):
                        qwen_pre = gr.Image(label="Pre-Event Image (Time T0)", type="filepath")
                        qwen_post = gr.Image(label="Post-Event Image (Time T1)", type="filepath")
                        qwen_query = gr.Textbox(
                            label="Change Query",
                            value="What changes occurred between Date 1 and Date 2?",
                        )
                        qwen_prior = gr.Textbox(
                            label="Optional Sensor Prior",
                            placeholder="e.g. Urban fabric expansion",
                        )
                        qwen_btn = gr.Button("🚀 Analyze Multi-Temporal Changes", variant="primary")

                    with gr.Column(scale=1):
                        qwen_out = gr.Markdown(label="Change Analysis Output")

                qwen_btn.click(
                    fn=predict_qwen2vl,
                    inputs=[qwen_pre, qwen_post, qwen_query, qwen_prior, api_key_input],
                    outputs=[qwen_out],
                )

            # -------------------------------------------------------------------
            # TAB 5: CDVQA Baseline
            # -------------------------------------------------------------------
            with gr.TabItem("🔍 5. CDVQA Baseline (Change Mask)"):
                gr.Markdown("### Dual-Temporal Change Mask Overlay & Surface Statistics")
                with gr.Row():
                    with gr.Column(scale=1):
                        cdvqa_img1 = gr.Image(label="Image / Patch 1 (Time T0)", type="filepath")
                        cdvqa_img2 = gr.Image(label="Image / Patch 2 (Time T1)", type="filepath")
                        cdvqa_query = gr.Textbox(
                            label="Analysis Query",
                            value="Identify changed regions.",
                        )
                        cdvqa_btn = gr.Button("🚀 Generate Change Mask", variant="primary")

                    with gr.Column(scale=1):
                        cdvqa_text = gr.Markdown(label="Change Statistics")
                        cdvqa_mask = gr.Image(label="Change Mask Overlay (Red = Change)")

                cdvqa_btn.click(
                    fn=predict_cdvqa,
                    inputs=[cdvqa_img1, cdvqa_img2, cdvqa_query],
                    outputs=[cdvqa_text, cdvqa_mask],
                )

            # -------------------------------------------------------------------
            # TAB 6: Agentic Orchestrator (Requirement 5)
            # -------------------------------------------------------------------
            with gr.TabItem("🤖 6. Agentic Orchestrator (Unified Pipeline)"):
                gr.Markdown(
                    "### Autonomous Multi-Model Agent Pipeline\n"
                    "The agent evaluates your query and satellite images, dynamically invokes and chains specialist tools "
                    "(**Optical-SAR Specialist** -> **CDVQA** delta mask -> **Qwen2-VL** temporal reasoning -> "
                    "**InternVL2** visual grounding -> **PaliGemma** captioning & VQA synthesis), "
                    "and combines the results into an executive intelligence report."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        agent_img1 = gr.Image(label="Primary Satellite Image (Time T0)", type="filepath")
                        agent_img2 = gr.Image(label="Optional Secondary Image (Time T1 for Change Detection)", type="filepath")
                        agent_prior_choice = gr.Radio(
                            choices=["ResNet-18", "ViT-Base"],
                            value="ResNet-18",
                            label="Select Land Cover Prior Specialist",
                        )
                        agent_query = gr.Textbox(
                            label="User Query / Mission Objective",
                            value="Identify the land cover types, describe the scene, locate prominent vegetation, and detect any temporal changes.",
                            lines=3,
                        )
                        agent_btn = gr.Button("🚀 Run Agentic Orchestrator", variant="primary")

                    with gr.Column(scale=1):
                        agent_report = gr.Markdown(label="Executive Report")
                        agent_viz = gr.Image(label="Annotated Visual Output (Grounding / Changes)")
                        agent_plan = gr.Markdown(label="Agent Decision & Execution Plan")
                        agent_trace = gr.Markdown(label="Execution Trace JSON")

                agent_btn.click(
                    fn=run_agentic_orchestrator,
                    inputs=[agent_img1, agent_img2, agent_query, agent_prior_choice, api_key_input],
                    outputs=[agent_report, agent_plan, agent_viz, agent_trace],
                )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  SatQuery AI - Interactive Web UI Starting")
    print("  Access the Web UI in your browser at: http://127.0.0.1:7860")
    print("=" * 60)
    app = build_ui()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860, inbrowser=True)
