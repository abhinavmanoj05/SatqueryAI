"""
Unified Real VLM Integration for SatQuery AI
=============================================
Provides production Vision-Language inference for:
1. Single-Image VQA (Visual Question Answering)
2. Spatial Grounding (extracting precise bounding box coordinates)
3. Scene Captioning (multi-sentence remote sensing descriptions)
4. Bi-Temporal Change Understanding (comparing Time T0 and Time T1 imagery)

Supported Engines:
- Google Gemini 2.0 Flash (Primary: high spatial resolution, coordinate detection, fast)
- Qwen 2.5-VL 72B Instruct (via OpenRouter Free Tier)

NO DUMMY CODE: If an API key is missing, raises an informative configuration error
prompting the user to configure GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env if present
load_dotenv()


def get_api_key(provided_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Retrieve API key and identify provider ('gemini', 'openrouter', or 'omniroute')."""
    if provided_key:
        k = provided_key.strip()
        if k.startswith("sk-or-"):
            return k, "openrouter"
        if k.startswith("sk-"):
            return k, "omniroute"
        return k, "gemini"

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and gemini_key.strip():
        return gemini_key.strip(), "gemini"

    omniroute_key = os.environ.get("OMNIRoute_API_KEY") or os.environ.get("OMNIROUTE_API_KEY")
    if omniroute_key and omniroute_key.strip():
        return omniroute_key.strip(), "omniroute"

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key and openrouter_key.strip():
        return openrouter_key.strip(), "openrouter"

    return None, "none"


def normalize_raster_image(img: Image.Image) -> Image.Image:
    """
    Normalize multi-bit, float32 SAR (dB), or single-channel images into clean 8-bit RGB.
    Prevents Sentinel-1 SAR negative dB values (-25 to -5 dB) from clipping to pitch black.
    """
    if img.mode in ("F", "I", "I;16", "I;16B", "I;16L", "I;16S"):
        arr = np.array(img, dtype=np.float32)
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            p2 = float(np.percentile(valid, 2))
            p98 = float(np.percentile(valid, 98))
            if p98 > p2:
                arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")
    elif img.mode == "L":
        return img.convert("RGB")
    elif img.mode == "RGBA":
        return img.convert("RGB")
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img


def encode_image_to_base64(image_input: Union[str, Path, Image.Image]) -> str:
    """Convert an image file path or PIL Image to base64 JPEG string with robust raster normalization."""
    if isinstance(image_input, (str, Path)):
        raw_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        raw_img = image_input
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    img = normalize_raster_image(raw_img)

    max_dim = 1536
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_grounding_boxes(text: str, img_width: int, img_height: int) -> List[List[float]]:
    """Parse bounding boxes from VLM text output into absolute [x1, y1, x2, y2] coordinates."""
    boxes: List[List[float]] = []

    # Match bounding box patterns: [ymin, xmin, ymax, xmax] or {"box_2d": [ymin, xmin, ymax, xmax]}
    pattern = r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"
    matches = re.findall(pattern, text)

    for m in matches:
        coords = [float(x) for x in m]
        # Gemini coordinate format: [ymin, xmin, ymax, xmax] normalized to 0-1000
        if max(coords) <= 1000 and any(c > 1.0 for c in coords):
            y1 = (coords[0] / 1000.0) * img_height
            x1 = (coords[1] / 1000.0) * img_width
            y2 = (coords[2] / 1000.0) * img_height
            x2 = (coords[3] / 1000.0) * img_width
            boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        elif max(coords) <= 1.0:
            # 0-1 normalized
            x1 = coords[0] * img_width
            y1 = coords[1] * img_height
            x2 = coords[2] * img_width
            y2 = coords[3] * img_height
            boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        else:
            # Raw pixel coordinates
            boxes.append([round(c, 1) for c in coords])

    return boxes


def call_live_vlm(
    prompt: str,
    images: List[Union[str, Path, Image.Image]],
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> Tuple[str, str, float]:
    """Execute live Vision-Language request against Google Gemini or OpenRouter Qwen 72B."""
    key, provider = get_api_key(api_key)
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not configured.\n"
            "Please set your Google Gemini API key in your .env file or environment:\n"
            "    GEMINI_API_KEY=AIzaSy...\n"
            "(Get a free API key at https://aistudio.google.com/app/apikey)"
        )

    start_time = time.perf_counter()

    if provider == "gemini":
        candidate_models = [
            preferred_model,
            os.environ.get("GEMINI_MODEL"),
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
        ]
        # Deduplicate while preserving order
        models_to_try = []
        for m in candidate_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        text = None
        model_display = None
        last_error = None

        parts = []
        for img in images:
            b64 = encode_image_to_base64(img)
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.15,
                "maxOutputTokens": 1024,
            },
        }
        headers = {"Content-Type": "application/json"}

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        model_display = f"Google Gemini ({model_name})"
                        break
                    except (KeyError, IndexError):
                        last_error = f"Unexpected response structure from Gemini ({model_name})"
                else:
                    last_error = f"Google Gemini API error ({resp.status_code}) on {model_name}"
            except Exception as e:
                last_error = f"Connection timeout or error on Gemini ({model_name}): {e}"
                continue

        # Fallback to local Ollama if Gemini API is blocked or rate-limited
        if text is None:
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            try:
                tag_resp = requests.get(f"{ollama_host}/api/tags", timeout=1.0)
                if tag_resp.status_code == 200:
                    installed = [m.get("name") for m in tag_resp.json().get("models", [])]
                    vision_candidate = None
                    for cand in ["qwen3-vl", "qwen2-vl", "llava"]:
                        for inst in installed:
                            if cand in inst:
                                vision_candidate = inst
                                break
                        if vision_candidate:
                            break
                    if not vision_candidate and installed:
                        vision_candidate = installed[0]

                    if vision_candidate:
                        img_b64s = [encode_image_to_base64(img) for img in images]
                        oresp = requests.post(
                            f"{ollama_host}/api/generate",
                            json={
                                "model": vision_candidate,
                                "prompt": prompt,
                                "images": img_b64s,
                                "stream": False,
                            },
                            timeout=25,
                        )
                        if oresp.status_code == 200:
                            text = oresp.json().get("response", "").strip()
                            model_display = f"Local Ollama ({vision_candidate})"
            except Exception:
                pass

        if text is None:
            raise RuntimeError(last_error or "Failed to obtain response from Gemini API.")

    elif provider == "omniroute":
        base_url = os.environ.get("OMNIROUTE_BASE_URL", "http://localhost:20128/v1").rstrip("/")
        model_name = preferred_model or "gpt-4o-mini"
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        content = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = encode_image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0.15,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        if resp.status_code != 200:
            raise RuntimeError(f"OmniRoute API error ({resp.status_code}): {resp.text}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        model_display = f"OmniRoute ({model_name})"

    else:
        # OpenRouter (Qwen 2.5-VL 72B Instruct)
        model_name = preferred_model or "qwen/qwen2.5-vl-72b-instruct:free"
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/abhinavmanoj05/SatqueryAI",
            "X-Title": "SatQuery AI",
        }
        content = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = encode_image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0.15,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {resp.text}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        model_display = f"Qwen-2.5-VL-72B ({model_name})"

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return text, model_display, elapsed_ms


def call_grounding_vlm(
    image: Union[str, Path, Image.Image],
    target_query: str,
    api_key: Optional[str] = None,
) -> Tuple[str, List[List[float]], str, float]:
    """Execute live visual grounding with Google Gemini 2.0 Flash to detect bounding boxes."""
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    else:
        pil_img = image
    w, h = pil_img.size

    prompt = (
        f"You are a remote sensing visual grounding specialist. Analyze this satellite imagery carefully.\n"
        f"Goal: Detect and localize: '{target_query}'.\n"
        f"Requirements:\n"
        f"1. Return bounding box coordinates for each identified instance in format [ymin, xmin, ymax, xmax] normalized between 0 and 1000.\n"
        f"2. Provide an analytical description explaining what features were identified, their spectral characteristics, and spatial distribution.\n"
        f"Example format:\n"
        f"- Target region: [120, 250, 480, 600]\n"
        f"Explanation: ..."
    )

    text, model, ms = call_live_vlm(prompt, [image], api_key=api_key)
    boxes = parse_grounding_boxes(text, w, h)
    return text, boxes, model, ms
