"""
Unified VLM API Integration for SatQuery AI
============================================
Provides live API inference for Vision-Language tasks (VQA, Captioning, Grounding,
and Bi-temporal Change VQA) using free cloud endpoints (OpenRouter / Google Gemini)
with seamless fallback to local heuristics when offline or when no API key is set.
"""
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from PIL import Image


def get_api_key(provided_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Retrieve API key and identify provider ('openrouter' or 'gemini')."""
    key = provided_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None, "none"
    key = key.strip()
    if key.startswith("sk-or-") or len(key) > 50:
        return key, "openrouter"
    if key.startswith("AIza"):
        return key, "gemini"
    # Default to openrouter if unknown format
    return key, "openrouter"


def encode_image_to_base64(image_input: Union[str, Path, Image.Image]) -> str:
    """Convert an image file path or PIL Image to base64 jpeg string."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_grounding_boxes(text: str, img_width: int, img_height: int) -> List[List[float]]:
    """Parse bounding boxes from text output into [x1, y1, x2, y2]."""
    boxes: List[List[float]] = []

    # 1. Look for [ymin, xmin, ymax, xmax] (normalized 0-1000 or 0-1)
    # Match patterns like [120, 340, 500, 600]
    pattern = r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"
    matches = re.findall(pattern, text)
    for m in matches:
        coords = [float(x) for x in m]
        # Check if coordinates are normalized 0-1000 (Gemini format: [ymin, xmin, ymax, xmax])
        if max(coords) <= 1000 and any(c > 1.0 for c in coords):
            # Gemini order: ymin, xmin, ymax, xmax -> convert to x1, y1, x2, y2 in pixels
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
            # Raw pixels
            boxes.append([round(c, 1) for c in coords])

    return boxes


def call_live_vlm(
    prompt: str,
    images: List[Union[str, Path, Image.Image]],
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> Tuple[str, str, float]:
    """
    Execute live VLM request. Returns (response_text, model_name, elapsed_ms).
    """
    key, provider = get_api_key(api_key)
    if not key:
        raise ValueError("No API key available. Provide an OpenRouter or Gemini API key.")

    start_time = time.perf_counter()

    if provider == "gemini":
        model_name = preferred_model or "gemini-2.0-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}

        parts = []
        for img in images:
            b64 = encode_image_to_base64(img)
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    else:
        # OpenRouter
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
            "max_tokens": 512,
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return text, model_name, elapsed_ms
