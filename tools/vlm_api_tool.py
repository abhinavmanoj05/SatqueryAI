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

import requests
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env if present
load_dotenv()


def get_api_key(provided_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Retrieve API key and identify provider ('gemini' or 'openrouter')."""
    key = (
        provided_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not key:
        return None, "none"
    key = key.strip()
    if key.startswith("AIza"):
        return key, "gemini"
    if key.startswith("sk-or-") or len(key) > 40:
        return key, "openrouter"
    # Default: if user provided a key, assume gemini unless sk-or- prefix
    return key, "gemini"


def encode_image_to_base64(image_input: Union[str, Path, Image.Image]) -> str:
    """Convert an image file path or PIL Image to base64 JPEG string."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

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
            "generationConfig": {
                "temperature": 0.15,
                "maxOutputTokens": 1024,
            },
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Google Gemini API error ({resp.status_code}): {resp.text}")
        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response structure from Gemini API: {data}") from e
        model_display = f"Google Gemini 2.0 Flash ({model_name})"

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
