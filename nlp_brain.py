"""
nlp_brain.py
============
Cognitive Natural Language Processing (NLP) Brain for SatQuery AI.
Acts as the central intelligence layer for:
1. Deep semantic understanding of natural language user queries.
2. Step-by-step agentic "thinking" and reasoning for specialist model selection.
3. Flexible support for multiple backends:
   - Local Ollama (e.g. llama-3.2-3b, deepseek-r1, qwen3-vl)
   - Google Gemini API (gemini-2.0-flash)
   - OpenRouter / OpenAI API
   - Deterministic rule engine fallback (always available offline)
4. Conversational / advisory dialogue when no imagery is uploaded or for general remote sensing queries.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

TaskType = Literal[
    "vqa",
    "captioning",
    "grounding",
    "change_detection",
    "fusion",
    "land_cover_analysis",
    "conversational",
]


@dataclass
class BrainDecision:
    task: TaskType
    thinking: str
    input_count: int
    expected_modality: str  # "optical", "sar", "both", or "none"
    requires_spatial_output: bool
    models_to_invoke: List[str]
    direct_response: Optional[str] = None
    provider_used: str = "rule_engine"


def get_available_ollama_model(host: str = "http://localhost:11434") -> Optional[str]:
    """Check if local Ollama daemon is reachable and return preferred model."""
    configured = os.environ.get("OLLAMA_MODEL")
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            models_info = resp.json().get("models", [])
            available_names = [m.get("name", "") for m in models_info]
            if not available_names:
                return None

            if configured and any(configured in name for name in available_names):
                return configured
            # Prioritize faster 3B/4B models for query parsing
            for preferred in [
                "llama-3.2",
                "richardyoung/llama-3.2-3b-instruct-abliterated:latest",
                "qwen3-vl",
                "deepseek-r1:8b",
            ]:
                for name in available_names:
                    if preferred in name:
                        return name
            return available_names[0]
    except Exception:
        return None
    return None


def _clean_json_string(raw: str) -> str:
    """Extract JSON object from markdown code fences or raw text."""
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def _build_brain_prompt(query: str, num_files: int) -> str:
    return f"""You are the cognitive orchestrator brain for SatQuery AI, an expert satellite remote-sensing multimodal platform.
User Query: "{query}"
Number of Uploaded Imagery Files: {num_files}

Available Specialist Pipelines & Models:
1. "change_detection": Bi-temporal change detection (Requires 2 optical images). Uses CDVQA Baseline pixel differencing mask + Google Gemini 2.0 Flash / Qwen 2.5-VL for change reasoning.
2. "fusion": Optical-SAR joint multi-modal analysis (Requires 1 Optical + 1 SAR image). Uses ViT-Base (BigEarthNet 12-channel) + Gemini 2.0 Flash.
3. "land_cover_analysis": Land-cover classification (1 or 2 images). Uses ViT-Base (BigEarthNet 12-channel) + Gemini 2.0 Flash.
4. "grounding": Text-guided spatial localization of features/objects (Requires 1 optical image). Uses Google Gemini 2.0 Flash coordinate grounding.
5. "captioning": Detailed scene description of landscape and topography (Requires 1 optical image). Uses Gemini 2.0 Flash / PaliGemma.
6. "vqa": Visual question answering on an image (Requires 1 image). Uses Gemini 2.0 Flash with ViT-Base sensor prior.
7. "conversational": General conversation, greetings, remote sensing knowledge explanation, or guidance on what imagery to upload when 0 images are provided.

You must respond ONLY with a valid JSON object with these exact keys:
{{
  "task": "change_detection" | "fusion" | "land_cover_analysis" | "grounding" | "captioning" | "vqa" | "conversational",
  "thinking": "Step-by-step reasoning explaining why this task and specialist model chain were chosen based on the user's intent.",
  "input_count": 0, 1, or 2,
  "expected_modality": "optical" | "sar" | "both" | "none",
  "requires_spatial_output": true or false,
  "models_to_invoke": ["Model 1", "Model 2"],
  "direct_response": "If conversational or if no imagery is uploaded, write a helpful, expert remote sensing response directly here. If an image analysis task with uploaded images, set to null."
}}"""


def call_ollama_brain(query: str, num_files: int, model_name: str, host: str = "http://localhost:11434") -> Optional[BrainDecision]:
    """Execute query comprehension and thinking through local Ollama."""
    prompt = _build_brain_prompt(query, num_files)
    url = f"{host.rstrip('/')}/api/generate"
    try:
        resp = requests.post(
            url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 700},
            },
            timeout=18,
        )
        if resp.status_code == 200:
            raw_text = resp.json().get("response", "")
            json_str = _clean_json_string(raw_text)
            data = json.loads(json_str)
            return BrainDecision(
                task=data.get("task", "vqa"),
                thinking=data.get("thinking", f"Ollama ({model_name}) analyzed query semantics and selected specialist routing."),
                input_count=int(data.get("input_count", 1 if num_files > 0 else 0)),
                expected_modality=data.get("expected_modality", "optical"),
                requires_spatial_output=bool(data.get("requires_spatial_output", False)),
                models_to_invoke=data.get("models_to_invoke", ["ViT-Base", "Google Gemini 2.0 Flash"]),
                direct_response=data.get("direct_response"),
                provider_used=f"ollama:{model_name}",
            )
    except Exception:
        return None
    return None


def call_gemini_brain(query: str, num_files: int, api_key: str) -> Optional[BrainDecision]:
    """Execute query comprehension and thinking through Google Gemini (Flash 3 / Flash 2.0)."""
    prompt = _build_brain_prompt(query, num_files)
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    candidate_models = [
        os.environ.get("GEMINI_MODEL"),
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    models_to_try = [m for m in candidate_models if m]

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = json.loads(raw_text)
                return BrainDecision(
                    task=parsed.get("task", "vqa"),
                    thinking=parsed.get("thinking", f"Google Gemini ({model_name}) parsed query intent and determined optimal specialist pipeline."),
                    input_count=int(parsed.get("input_count", 1 if num_files > 0 else 0)),
                    expected_modality=parsed.get("expected_modality", "optical"),
                    requires_spatial_output=bool(parsed.get("requires_spatial_output", False)),
                    models_to_invoke=parsed.get("models_to_invoke", ["ViT-Base", f"Google Gemini ({model_name})"]),
                    direct_response=parsed.get("direct_response"),
                    provider_used=f"gemini:{model_name}",
                )
        except Exception:
            continue
    return None
    return None


def call_rule_based_brain(query: str, num_files: int) -> BrainDecision:
    """Deterministic, high-speed remote-sensing rule brain for offline environments."""
    q = query.lower()

    # Conversational or greeting detection when 0 files uploaded
    if num_files == 0:
        if any(w in q for w in ["hello", "hi", "hey", "help", "who are you", "what can you do", "explain", "model"]):
            return BrainDecision(
                task="conversational",
                thinking="User entered a conversational or exploratory prompt with no imagery attached. Generating system guidance and remote sensing capabilities overview.",
                input_count=0,
                expected_modality="none",
                requires_spatial_output=False,
                models_to_invoke=["SatQuery AI Knowledge Core"],
                direct_response=(
                    "👋 **Welcome to SatQuery AI** — your multimodal Earth Observation & satellite intelligence assistant.\n\n"
                    "Here is what you can do:\n"
                    "• **Optical & SAR Fusion**: Upload an optical image + Sentinel-1 SAR GeoTIFF. SatQuery uses **ViT-Base (BigEarthNet)** for 12-channel land-cover classification.\n"
                    "• **Visual Grounding**: Ask to highlight or locate features (e.g. *'Highlight the water bodies and forest parcels'*).\n"
                    "• **Bi-Temporal Change Detection**: Upload two images (pre- and post-event) to generate pixel difference masks via **CDVQA** and reason about surface delta.\n"
                    "• **Scene Captioning**: Generate technical descriptions of topography, canopy density, and built-up areas.\n\n"
                    "👉 *To begin, drag & drop satellite imagery into the Upload Zone above and submit your query!*"
                ),
                provider_used="rule_engine",
            )
        else:
            return BrainDecision(
                task="conversational",
                thinking=f"User submitted analysis query '{query}' but no imagery is uploaded yet. Providing preparatory guidance for the requested remote sensing task.",
                input_count=0,
                expected_modality="none",
                requires_spatial_output=False,
                models_to_invoke=["SatQuery AI Assistant"],
                direct_response=(
                    f"I understand your query: *\"{query}\"*.\n\n"
                    "To execute this analysis, please **upload the satellite imagery** in the panel above:\n"
                    "• For **Single Image Analysis** (VQA, Captioning, Grounding): Upload 1 Optical GeoTIFF or PNG.\n"
                    "• For **Bi-Temporal Change Detection**: Upload 2 acquisition dates (T0 pre-event and T1 post-event).\n"
                    "• For **Optical-SAR Fusion**: Upload 1 Sentinel-2 optical image + 1 Sentinel-1 SAR image.\n\n"
                    "Once uploaded, press **Enter** to run the agentic workflow."
                ),
                provider_used="rule_engine",
            )

    # 1. Change detection
    if any(k in q for k in ["change", "before and after", "compare these two", "difference between", "what changed", "over time", "bi-temporal"]) or (num_files >= 2 and any(k in q for k in ["then", "now", "between"])):
        return BrainDecision(
            task="change_detection",
            thinking="Detected temporal comparison semantics. Routing to CDVQA Baseline for pixel differencing mask computation and Gemini 2.0 Flash for multi-temporal land-cover change reasoning.",
            input_count=2,
            expected_modality="optical",
            requires_spatial_output=True,
            models_to_invoke=["CDVQA Baseline", "Google Gemini 2.0 Flash"],
            provider_used="rule_engine",
        )

    # 2. Optical-SAR Fusion
    if any(k in q for k in ["sar", "radar", "optical and sar", "fuse", "fusion", "both images together", "combine the images"]):
        return BrainDecision(
            task="fusion",
            thinking="Query requests joint SAR and optical sensor fusion. Routing to local PyTorch ViT-Base (BigEarthNet 12-channel) to extract sensor priors from VV, VH, and 10 Sentinel-2 bands, synthesizing with Gemini 2.0 Flash.",
            input_count=2,
            expected_modality="both",
            requires_spatial_output=True,
            models_to_invoke=["ViT-Base (BigEarthNet 12-channel)", "Google Gemini 2.0 Flash"],
            provider_used="rule_engine",
        )

    # 3. Land cover classification
    if any(k in q for k in ["land cover", "land-cover", "classify", "land use", "landuse"]):
        return BrainDecision(
            task="land_cover_analysis",
            thinking="Land cover categorization requested. Executing ViT-Base multi-label classification across the 19 BigEarthNet CORINE classes to determine land-use distribution.",
            input_count=max(1, min(num_files, 2)),
            expected_modality="both" if num_files >= 2 else "optical",
            requires_spatial_output=True,
            models_to_invoke=["ViT-Base (BigEarthNet 12-channel)"],
            provider_used="rule_engine",
        )

    # 4. Visual Grounding
    if any(k in q for k in ["highlight", "locate", "point out", "circle", "outline", "mark the", "bounding box", "where is", "find the"]):
        return BrainDecision(
            task="grounding",
            thinking="Spatial localization requested. Directing to Gemini 2.0 Flash coordinate localization engine to detect normalized 2D bounding boxes for targeted features.",
            input_count=1,
            expected_modality="optical",
            requires_spatial_output=True,
            models_to_invoke=["Google Gemini 2.0 Flash"],
            provider_used="rule_engine",
        )

    # 5. Scene Captioning
    if any(k in q for k in ["describe", "caption", "summarize the scene", "what does this image show", "give a description"]):
        return BrainDecision(
            task="captioning",
            thinking="High-level scene description requested. Routing to Gemini 2.0 Flash for comprehensive remote sensing landscape and terrain characterization.",
            input_count=1,
            expected_modality="optical",
            requires_spatial_output=False,
            models_to_invoke=["Google Gemini 2.0 Flash"],
            provider_used="rule_engine",
        )

    # Default VQA
    return BrainDecision(
        task="vqa",
        thinking="Standard visual question answering query. Invoking ViT-Base for initial multispectral sensor prior generation and Google Gemini 2.0 Flash for evidence-grounded answer formulation.",
        input_count=max(1, min(num_files, 1)),
        expected_modality="optical",
        requires_spatial_output=False,
        models_to_invoke=["ViT-Base (BigEarthNet 12-channel)", "Google Gemini 2.0 Flash"],
        provider_used="rule_engine",
    )


def understand_query_with_nlp_brain(
    query: str,
    num_files: int,
    api_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
) -> BrainDecision:
    """
    Main entry point for the Cognitive NLP Brain.
    - If num_files == 0: handles conversational/guidance queries.
    - If num_files > 0: accurately routes remote sensing task and generates thinking trace.
    """
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    omniroute_key = os.environ.get("OMNIRoute_API_KEY") or "sk-2cb602d99709fd78-a3daf9-ff8ee46f"
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # 1. When no images are uploaded, handle conversationally
    if num_files == 0:
        if effective_key:
            decision = call_gemini_brain(query, num_files, api_key=effective_key)
            if decision:
                return decision
        elif omniroute_key:
            decision = call_omniroute_brain(query, num_files, api_key=omniroute_key)
            if decision:
                return decision
        model = get_available_ollama_model(ollama_host)
        if model:
            decision = call_ollama_brain(query, num_files, model_name=model, host=ollama_host)
            if decision:
                return decision
        return call_rule_based_brain(query, num_files)

    # 2. When images are uploaded, determine the task with high precision
    rule_decision = call_rule_based_brain(query, num_files)
    rule_decision.direct_response = None

    # Enrich thinking with Ollama, Gemini, or Omniroute if available, maintaining task routing accuracy
    if effective_key:
        try:
            gemini_dec = call_gemini_brain(query, num_files, api_key=effective_key)
            if gemini_dec and gemini_dec.thinking:
                rule_decision.thinking = gemini_dec.thinking
                rule_decision.provider_used = "gemini-2.0-flash"
        except Exception:
            pass
    elif preferred_provider == "ollama":
        try:
            model = get_available_ollama_model(ollama_host)
            if model:
                ollama_dec = call_ollama_brain(query, num_files, model_name=model, host=ollama_host)
                if ollama_dec and ollama_dec.thinking:
                    rule_decision.thinking = ollama_dec.thinking
                    rule_decision.provider_used = f"ollama:{model}"
        except Exception:
            pass
    elif omniroute_key:
        try:
            omni_dec = call_omniroute_brain(query, num_files, api_key=omniroute_key)
            if omni_dec and omni_dec.thinking:
                rule_decision.thinking = omni_dec.thinking
                rule_decision.provider_used = "omniroute"
        except Exception:
            pass

    return rule_decision
