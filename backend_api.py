"""
backend_api.py
==============
Production FastAPI backend serving the SatQuery AI Agentic Orchestrator.
Exposes REST endpoints consumed by the React + Vite frontend:
- POST /api/analyze (also aliased at /analyze)
- GET /api/health (also aliased at /health)

Receives queries and remote sensing imagery, routes through the LangGraph StateGraph,
executes specialist models (ResNet-18, ViT-Base, PaliGemma, InternVL2, Qwen2-VL, CDVQA),
and responds with structured visual explanations and auditable execution traces.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_orchestrator import run_satquery

app = FastAPI(
    title="SatQuery AI Agentic Orchestrator API",
    version="1.0.0",
    description="Multimodal Remote Sensing Agentic Vision-Language Assistant Backend",
)

# Enable CORS for local Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_CACHE_DIR = _ROOT / ".tmp_uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health status and registered specialist models."""
    return {
        "status": "healthy",
        "service": "SatQuery AI Agentic Orchestrator",
        "models": [
            "ResNet-18 (BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0)",
            "ViT-Base (BIFOLD-BigEarthNetv2-0/vit_base_patch8_224-all-v0.2.0)",
            "PaliGemma-3B (Captioning & VQA Synthesis)",
            "InternVL2-8B (VQA & Visual Grounding)",
            "Qwen2-VL-7B (Bi-Temporal Change Understanding)",
            "CDVQA Baseline (Pixel Change Mask)",
        ],
    }


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze(
    query: str = Form("Analyze the satellite image and describe findings."),
    files: Optional[List[UploadFile]] = File(None),
    api_key: Optional[str] = Form(None),
):
    """
    Main analysis endpoint: routes query & images through the LangGraph orchestrator.
    """
    temp_files = []
    saved_file_records = []

    try:
        if files:
            for idx, file_upload in enumerate(files):
                if not file_upload.filename:
                    continue
                ext = Path(file_upload.filename).suffix or ".png"
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=ext,
                    dir=str(UPLOAD_CACHE_DIR),
                )
                temp_path = Path(temp_file.name)
                temp_files.append(temp_path)

                with open(temp_path, "wb") as f_out:
                    shutil.copyfileobj(file_upload.file, f_out)

                # Determine modality based on filename or index
                fname_lower = file_upload.filename.lower()
                if "sar" in fname_lower or "s1" in fname_lower:
                    modality = "sar"
                elif idx == 1 and ("t1" in fname_lower or "post" in fname_lower):
                    modality = "optical"
                elif idx == 1 and len(files) == 2 and ("sar" in query.lower() or "radar" in query.lower()):
                    modality = "sar"
                else:
                    modality = "optical"

                saved_file_records.append({
                    "path": str(temp_path),
                    "modality": modality,
                    "format": ext.lstrip(".").lower() or "png",
                })

        # Run the agentic orchestrator
        result = run_satquery(
            user_query=query,
            uploaded_files=saved_file_records,
            api_key=api_key,
        )

        return JSONResponse(content=result)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        # Clean up temporary uploaded files
        for t in temp_files:
            try:
                if t.exists():
                    t.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
