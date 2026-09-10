"""
tests/test_report_generation.py
================================
Unit & integration tests for the SatQuery AI research report generation endpoint (/api/generate-report).
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend_api import app


client = TestClient(app)


def test_generate_report_endpoint_with_sample_session():
    """Verify that /api/generate-report returns an executive summary with telemetry."""
    payload = {
        "session_id": "test-session-123",
        "session_title": "Agricultural and Water Body Analysis in South Asia",
        "preferred_model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "Identify the primary land-cover categories in this Sentinel-2 scene.",
                "timestamp": "10:30 AM",
            },
            {
                "role": "assistant",
                "content": "Multispectral analysis indicates dominant agricultural land (84.2%) and water bodies (12.1%).",
                "timestamp": "10:31 AM",
                "response": {
                    "final_answer": "Multispectral analysis indicates dominant agricultural land (84.2%) and water bodies (12.1%).",
                    "visual_evidence": {
                        "top_k": [
                            {"class": "Permanently irrigated land", "probability": 0.842},
                            {"class": "Water bodies", "probability": 0.121},
                        ],
                        "boxes": [
                            {"coords": [100, 150, 400, 500], "label": "Irrigated Crop Parcel", "confidence": 0.91}
                        ],
                    },
                    "confidence": 0.88,
                    "execution_trace": {
                        "task": "land_cover_analysis",
                        "models_used": ["ViT-Base (BigEarthNet 12-Band)", "Google Gemini 3.8 Flash"],
                        "duration_ms": 1150.0,
                    },
                },
            },
            {
                "role": "user",
                "content": "Are there any bi-temporal changes in the irrigation network?",
                "timestamp": "10:32 AM",
            },
            {
                "role": "assistant",
                "content": "Bi-temporal differencing detected 6.4% change with 4.1% vegetation expansion.",
                "timestamp": "10:33 AM",
                "response": {
                    "final_answer": "Bi-temporal differencing detected 6.4% change with 4.1% vegetation expansion.",
                    "visual_evidence": {
                        "stats": {
                            "percentage_changed": 6.4,
                            "vegetation_gain_percentage": 4.1,
                            "vegetation_loss_percentage": 1.2,
                            "active_hotspots": 3,
                        }
                    },
                    "confidence": 0.85,
                    "execution_trace": {
                        "task": "change_detection",
                        "models_used": ["CDVQA Baseline", "Google Gemini 3.8 Flash"],
                        "duration_ms": 1420.0,
                    },
                },
            },
        ],
    }

    resp = client.post("/api/generate-report", json=payload)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()

    assert data["session_id"] == "test-session-123"
    assert data["session_title"] == "Agricultural and Water Body Analysis in South Asia"
    assert "executive_summary" in data
    assert len(data["executive_summary"]) > 50
    assert "model_used" in data
    assert "generated_at" in data

    # Executive summary must contain key sections
    summary = data["executive_summary"]
    assert "Mission" in summary or "Objectives" in summary
    assert "Spectral" in summary or "Land-Cover" in summary


def test_generate_report_endpoint_empty_session():
    """Verify that an empty or minimal session still produces a valid fallback report."""
    payload = {
        "session_id": "empty-session",
        "session_title": "New Empty Query",
        "messages": [],
    }

    resp = client.post("/api/generate-report", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "executive_summary" in data
    assert len(data["executive_summary"]) > 20
