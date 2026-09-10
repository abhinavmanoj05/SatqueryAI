"""
tests/test_orchestrator.py
===========================
Unit tests for individual nodes + integration tests for the full compiled
graph (via `run_satquery`). Run with:

    cd satquery
    python -m pytest tests/ -v

All specialist backends (InternVL2Tool, PaliGemmaTool, Qwen2VLTool,
ResNet18Tool) are the deterministic mock implementations shipped in this
project (see e.g. internvl2_tool.py), so these tests are fully offline and
reproducible -- no network access or model weights required.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agentic_orchestrator import (
    build_graph,
    output_combinator,
    should_validate,
    task_router,
    validation_gate,
)
from intent_classifier import classify_intent_rule_based
from run_orchestrator import run_satquery
from state_schema import IntentSchema


# ---------------------------------------------------------------------------
# 1. Query parser / intent classification (unit)
# ---------------------------------------------------------------------------
class TestQueryParser:
    def test_change_detection_intent(self):
        intent = classify_intent_rule_based("What changed between these two dates?", num_files=2)
        assert intent.primary_task == "change_detection"
        assert intent.input_count == 2
        assert intent.requires_spatial_output is True

    def test_fusion_intent(self):
        intent = classify_intent_rule_based(
            "Use the optical and SAR images together to identify built-up and water.",
            num_files=2,
        )
        assert intent.primary_task == "fusion"
        assert intent.expected_modality == "both"

    def test_grounding_intent(self):
        intent = classify_intent_rule_based("Highlight the water body in this image.", num_files=1)
        assert intent.primary_task == "grounding"
        assert intent.requires_spatial_output is True

    def test_captioning_intent(self):
        intent = classify_intent_rule_based("Describe this satellite scene.", num_files=1)
        assert intent.primary_task == "captioning"

    def test_land_cover_intent(self):
        intent = classify_intent_rule_based("Classify the land cover in this pair of images.", num_files=2)
        assert intent.primary_task == "land_cover_analysis"

    def test_default_vqa_intent(self):
        intent = classify_intent_rule_based("Is there a road in this image?", num_files=1)
        assert intent.primary_task == "vqa"
        assert intent.input_count == 1

    def test_intent_schema_rejects_bad_task(self):
        with pytest.raises(Exception):
            IntentSchema(primary_task="not_a_real_task")


# ---------------------------------------------------------------------------
# 2. Validation gate (unit)
# ---------------------------------------------------------------------------
class TestValidationGate:
    def test_valid_single_optical_file(self):
        state = {
            "intent": {"primary_task": "vqa", "input_count": 1, "expected_modality": "optical"},
            "uploaded_files": [{"path": "a.tif", "modality": "optical", "format": "geotiff"}],
        }
        result = validation_gate(state)
        assert result["validation_result"]["is_valid"] is True
        assert result["error"] is None

    def test_missing_file_count(self):
        state = {
            "intent": {"primary_task": "change_detection", "input_count": 2, "expected_modality": "optical"},
            "uploaded_files": [{"path": "a.tif", "modality": "optical", "format": "geotiff"}],
        }
        result = validation_gate(state)
        assert result["validation_result"]["is_valid"] is False
        assert "requires 2 image" in result["error"]

    def test_fusion_missing_sar(self):
        state = {
            "intent": {"primary_task": "fusion", "input_count": 2, "expected_modality": "both"},
            "uploaded_files": [
                {"path": "a.tif", "modality": "optical", "format": "geotiff"},
                {"path": "b.tif", "modality": "optical", "format": "geotiff"},
            ],
        }
        result = validation_gate(state)
        assert result["validation_result"]["is_valid"] is False
        assert "SAR image" in result["error"]

    def test_unsupported_format_rejected(self):
        state = {
            "intent": {"primary_task": "vqa", "input_count": 1, "expected_modality": "optical"},
            "uploaded_files": [{"path": "a.bmp", "modality": "optical", "format": "bmp"}],
        }
        result = validation_gate(state)
        assert result["validation_result"]["is_valid"] is False
        assert "Unsupported format" in result["error"]

    def test_should_validate_conditional_edge(self):
        valid_state = {"validation_result": {"is_valid": True}}
        invalid_state = {"validation_result": {"is_valid": False}}
        assert should_validate(valid_state) == "continue"
        assert should_validate(invalid_state) == "error"


# ---------------------------------------------------------------------------
# 3. Task router (unit)
# ---------------------------------------------------------------------------
class TestTaskRouter:
    @pytest.mark.parametrize(
        "task,expected_node",
        [
            ("vqa", "vqa_node"),
            ("captioning", "captioning_node"),
            ("grounding", "grounding_node"),
            ("change_detection", "change_detection_node"),
            ("fusion", "fusion_node"),
            ("land_cover_analysis", "fusion_node"),
            ("some_unknown_task", "vqa_node"),  # falls back to vqa_node
        ],
    )
    def test_routing_map(self, task, expected_node):
        state = {"validation_result": {"is_valid": True}, "intent": {"primary_task": task}}
        result = task_router(state)
        assert result["routing_decision"] == expected_node

    def test_router_defensive_error_branch(self):
        state = {"validation_result": {"is_valid": False}, "intent": {"primary_task": "vqa"}}
        result = task_router(state)
        assert result["routing_decision"] == "error"


# ---------------------------------------------------------------------------
# 4. Output combinator (unit)
# ---------------------------------------------------------------------------
class TestOutputCombinator:
    def test_combines_vqa_output(self):
        state = {
            "intent": {"primary_task": "vqa"},
            "tool_outputs": {"vqa": {"answer": "Yes, there is a road.", "confidence": 0.9}},
        }
        result = output_combinator(state)
        assert result["final_answer"] == "Yes, there is a road."
        assert result["confidence"] == 0.9
        assert result["execution_trace"]["models_used"] == ["InternVL2-8B"]

    def test_combines_fusion_output(self):
        state = {
            "intent": {"primary_task": "fusion"},
            "tool_outputs": {
                "fusion": {
                    "resnet_prior": {
                        "sensor_prior": "ResNet-18 detects: Urban fabric (82%)",
                        "top_k": [{"class": "Urban fabric", "probability": 0.82}],
                        "confidence": 0.82,
                    },
                    "vlm_answer": "Built-up area dominant.",
                }
            },
        }
        result = output_combinator(state)
        assert result["final_answer"] == "Built-up area dominant."
        assert result["confidence"] == 0.82
        assert "ResNet-18" in result["execution_trace"]["models_used"]
        assert "InternVL2-8B" in result["execution_trace"]["models_used"]
        assert result["execution_trace"]["parameters"]["sensor_prior"].startswith("ResNet-18 detects")

    def test_surfaces_specialist_error_without_generic_fallback(self):
        state = {
            "intent": {"primary_task": "fusion"},
            "tool_outputs": {},
            "error": "Fusion requires both optical and SAR images.",
        }
        result = output_combinator(state)
        assert "Fusion requires both optical and SAR images." in result["final_answer"]
        assert result["confidence"] == 0.0

    def test_generic_fallback_when_truly_empty(self):
        state = {"intent": {"primary_task": "vqa"}, "tool_outputs": {}}
        result = output_combinator(state)
        assert result["final_answer"] == "Unable to process the query. Please check your inputs."


# ---------------------------------------------------------------------------
# 5. Full graph (integration)
# ---------------------------------------------------------------------------
class TestFullGraph:
    def test_graph_compiles(self):
        g = build_graph()
        assert g is not None

    def test_vqa_end_to_end(self):
        result = run_satquery(
            user_query="Is there a road visible in this image?",
            uploaded_files=[{"path": "scene.tif", "modality": "optical", "format": "geotiff"}],
        )
        assert result["final_answer"]
        assert result["execution_trace"]["task"] == "vqa"
        assert result["execution_trace"]["models_used"] == ["InternVL2-8B"]
        assert result["execution_trace"]["input_count"] == 1
        assert "duration_ms" in result["execution_trace"]
        assert "timestamp" in result["execution_trace"]

    def test_captioning_end_to_end(self):
        result = run_satquery(
            user_query="Describe this satellite scene.",
            uploaded_files=[{"path": "scene.tif", "modality": "optical", "format": "png"}],
        )
        assert result["execution_trace"]["task"] == "captioning"
        assert "PaliGemma-3B" in result["execution_trace"]["models_used"]

    def test_grounding_end_to_end(self):
        result = run_satquery(
            user_query="Highlight the water body in this image.",
            uploaded_files=[{"path": "scene.tif", "modality": "optical", "format": "geotiff"}],
        )
        assert result["execution_trace"]["task"] == "grounding"
        assert result["visual_evidence"]["boxes"]

    def test_change_detection_end_to_end(self):
        result = run_satquery(
            user_query="What changed between these two dates?",
            uploaded_files=[
                {"path": "t1.tif", "modality": "optical", "format": "geotiff"},
                {"path": "t2.tif", "modality": "optical", "format": "geotiff"},
            ],
        )
        assert result["execution_trace"]["task"] == "change_detection"
        assert "Qwen2-VL-7B" in result["execution_trace"]["models_used"]
        assert result["execution_trace"]["input_count"] == 2

    def test_fusion_end_to_end(self):
        result = run_satquery(
            user_query="Use the optical and SAR images together to identify built-up and water.",
            uploaded_files=[
                {"path": "optical.tif", "modality": "optical", "format": "geotiff"},
                {"path": "sar.tif", "modality": "sar", "format": "geotiff"},
            ],
        )
        assert result["final_answer"] is not None
        assert result["execution_trace"]["task"] == "fusion"
        assert "ResNet-18" in result["execution_trace"]["models_used"]
        assert result["confidence"] > 0

    def test_validation_failure_short_circuits_before_any_specialist_runs(self):
        result = run_satquery(
            user_query="What changed between these two dates?",
            uploaded_files=[{"path": "only_one.tif", "modality": "optical", "format": "geotiff"}],
        )
        assert result["tool_outputs"] == {}
        assert "requires 2 image" in result["error"]
        assert "Request could not be validated" in result["final_answer"]
        # Even an early-exit still gets a well-formed audit trace.
        assert result["execution_trace"]["task"] == "change_detection"
        assert result["execution_trace"]["models_used"] == []

    def test_unsupported_format_short_circuits(self):
        result = run_satquery(
            user_query="Is there a road visible?",
            uploaded_files=[{"path": "scene.bmp", "modality": "optical", "format": "bmp"}],
        )
        assert "Unsupported format" in result["error"]
        assert result["confidence"] == 0.0
