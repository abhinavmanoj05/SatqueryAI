"""
run_orchestrator.py
====================
Main execution interface for the SatQuery AI Agentic Orchestrator.
Accepts user query & uploaded files, executes the LangGraph StateGraph,
and returns structured responses matching the frontend contract and test requirements.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentic_orchestrator import build_graph, default_graph


def run_satquery(
    user_query: str,
    uploaded_files: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[Any] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the SatQuery AI agentic orchestrator pipeline end-to-end.

    Args:
        user_query: User text prompt or question.
        uploaded_files: List of file dictionaries, e.g.:
            [{"path": "sample.png", "modality": "optical", "format": "png"}]
        llm: Optional LangChain chat model for intent classification.
        api_key: Optional live Gemini / OpenRouter API key for VLM models.

    Returns:
        Structured dictionary matching SatQueryState and OrchestratorResponse.
    """
    if uploaded_files is None:
        uploaded_files = []

    compiled_graph = default_graph if llm is None else build_graph(llm=llm)
    start_time = time.perf_counter()

    initial_state: Dict[str, Any] = {
        "messages": [],
        "user_query": user_query,
        "uploaded_files": uploaded_files,
        "api_key": api_key,
        "tool_outputs": {},
    }

    result = compiled_graph.invoke(initial_state)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if result.get("execution_trace"):
        result["execution_trace"]["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["execution_trace"]["duration_ms"] = duration_ms
        result["execution_trace"]["input_count"] = len(uploaded_files)
    else:
        task_detected = (result.get("intent") or {}).get("primary_task", "unknown")
        result["execution_trace"] = {
            "task": task_detected,
            "models_used": [],
            "parameters": {},
            "input_count": len(uploaded_files),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "confidence_label": "Low",
            "thinking": result.get("thinking", ""),
        }
        if not result.get("final_answer"):
            err = result.get("error", "Validation error")
            result["final_answer"] = f"Request could not be validated: {err}"
        result.setdefault("confidence", 0.0)

    result.setdefault("tool_outputs", {})
    if result.get("visual_evidence") is None:
        result["visual_evidence"] = {}
    return result


if __name__ == "__main__":
    import json
    res = run_satquery("Describe this satellite scene.")
    print(json.dumps(res, indent=2))
