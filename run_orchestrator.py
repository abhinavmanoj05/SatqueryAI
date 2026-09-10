"""
run_orchestrator.py
====================
Entry point for the SatQuery AI agentic orchestrator.

Usage:
    from run_orchestrator import run_satquery

    result = run_satquery(
        user_query="What changed between these two dates?",
        uploaded_files=[
            {"path": "img1.tif", "modality": "optical", "format": "geotiff"},
            {"path": "img2.tif", "modality": "optical", "format": "geotiff"},
        ],
    )
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentic_orchestrator import build_graph

# Default graph instance (rule-based intent classifier). Pass `llm=` to
# `run_satquery` if you want to route intent classification through a real
# chat model instead.
_default_graph = build_graph()


def run_satquery(
    user_query: str,
    uploaded_files: List[Dict[str, Any]],
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the SatQuery AI agentic orchestrator end to end.

    Args:
        user_query: The natural-language request from the user.
        uploaded_files: List of dicts, each shaped like
            `{"path": str, "modality": "optical"|"sar", "format": str, ...}`.
        llm: Optional LangChain chat model for production intent
            classification (see `agentic_orchestrator.build_graph`). If
            omitted, uses the default rule-based-classifier graph.

    Returns:
        A dict with `final_answer`, `visual_evidence`, `confidence`,
        `execution_trace`, and (if something went wrong) `error`.
    """
    compiled_graph = _default_graph if llm is None else build_graph(llm=llm)

    start_time = time.time()

    initial_state: Dict[str, Any] = {
        "messages": [],
        "user_query": user_query,
        "uploaded_files": uploaded_files,
        "tool_outputs": {},
    }

    result = compiled_graph.invoke(initial_state)

    duration_ms = (time.time() - start_time) * 1000

    if result.get("execution_trace"):
        result["execution_trace"]["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["execution_trace"]["duration_ms"] = round(duration_ms, 2)
        result["execution_trace"]["input_count"] = len(uploaded_files)
    elif result.get("error"):
        # Validation failed before output_combinator ever ran (graph ended
        # at the "error" branch) -- still return a minimal, well-formed
        # audit trace rather than silently omitting it.
        result["execution_trace"] = {
            "task": (result.get("intent") or {}).get("primary_task", "unknown"),
            "models_used": [],
            "parameters": {},
            "input_count": len(uploaded_files),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration_ms, 2),
        }
        result.setdefault("final_answer", f"Request could not be validated: {result['error']}")
        result.setdefault("confidence", 0.0)

    return result


if __name__ == "__main__":

 import json 
 demo_result = run_satquery( user_query="Describe this satellite scene.",
 uploaded_files=[ {"path": "scene.tif", "modality": "optical", "format": "geotiff"}, ], )
 print(json.dumps(demo_result, indent=2, default=str))