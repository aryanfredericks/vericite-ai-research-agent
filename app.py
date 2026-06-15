"""Flask API for the literature-review agent.

The compiled LangGraph is built ONCE at startup (it loads LLM clients and the
embedding model and compiles the graph -- all expensive), then reused for every
request. graph.invoke() is stateless: the whole run state is passed in and
returned, so concurrent requests don't interfere.

Run (dev):      python app.py
Run (prod):     gunicorn -w 2 -k gthread -t 120 app:app
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

from src.agent.agentic_workflow import GraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Initializing agent graph (loading models, compiling graph)...")
_builder = GraphBuilder()
GRAPH = _builder.graph
logger.info("Agent graph ready.")


# Helpers
def _as_text(message: Any) -> str:
    """Coerce an LLM message (AIMessage) or raw value into plain text."""
    if message is None:
        return ""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _build_sources(bundle: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Turn the context bundle into a clean source list the client can render
    as a references section. Pulls fields from each paper's metadata blob."""
    sources: List[Dict[str, Any]] = []
    for paper in bundle or []:
        md = paper.get("metadata") or {}
        sources.append({
            "s2_paper_id": paper.get("s2_paper_id"),
            "title": md.get("title"),
            "authors": md.get("authors"),
            "year": md.get("year"),
            "venue": md.get("venue"),
            "url": md.get("url"),
            "score": paper.get("score"),
        })
    return sources


def _jsonable_critique(critique: Any) -> Any:
    """CriticAgentOutput is a Pydantic model -> dict; pass through otherwise."""
    if critique is None:
        return None
    if hasattr(critique, "model_dump"):
        return critique.model_dump()
    return critique


# Routes
@app.post("/query")
def query():
    data = request.get_json(silent=True) or {}
    user_query = (data.get("query") or "").strip()
    if not user_query:
        return jsonify({"error": "Request body must include a non-empty 'query'."}), 400

    logger.info("Running graph for query: %s", user_query[:120])
    try:
        final_state = GRAPH.invoke({"query": user_query})
    except Exception:
        logger.exception("Graph execution failed")
        return jsonify({"error": "Failed to process the query."}), 500

    critique = final_state.get("critique")
    response = {
        "query": user_query,
        # The literature review -- this is the primary output the user reads.
        "answer": _as_text(final_state.get("draft_answer")),
        # Did the critic approve the draft? Lets the client flag unreviewed output.
        "passed": getattr(critique, "passed", None),
        # Deterministic + critic-judged metrics (already structured into panels).
        "metrics": final_state.get("metrics", {}),
        # Full critic report (issues + suggested fixes).
        "critique": _jsonable_critique(critique),
        # Sources for a references section / clickable citations.
        "sources": _build_sources(final_state.get("context_bundle")),
    }
    return jsonify(response)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)