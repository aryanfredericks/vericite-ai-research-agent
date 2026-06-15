"""Deterministic (model-free, reproducible) evaluation metrics.

Each function takes plain inputs and returns a small dict, so they can be unit
tested without a graph, a session, or an LLM. The embedder is passed in rather
than imported, so it can be shared (and mocked) by the caller.

Grounding is measured against the CONTEXT BUNDLE -- i.e. what was retrieved --
not against ground truth. These numbers tell you whether the draft is faithful
to its sources, not whether those sources are correct.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence

# Matches in-text citations like [<paper_id>]. Captures the bracket contents so
# we can also split combined citations and skip gap markers ([...]) the
# aggregator may have inserted.
_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE_RE = re.compile(r"[.!?]+")


def extract_citations(draft: str) -> List[str]:
    """Return every in-text citation token, one entry per occurrence.

    Handles `[id1][id2]` (separate brackets) and `[id1, id2]` (combined), and
    skips empty brackets and ellipsis gap markers like `[...]`.
    """
    tokens: List[str] = []
    for raw in _CITATION_RE.findall(draft):
        for part in re.split(r"[,\s;]+", raw.strip()):
            part = part.strip()
            if not part or set(part) <= {"."}:  # '' or '...' gap marker
                continue
            tokens.append(part)
    return tokens


def citation_validity(draft: str, bundle_ids: Sequence[str]) -> Dict[str, Any]:
    """How many of the draft's citations point at sources actually in the bundle.

    The one objective grounding check: an id is either in the context or it is
    not. A rate below 1.0 means the writer cited a source it was never given --
    a hallucinated citation. (rate is vacuously 1.0 when there are no citations;
    read `total_citations` to catch the "didn't cite anything" case.)
    """
    valid = set(bundle_ids)
    cites = extract_citations(draft)
    total = len(cites)
    valid_occurrences = sum(1 for c in cites if c in valid)
    unique_cited = set(cites)
    return {
        "total_citations": total,
        "unique_cited_sources": len(unique_cited),
        "citation_validity_rate": (valid_occurrences / total) if total else 1.0,
        "hallucinated_source_ids": sorted(unique_cited - valid),
    }


def source_utilization(draft: str, bundle_ids: Sequence[str]) -> Dict[str, Any]:
    """Fraction of retrieved papers the draft actually cites.

    Low utilization means the writer ignored most of what retrieval supplied --
    either the draft is thin or retrieval pulled irrelevant papers.
    """
    valid = set(bundle_ids)
    if not valid:
        return {"source_utilization": 0.0, "papers_cited": 0, "papers_available": 0}
    cited = set(extract_citations(draft)) & valid
    return {
        "source_utilization": len(cited) / len(valid),
        "papers_cited": len(cited),
        "papers_available": len(valid),
    }


def citation_density(draft: str, total_citations: int) -> float:
    """Citations per sentence. Very low = under-cited; very high = padding."""
    n_sentences = max(1, len(_SENTENCE_RE.findall(draft)))
    return total_citations / n_sentences


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, robust to non-normalized vectors and zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def query_draft_relevance(query: str, draft: str, embedder) -> Dict[str, Any]:
    """Cosine similarity between the query and the whole draft.

    A crude "did it answer what was asked" signal. It is a proxy: similar text
    can still be wrong, so use it for relative comparison across runs, not as an
    absolute correctness score. `embedder` must be the SAME model used at
    ingestion for the number to mean anything.
    """
    if not draft.strip():
        return {"query_draft_similarity": 0.0}
    q_vec, d_vec = embedder.embed([query, draft])
    return {"query_draft_similarity": _cosine(q_vec, d_vec)}