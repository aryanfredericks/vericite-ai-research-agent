from typing import Dict, List, TypedDict
from pydantic import BaseModel,Field

from typing import List, Optional
from pydantic import BaseModel, Field


class SubTaskFilters(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description=(
            "Filter to ONE category, only if the sub-query clearly belongs to it. "
            "Must be EXACTLY one of: 'retrieval augmented generation', "
            "'vector databases approximate nearest neighbor', "
            "'large language model agents', 'dense retrieval text embeddings', 'agentic ai', 'generative ai', 'machine learning', 'deep learning'."
            "Use null when unsure — a wrong value matches nothing."))
    year: Optional[int] = Field(
        default=None,
        description="Filter to a publication year, only if the query implies recency. Else null.")


class SubTask(BaseModel):
    sub_query: str = Field(
        description="A focused, keyword-style sub-query derived from the user's question.")
    top_k: int = Field(
        default=10,
        description="How many chunks to retrieve for this sub-query.")
    filters: Optional[SubTaskFilters] = Field(
        default=None,
        description="Optional metadata constraints. Prefer null and rely on sub_query text.")


class Plan(BaseModel):
    subtasks: List[SubTask] = Field(
        description="The user's question decomposed into focused retrieval sub-tasks.")

class GraphState(TypedDict):
    query : str
    plan : Plan | None
    retrieved : List[Dict]
    context_bundle : Dict
    draft_answer : str
    critique : Dict
    metrics : Dict
    
class RetrievedPaperChunk(TypedDict):
    chunk_index : str
    s2_paper_id : str
    content : str
    metadata : dict
    score : float
    sub_query : str
    
class Papers(TypedDict):
    s2_paper_id : str
    metadata : str
    score : float
    full_content : str
    
from pydantic import BaseModel, Field

class CriticAgentOutput(BaseModel):
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="The exact draft sentences that have no support in any context excerpt.")
    missing_citations: list[str] = Field(
        default_factory=list,
        description="Draft claims that are supported by the context but are not cited.")
    misattributions: list[str] = Field(
        default_factory=list,
        description="Draft claims cited to an id whose excerpt does not actually support them.")
    hallucinated_sources: list[str] = Field(
        default_factory=list,
        description="Cited ids that do not appear anywhere in the context.")
    overgeneralizations: list[str] = Field(
        default_factory=list,
        description="Claims stated more broadly or strongly than the cited source warrants.")
    coverage_gaps: list[str] = Field(
        default_factory=list,
        description="Relevant context the draft ignored, or parts of the question left unaddressed.")
    fixes: str = Field(
        description="A short, concrete summary of what the writer should revise.")
    passed: bool = Field(
        description="True only if there are no unsupported claims, misattributions, or hallucinated sources.")