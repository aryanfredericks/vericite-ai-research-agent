from typing import List, TypedDict, Optional, Dict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from ingestion.db import session_scope
from ingestion.embedder import Embedder
from ingestion.retriever import HybridRetriever
from utils.load_critic_llm import LoadCriticLLM
from utils.load_planner_llm import LoadPlannerLLM
from utils.load_writer_llm import LoadWriterLLM
from utils.structured_outputs import GraphState, RetrievedPaperChunk, Papers
from prompts.planner_prompt import PLANNER_PROMPT
from prompts.writer_prompt import WRITER_PROMPT
from prompts.critic_prompt import CRITIC_PROMPT

from collections import defaultdict

import re
from utils import evaluation_metrics as em

class GraphBuilder():
    def __init__(self):
        self.planner_llm = LoadPlannerLLM(model_name="qwen/qwen3-32b").get_model()
        self.writer_llm  = LoadWriterLLM(model_name="llama-3.3-70b-versatile").get_model()
        self.critic_llm  = LoadCriticLLM(model_name="llama-3.3-70b-versatile").get_model()
        self.embedder    = Embedder()
        self.graph       = self.build_graph()
        
        
    def _planner_node(self, state : GraphState) -> dict:
        plan = self.planner_llm.invoke([
            PLANNER_PROMPT,
            HumanMessage(content = state['query'])
        ])
        return {
            'plan' : plan,
        }
    
    def _retrieval_node(self, state: GraphState) -> dict:
        plan = state['plan']
        hits: list[RetrievedPaperChunk] = []
        with session_scope() as session:
            retriever = HybridRetriever(session)
            for task in plan.subtasks:
                # SubTaskFilters object -> plain dict, dropping null keys.
                flt = task.filters.model_dump(exclude_none=True) if task.filters else None
                for rc in retriever.search(
                    query=task.sub_query,
                    top_k=task.top_k,
                    metadata_filter=flt or None,   # {} -> None so an empty filter doesn't restrict
                ):
                    hits.append(
                        RetrievedPaperChunk(
                            chunk_index=rc.chunk_index,
                            s2_paper_id=rc.s2_paper_id,
                            content=rc.content,
                            metadata=rc.metadata,
                            score=rc.score,
                            sub_query=task.sub_query,
                        )
                    )

        print(f"[retrieval] subtasks={len(plan.subtasks)} hits={len(hits)}")
        for t in plan.subtasks:
            flt = t.filters.model_dump(exclude_none=True) if t.filters else None
            print(f"   sub_query={t.sub_query!r} filters={flt}")

        return {'retrieved': hits}
    
    def _aggregator_node(self, state: GraphState) -> dict:
        retrieved: List[RetrievedPaperChunk] = state["retrieved"]
        by_paper: dict[str, dict[int, RetrievedPaperChunk]] = defaultdict(dict)

        for rc in retrieved:
            slot = by_paper[rc["s2_paper_id"]]
            kept = slot.get(rc["chunk_index"])
            if kept is None or rc["score"] > kept["score"]:
                slot[rc["chunk_index"]] = rc

        papers: List[Papers] = []
        for pid, chunk_map in by_paper.items():
            ordered = sorted(chunk_map.values(), key=lambda c: c["chunk_index"])

            parts: List[str] = []
            prev: int | None = None
            for c in ordered:
                if prev is not None and c["chunk_index"] != prev + 1:
                    parts.append("[...]")
                parts.append(c["content"])
                prev = c["chunk_index"]
            full_content = "\n\n".join(parts)

            papers.append(
                Papers(
                    s2_paper_id=pid,
                    full_content=full_content,
                    score=max(c["score"] for c in ordered),
                    metadata=ordered[0]["metadata"],
                )
            )

        papers.sort(key=lambda p: p["score"], reverse=True)
        return {"context_bundle": papers}
    
    def _writer_node(self, state : GraphState) -> dict:
        draft_ans = self.writer_llm.invoke([
            WRITER_PROMPT,
            HumanMessage(
                content=f"""
                Research Question : 
                {state['query']}
                
                Context Bundle : 
                {render_context(state['context_bundle'])}
                """,
            ),
        ])
        
        return {
            'draft_answer' : draft_ans
        }
    
    def _critic_node(self, state : GraphState) -> dict:
        drafted_ans = state["draft_answer"]
        critique = self.critic_llm.invoke([
            CRITIC_PROMPT,
            HumanMessage(content=f"Write your critic for the following drafted answer {drafted_ans}")
        ])
        return {
            'critique' : critique
        }
    
    def _eval_node(self, state: GraphState) -> dict:
        query = state["query"]
        draft_msg = state.get("draft_answer")
        draft = getattr(draft_msg, "content", draft_msg) or ""
        if not isinstance(draft, str):
            draft = str(draft)
        bundle: List[Papers] = state.get("context_bundle", []) or []
        bundle_ids = [p["s2_paper_id"] for p in bundle]

        # --- grounding panel (draft vs context) ---
        grounding = {}
        grounding.update(em.citation_validity(draft, bundle_ids))
        grounding.update(em.source_utilization(draft, bundle_ids))
        grounding["citation_density"] = em.citation_density(
            draft, grounding["total_citations"]
        )

        # --- relevance panel (draft vs query) ---
        relevance = em.query_draft_relevance(query, draft, self.embedder)

        metrics = {"grounding": grounding, "relevance": relevance}

        # --- critic panel (model-judged -> kept SEPARATE, not blended in) ---
        critique = state.get("critique")
        if critique is not None:
            metrics["critique"] = {
                "passed": critique.passed,
                "issue_score": (
                    len(critique.unsupported_claims) * 3
                    + len(critique.missing_citations) * 2
                    + len(critique.misattributions) * 3
                    + len(critique.hallucinated_sources) * 3
                    + len(critique.overgeneralizations)
                    + len(critique.coverage_gaps)
                ),
            }

        return {"metrics": metrics}
        
    def build_graph(self):
        b = StateGraph(GraphState)
        b.add_node("planner", self._planner_node)
        b.add_node("retrieval", self._retrieval_node)
        b.add_node("aggregator", self._aggregator_node)
        b.add_node("writer", self._writer_node)
        b.add_node("critic", self._critic_node)
        b.add_node("evaluation", self._eval_node)

        b.add_edge(START, "planner")
        b.add_edge("planner", "retrieval")
        b.add_edge("retrieval", "aggregator")
        b.add_edge("aggregator", "writer")
        b.add_edge("writer", "critic")
        b.add_edge("critic", "evaluation")
        b.add_edge("evaluation", END)

        graph = b.compile()
        return graph
    
    def __call__(self):
        return self.graph


# top level, outside the class
def render_context(bundle, max_chars_per_paper: int = 1500) -> str:
    blocks = []
    for p in bundle:
        md = p.get("metadata") or {}
        header = f'[SOURCE id={p["s2_paper_id"]} | "{md.get("title","")}" | {md.get("year","")}]'
        body = p["full_content"][:max_chars_per_paper]
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)