"""BONUS: a hybrid retriever to prove the schema supports hybrid RAG.

Combines pgvector cosine similarity with PostgreSQL full-text search and fuses
the two ranked lists with Reciprocal Rank Fusion (RRF). You'll use something
like this in the agent/query phase -- it's included here only to show how the
stored embeddings + tsvector + JSONB metadata work together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select

from .embedder import Embedder
from .models import PaperChunk


@dataclass
class RetrievedChunk:
    """A single search result returned by :class:`HybridRetriever`.

    Attributes:
        chunk_id: Primary key of the chunk in ``paper_chunks``.
        s2_paper_id: Semantic Scholar id of the paper the chunk belongs to,
            useful for grouping results or looking up full paper metadata.
        content: The chunk's text, ready to drop into a prompt context window.
        metadata: The chunk's denormalized JSONB metadata (title, year,
            authors, fields_of_study, category, citation_count, ...). Use this
            for citing sources or post-filtering without a second DB round trip.
        score: The fused RRF score. Higher is more relevant. Scores are only
            meaningful relative to each other within one ``search`` call -- they
            are not probabilities and not comparable across queries.
    """

    chunk_id: int
    s2_paper_id: str
    chunk_index: int           
    section: str | None        
    content: str
    metadata: Dict[str, Any]
    score: float


class HybridRetriever:
    """Retrieves relevant chunks by combining semantic and keyword search.

    Two independent searches are run over ``paper_chunks`` and merged:

    * **Vector search** -- embeds the query and finds the nearest chunk
      embeddings by cosine distance (semantic similarity; good at paraphrase
      and synonyms).
    * **Lexical search** -- PostgreSQL full-text search over the generated
      ``content_tsv`` column (exact term / phrase matching; good at acronyms,
      identifiers, and rare words the embedder may blur).

    Their ranked outputs are combined with Reciprocal Rank Fusion so a chunk
    that scores well in *either* method surfaces, and chunks strong in *both*
    rise to the top. This balances recall (vector) against precision (lexical)
    without hand-tuned score weighting.
    """

    def __init__(
        self,
        session,
        embedder: Optional[Embedder] = None,
        rrf_k: int = 60,
        candidate_pool: int = 50,
    ) -> None:
        """Configure the retriever.

        Args:
            session: An open SQLAlchemy session bound to the RAG database. The
                retriever does not open or close it; the caller owns its
                lifecycle (e.g. via ``session_scope()``).
            embedder: Embedder used to vectorize the query. Must be the *same*
                model used at ingestion time, or vector distances are
                meaningless. Defaults to a new :class:`Embedder` from settings.
            rrf_k: Reciprocal Rank Fusion smoothing constant. Larger values
                flatten the contribution of rank position (so being #1 vs #5
                matters less); the conventional default is 60.
            candidate_pool: How many results to pull from *each* search before
                fusing. Larger pools improve recall at the cost of latency.
                The final result count is capped separately by ``top_k``.
        """
        self.session = session
        self.embedder = embedder or Embedder()
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Run hybrid retrieval and return the best chunks for ``query``.

        Executes the vector and lexical searches, fuses their rankings with
        RRF, and returns the top results. This is the only method the agent
        needs to call.

        Args:
            query: The natural-language search string (e.g. "current advances
                in RAG systems").
            top_k: Maximum number of chunks to return after fusion.
            metadata_filter: Optional JSONB containment filter applied to both
                searches before ranking. Keys/values must match the chunk
                metadata exactly, e.g. ``{"category": "vector databases"}`` or
                ``{"year": 2023}``. ``None`` searches the whole corpus.

        Returns:
            Up to ``top_k`` :class:`RetrievedChunk` objects, ordered by fused
            relevance (highest ``score`` first). May be empty if nothing
            matches the filter or query.
        """
        vector_hits = self._vector_search(query, metadata_filter)
        lexical_hits = self._lexical_search(query, metadata_filter)
        return self._reciprocal_rank_fusion(vector_hits, lexical_hits)[:top_k]

    
    def _apply_filter(self, stmt, metadata_filter: Optional[Dict[str, Any]]):
        """Attach the optional metadata filter to a SELECT statement.

        Adds a JSONB containment (``@>``) predicate when a filter is supplied,
        which restricts results to chunks whose ``metadata`` contains the given
        key/value pairs. This predicate is served by the GIN index on the
        metadata column, so filtering stays cheap. Returns the statement
        unchanged when ``metadata_filter`` is ``None``.

        Args:
            stmt: The SQLAlchemy ``select(PaperChunk)`` to augment.
            metadata_filter: Key/value pairs to require, or ``None``.

        Returns:
            The (possibly filtered) statement, for chaining.
        """
        if metadata_filter:
            # JSONB containment (@>) -> uses the GIN metadata index.
            stmt = stmt.where(PaperChunk.metadata_.contains(metadata_filter))
        return stmt

    def _vector_search(
        self, query: str, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[PaperChunk]:
        """Semantic search: nearest chunk embeddings to the query embedding.

        Embeds ``query`` with the configured embedder, then orders chunks by
        cosine distance (``<=>``) using the HNSW vector index, returning the
        closest ``candidate_pool`` chunks. Any metadata filter is applied first.

        Args:
            query: The search string to embed.
            metadata_filter: Optional containment filter (see ``_apply_filter``).

        Returns:
            Up to ``candidate_pool`` :class:`PaperChunk` rows, nearest first.
        """
        qvec = self.embedder.embed_one(query)
        distance = PaperChunk.embedding.cosine_distance(qvec)
        stmt = self._apply_filter(select(PaperChunk), metadata_filter)
        stmt = stmt.order_by(distance).limit(self.candidate_pool)
        return list(self.session.execute(stmt).scalars())

    def _lexical_search(
        self, query: str, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[PaperChunk]:
        """Keyword search: full-text matches ranked by term relevance.

        Parses ``query`` with ``websearch_to_tsquery`` (Google-style syntax:
        quotes for phrases, ``-`` to exclude), matches it against the generated
        ``content_tsv`` column via the ``@@`` operator (served by the GIN
        index), and ranks hits with ``ts_rank_cd``. Any metadata filter is
        applied first.

        Args:
            query: The search string to parse into a tsquery.
            metadata_filter: Optional containment filter (see ``_apply_filter``).

        Returns:
            Up to ``candidate_pool`` :class:`PaperChunk` rows, best-ranked
            first. Only chunks containing the query terms are included, so this
            list can be shorter than the vector list (or empty).
        """
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(PaperChunk.content_tsv, tsquery)
        stmt = select(PaperChunk).where(PaperChunk.content_tsv.op("@@")(tsquery))
        stmt = self._apply_filter(stmt, metadata_filter)
        stmt = stmt.order_by(rank.desc()).limit(self.candidate_pool)
        return list(self.session.execute(stmt).scalars())

    def _reciprocal_rank_fusion(
        self, vector_hits: List[PaperChunk], lexical_hits: List[PaperChunk]
    ) -> List[RetrievedChunk]:
        """Merge two ranked lists into one using Reciprocal Rank Fusion.

        Each chunk earns ``1 / (rrf_k + position)`` from every list it appears
        in (position is 0-based: the top hit contributes the most), and the
        contributions are summed across both lists. A chunk found by both
        searches therefore accumulates two contributions and outranks chunks
        found by only one. RRF depends only on rank order, not the raw cosine
        distances or text-rank scores, so the two incomparable scales never
        need normalizing.

        Args:
            vector_hits: Chunks from ``_vector_search``, nearest first.
            lexical_hits: Chunks from ``_lexical_search``, best-ranked first.

        Returns:
            All distinct chunks across both lists as :class:`RetrievedChunk`
            objects, sorted by descending fused score. ``search`` slices this
            down to ``top_k``.
        """
        scores: Dict[int, float] = {}
        objs: Dict[int, PaperChunk] = {}
        for ranked in (vector_hits, lexical_hits):
            for rank, chunk in enumerate(ranked):
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                objs[chunk.id] = chunk

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=cid,
                s2_paper_id=objs[cid].s2_paper_id,
                content=objs[cid].content,
                metadata=objs[cid].metadata_ or {},
                score=score,
                chunk_index=objs[cid].chunk_index,
                section=objs[cid].section
            )
            for cid, score in ordered
        ]