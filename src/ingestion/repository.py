"""Database access layer. Keeps SQL/ORM concerns out of the pipeline."""
from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .models import Paper, PaperChunk
from .schemas import Chunk, NormalizedPaper

logger = logging.getLogger(__name__)


class PaperRepository:
    def __init__(self, session) -> None:
        self.session = session

    def upsert_paper(self, paper: NormalizedPaper) -> Paper:
        """Insert or update a paper row keyed on s2_paper_id, returning the ORM object."""
        values = {
            "s2_paper_id": paper.s2_paper_id,
            "corpus_id": paper.corpus_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "publication_date": paper.publication_date,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
            "reference_count": paper.reference_count,
            "influential_citation_count": paper.influential_citation_count,
            "is_open_access": paper.is_open_access,
            "open_access_url": paper.open_access_url,
            "open_access_status": paper.open_access_status,
            "doi": paper.doi,
            "url": paper.url,
            "authors": paper.authors,
            "fields_of_study": paper.fields_of_study,
            "publication_types": paper.publication_types,
            "tldr": paper.tldr,
            "category": paper.category,
            "raw": paper.raw,
        }
        stmt = pg_insert(Paper).values(**values)
        update_cols = {k: stmt.excluded[k] for k in values if k != "s2_paper_id"}
        stmt = stmt.on_conflict_do_update(
            index_elements=["s2_paper_id"], set_=update_cols
        ).returning(Paper.id)

        paper_id = self.session.execute(stmt).scalar_one()
        self.session.flush()
        return self.session.get(Paper, paper_id)

    def is_processed(self, s2_paper_id: str) -> bool:
        processed = self.session.execute(
            select(Paper.processed).where(Paper.s2_paper_id == s2_paper_id)
        ).scalar_one_or_none()
        return bool(processed)

    def replace_chunks(
        self,
        paper_row: Paper,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        """Delete any existing chunks for this paper, then insert the new set.
        Makes re-running ingestion idempotent."""
        self.session.query(PaperChunk).filter(
            PaperChunk.paper_id == paper_row.id
        ).delete(synchronize_session=False)

        self.session.add_all(
            PaperChunk(
                paper_id=paper_row.id,
                s2_paper_id=paper_row.s2_paper_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                char_count=chunk.char_count,
                section=chunk.section,
                embedding=embedding,
                metadata_=chunk.metadata,
            )
            for chunk, embedding in zip(chunks, embeddings)
        )

    def mark_pdf(self, paper_row: Paper, pdf_path: Optional[str]) -> None:
        paper_row.pdf_downloaded = pdf_path is not None
        paper_row.pdf_path = pdf_path

    def mark_processed(self, paper_row: Paper, processed: bool = True) -> None:
        paper_row.processed = processed
