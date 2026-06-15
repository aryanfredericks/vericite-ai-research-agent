"""SQLAlchemy ORM models: Paper (metadata) and PaperChunk (text + embedding)."""
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from .config import settings
from .db import Base


class Paper(Base):
    """One row per paper. Always stored, even when no PDF is available, so the
    metadata can still enrich retrieval / be re-processed later."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True)
    s2_paper_id = Column(String, unique=True, nullable=False, index=True)
    corpus_id = Column(BigInteger, nullable=True)

    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    year = Column(Integer, nullable=True)
    publication_date = Column(Date, nullable=True)
    venue = Column(Text, nullable=True)

    citation_count = Column(Integer, default=0)
    reference_count = Column(Integer, default=0)
    influential_citation_count = Column(Integer, default=0)

    is_open_access = Column(Boolean, default=False)
    open_access_url = Column(Text, nullable=True)
    open_access_status = Column(String, nullable=True)

    doi = Column(String, nullable=True)
    url = Column(Text, nullable=True)

    authors = Column(JSONB, nullable=True)            # [{"authorId":.., "name":..}]
    fields_of_study = Column(JSONB, nullable=True)    # ["Computer Science", ...]
    publication_types = Column(JSONB, nullable=True)  # ["JournalArticle", ...]
    tldr = Column(Text, nullable=True)

    category = Column(String, nullable=True, index=True)  # ingestion query bucket

    pdf_downloaded = Column(Boolean, default=False)
    pdf_path = Column(Text, nullable=True)
    processed = Column(Boolean, default=False)  # chunks + embeddings persisted

    raw = Column(JSONB, nullable=True)  # original API payload, for reprocessing

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks = relationship(
        "PaperChunk", back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Paper {self.s2_paper_id} {self.title[:40]!r}>"


class PaperChunk(Base):
    """One row per text chunk, holding the embedding, a denormalized metadata
    blob (for hybrid filtering/ranking) and a generated tsvector for lexical
    full-text search."""

    __tablename__ = "paper_chunks"
    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index", name="uq_paper_chunk_idx"),
    )

    id = Column(Integer, primary_key=True)
    paper_id = Column(
        Integer,
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    s2_paper_id = Column(String, nullable=False, index=True)

    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
    section = Column(String, nullable=True)

    embedding = Column(Vector(settings.embedding_dim), nullable=False)

    # `metadata` is reserved on declarative classes -> map attr `metadata_`.
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Generated, STORED tsvector for lexical search (the GIN index lives on this).
    content_tsv = Column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(content, ''))", persisted=True),
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    paper = relationship("Paper", back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PaperChunk {self.s2_paper_id}#{self.chunk_index}>"
