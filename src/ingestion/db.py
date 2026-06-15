"""Database engine, session management, and schema/index initialization."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()
engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, future=True
)


def init_db() -> None:
    """Create the pgvector extension, all tables, and supporting indexes.

    Safe to call repeatedly (everything uses IF NOT EXISTS semantics).
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Import models so they are registered on Base before create_all().
    from . import models  # noqa: F401

    Base.metadata.create_all(engine)
    _create_indexes()
    logger.info("Database initialized.")


def _create_indexes() -> None:
    """Indexes that SQLAlchemy's create_all does not build for us.

    - HNSW for fast approximate vector search (requires pgvector >= 0.5.0).
    - GIN on the generated tsvector column for lexical / full-text search.
    - GIN on the JSONB metadata column for structured filtering.
    """
    statements = [
        """CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
               ON paper_chunks USING hnsw (embedding vector_cosine_ops)""",
        """CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
               ON paper_chunks USING gin (content_tsv)""",
        """CREATE INDEX IF NOT EXISTS idx_chunks_metadata
               ON paper_chunks USING gin (metadata jsonb_path_ops)""",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context: commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
