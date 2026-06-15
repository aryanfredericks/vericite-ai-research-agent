"""Central configuration. All values can be overridden via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Semantic Scholar API ---
    s2_api_key: Optional[str] = os.getenv("S2_API_KEY")  # optional, raises your rate limit
    s2_base_url: str = os.getenv("S2_BASE_URL", "https://api.semanticscholar.org")
    # ~1 req/sec is the documented shared limit; 1.1s leaves headroom.
    s2_min_request_interval: float = float(os.getenv("S2_MIN_REQUEST_INTERVAL", "1.1"))
    s2_max_retries: int = int(os.getenv("S2_MAX_RETRIES", "5"))
    s2_timeout: float = float(os.getenv("S2_TIMEOUT", "30"))

    # --- Database ---
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/rag",
    )

    # --- Embeddings ---
    # Default is a small, fast local model (384 dims). EMBEDDING_DIM MUST match the model.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # --- Chunking (character based) ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- PDF handling ---
    pdf_dir: str = os.getenv("PDF_DIR", "./data/pdfs")
    download_pdfs: bool = _get_bool("DOWNLOAD_PDFS", True)
    pdf_timeout: float = float(os.getenv("PDF_TIMEOUT", "60"))

    # --- Ingestion controls ---
    papers_per_category: int = int(os.getenv("PAPERS_PER_CATEGORY", "100"))
    open_access_only: bool = _get_bool("OPEN_ACCESS_ONLY", True)
    min_citation_count: int = int(os.getenv("MIN_CITATION_COUNT", "0"))
    year_range: Optional[str] = os.getenv("YEAR_RANGE")  # e.g. "2018-2026"

    # The categories (search queries) you want to seed the corpus with.
    categories: List[str] = field(
        default_factory=lambda: [
            "retrieval augmented generation",
            "vector databases approximate nearest neighbor",
            "large language model agents",
            "dense retrieval text embeddings",
            "agentic ai",
            "generative ai",
            "machine learning",
            "deep learning"
        ]
    )

    def __post_init__(self) -> None:
        os.makedirs(self.pdf_dir, exist_ok=True)


settings = Settings()
