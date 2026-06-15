"""End-to-end ingestion orchestration.

Per category:
    bulk search -> normalize -> upsert metadata
                -> (if OA pdf) download -> extract -> chunk -> embed -> store chunks
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .chunker import TextChunker
from .config import Settings, settings
from .db import init_db, session_scope
from .embedder import Embedder
from .normalizer import PaperNormalizer
from .pdf_handler import PdfDownloader, PdfTextExtractor
from .repository import PaperRepository
from .s2_client import SemanticScholarClient

logger = logging.getLogger(__name__)

_MIN_USABLE_TEXT = 200  # chars; below this the extraction is treated as failed


class IngestionPipeline:
    def __init__(
        self, cfg: Settings = settings, embedder: Optional[Embedder] = None
    ) -> None:
        self.settings = cfg
        self.client = SemanticScholarClient(cfg)
        self.normalizer = PaperNormalizer()
        self.downloader = PdfDownloader(cfg)
        self.extractor = PdfTextExtractor()
        self.chunker = TextChunker(cfg.chunk_size, cfg.chunk_overlap)
        self.embedder = embedder or Embedder()

    def run(
        self,
        categories: Optional[List[str]] = None,
        papers_per_category: Optional[int] = None,
    ) -> Dict[str, int]:
        categories = categories or self.settings.categories
        per_category = papers_per_category or self.settings.papers_per_category
        stats = {
            "fetched": 0, "stored": 0, "embedded": 0,
            "skipped": 0, "no_pdf": 0, "errors": 0,
        }
        for category in categories:
            logger.info("=== Category: %s ===", category)
            self._ingest_category(category, per_category, stats)
        logger.info("Ingestion complete: %s", stats)
        return stats

    # ------------------------------------------------------------------ #
    def _ingest_category(self, category: str, limit: int, stats: Dict[str, int]) -> None:
        papers = self.client.search_bulk(
            query=category,
            limit=limit,
            year=self.settings.year_range,
            open_access_only=self.settings.open_access_only,
            min_citation_count=self.settings.min_citation_count,
            sort="citationCount:desc",
        )
        for raw in papers:
            stats["fetched"] += 1
            try:
                self._process_paper(raw, category, stats)
            except Exception:  # one bad paper shouldn't kill the run
                stats["errors"] += 1
                logger.exception("Failed processing paper %s", raw.get("paperId"))

    def _process_paper(self, raw: dict, category: str, stats: Dict[str, int]) -> None:
        paper = self.normalizer.normalize(raw, category=category)
        if paper is None:
            stats["skipped"] += 1
            return

        with session_scope() as session:
            repo = PaperRepository(session)

            if repo.is_processed(paper.s2_paper_id):
                stats["skipped"] += 1
                return

            paper_row = repo.upsert_paper(paper)  # metadata always stored

            if not (self.settings.download_pdfs and paper.has_pdf):
                stats["no_pdf"] += 1
                return

            pdf_path = self.downloader.download(paper)
            repo.mark_pdf(paper_row, pdf_path)
            if not pdf_path:
                stats["no_pdf"] += 1
                return

            text = self.extractor.extract(pdf_path)
            if not text or len(text) < _MIN_USABLE_TEXT:
                logger.debug("Unusable extraction for %s", paper.s2_paper_id)
                stats["no_pdf"] += 1
                return

            # Prepend title + abstract so they are embedded even if PDF parsing is weak.
            header = paper.title
            if paper.abstract:
                header = f"{header}\n\n{paper.abstract}"
            document = f"{header}\n\n{text}"

            chunks = self.chunker.split(document, base_metadata=paper.to_metadata())
            if not chunks:
                stats["no_pdf"] += 1
                return

            embeddings = self.embedder.embed([c.content for c in chunks])
            repo.replace_chunks(paper_row, chunks, embeddings)
            repo.mark_processed(paper_row, True)

            stats["stored"] += 1
            stats["embedded"] += len(chunks)
            logger.info("Stored %d chunks | %s", len(chunks), paper.title[:70])


def run_pipeline(
    categories: Optional[List[str]] = None,
    papers_per_category: Optional[int] = None,
) -> Dict[str, int]:
    """Convenience entrypoint: ensure schema exists, then ingest."""
    init_db()
    return IngestionPipeline().run(categories, papers_per_category)
