"""Internal data-transfer objects passed between pipeline stages.

These are intentionally decoupled from the ORM models so the API/normalize/chunk
stages don't need a database session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class NormalizedPaper:
    s2_paper_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    publication_date: Optional[date] = None
    venue: Optional[str] = None
    corpus_id: Optional[int] = None
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    is_open_access: bool = False
    open_access_url: Optional[str] = None
    open_access_status: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    authors: List[Dict[str, Any]] = field(default_factory=list)
    fields_of_study: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    tldr: Optional[str] = None
    category: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_pdf(self) -> bool:
        """Used to check if a paper has an openly accessible pdf option

        Returns:
            bool: True or False
        """
        return bool(self.open_access_url) or bool(self.arxiv_id)

    def to_metadata(self) -> Dict[str, Any]:
        """Compact metadata copied onto every chunk so the hybrid retriever can
        filter (category, year, fields_of_study) and re-rank (citation_count)
        without joining back to the papers table."""
        return {
            "s2_paper_id": self.s2_paper_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "authors": [a.get("name") for a in self.authors if a.get("name")],
            "fields_of_study": self.fields_of_study,
            "doi": self.doi,
            "category": self.category,
            "citation_count": self.citation_count,
            "url": self.url,
        }


@dataclass
class Chunk:
    chunk_index: int
    content: str
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Count the number of charachters in a chunk

        Returns:
            int: number of charachters
        """
        return len(self.content)
