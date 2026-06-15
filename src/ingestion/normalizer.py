"""Convert raw Semantic Scholar paper JSON into the internal NormalizedPaper DTO."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from .schemas import NormalizedPaper

logger = logging.getLogger(__name__)


class PaperNormalizer:
    def normalize(
        self, raw: Dict[str, Any], category: Optional[str] = None
    ) -> Optional[NormalizedPaper]:
        paper_id = raw.get("paperId")
        title = raw.get("title")
        if not paper_id or not title:
            logger.debug("Dropping paper with missing id/title (%s).", paper_id)
            return None

        oa = raw.get("openAccessPdf") or {}
        external = raw.get("externalIds") or {}
        tldr = raw.get("tldr") or {}

        return NormalizedPaper(
            s2_paper_id=paper_id,
            title=title.strip(),
            abstract=raw.get("abstract") or None,
            year=raw.get("year"),
            publication_date=self._parse_date(raw.get("publicationDate")),
            venue=raw.get("venue") or None,
            corpus_id=raw.get("corpusId"),
            citation_count=raw.get("citationCount") or 0,
            reference_count=raw.get("referenceCount") or 0,
            influential_citation_count=raw.get("influentialCitationCount") or 0,
            is_open_access=bool(raw.get("isOpenAccess")),
            open_access_url=oa.get("url") or None,
            open_access_status=oa.get("status") or None,
            doi=external.get("DOI"),
            url=raw.get("url"),
            authors=[
                {"authorId": a.get("authorId"), "name": a.get("name")}
                for a in (raw.get("authors") or [])
            ],
            fields_of_study=self._fields_of_study(raw),
            publication_types=raw.get("publicationTypes") or [],
            tldr=tldr.get("text") if isinstance(tldr, dict) else None,
            category=category,
            raw=raw,
        )

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _fields_of_study(raw: Dict[str, Any]) -> List[str]:
        fos = raw.get("fieldsOfStudy")
        if fos:
            return list(fos)
        s2fos = raw.get("s2FieldsOfStudy") or []
        return sorted({f.get("category") for f in s2fos if f.get("category")})
