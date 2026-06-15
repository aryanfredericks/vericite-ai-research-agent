"""Download open-access PDFs and extract clean text from them.

`PdfDownloader` tries the Semantic Scholar openAccessPdf URL first, then an
arXiv fallback if an arXiv id is present. Many "OA" links point at publisher
landing pages rather than a raw PDF, so we validate the %PDF magic header and
discard anything that isn't a real PDF.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

import requests

from .config import Settings, settings
from .schemas import NormalizedPaper

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


class PdfDownloader:
    def __init__(self, cfg: Settings = settings) -> None:
        self.pdf_dir = cfg.pdf_dir
        self.timeout = cfg.pdf_timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "rag-ingestion/1.0"})

    def candidate_urls(self, paper: NormalizedPaper) -> List[str]:
        urls: List[str] = []
        if paper.open_access_url:
            urls.append(paper.open_access_url)
        return urls

    def download(self, paper: NormalizedPaper) -> Optional[str]:
        """Return a local path to a validated PDF, or None if none could be fetched."""
        target = os.path.join(self.pdf_dir, f"{self._safe(paper.s2_paper_id)}.pdf")
        if os.path.exists(target) and self._is_pdf(target):
            return target

        for url in self.candidate_urls(paper):
            try:
                resp = self._session.get(
                    url, timeout=self.timeout, allow_redirects=True
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.debug("PDF fetch failed (%s): %s", url, exc)
                continue

            content = resp.content
            if not content.startswith(_PDF_MAGIC):
                logger.debug("Not a PDF (no %%PDF header): %s", url)
                continue
            with open(target, "wb") as fh:
                fh.write(content)
            return target

        return None

    @staticmethod
    def _safe(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

    @staticmethod
    def _is_pdf(path: str) -> bool:
        try:
            with open(path, "rb") as fh:
                return fh.read(4) == _PDF_MAGIC
        except OSError:
            return False


class PdfTextExtractor:
    """Extracts and lightly cleans text using PyMuPDF (fitz)."""

    def extract(self, path: str) -> str:
        import fitz  # PyMuPDF; imported lazily so the package loads without it

        parts: List[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                parts.append(page.get_text("text"))
        return self._clean("\n".join(parts))

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("\x00", " ")
        # Join words hyphenated across line breaks: "infor-\nmation" -> "information"
        text = re.sub(r"-\n(\w)", r"\1", text)
        # Collapse single newlines (intra-paragraph) into spaces; keep blank lines.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
