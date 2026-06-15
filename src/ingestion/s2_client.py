"""Thin Semantic Scholar Graph API client.

Handles:
  * authentication header (x-api-key) when a key is configured
  * a minimum-interval rate limiter (the public limit is ~1 req/sec)
  * retries with exponential backoff on 429 / 5xx (honours Retry-After)
  * token-based pagination for the /paper/search/bulk endpoint
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

from .config import Settings, settings

logger = logging.getLogger(__name__)

# Fields requested from the API. Keep this list lean -- extra fields slow responses.
DEFAULT_PAPER_FIELDS: List[str] = [
    "paperId",
    "corpusId",
    "title",
    "abstract",
    "year",
    "publicationDate",
    "venue",
    "citationCount",
    "referenceCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf",
    "externalIds",
    "publicationTypes",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "authors",
    "tldr",
    "url",
]

# The /paper/search/bulk endpoint supports a narrower field set than /paper/{id}
# and /paper/batch. In particular it rejects `tldr` (and would 400 the whole
# request). So bulk search uses this trimmed list; tldr can still be enriched
# later via get_papers_batch(), which does support it.
BULK_SEARCH_FIELDS: List[str] = [f for f in DEFAULT_PAPER_FIELDS if f != "tldr"]

_BATCH_MAX = 500  # /paper/batch accepts up to 500 ids per call


class RateLimiter:
    """Blocks until at least `min_interval` seconds have passed since the last call."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class SemanticScholarClient:
    def __init__(self, cfg: Settings = settings) -> None:
        self.base_url = cfg.s2_base_url.rstrip("/")
        self.timeout = cfg.s2_timeout
        self.max_retries = cfg.s2_max_retries
        self._limiter = RateLimiter(cfg.s2_min_request_interval)

        self._session = requests.Session()
        headers = {"User-Agent": "rag-ingestion/1.0"}
        if cfg.s2_api_key:
            headers["x-api-key"] = cfg.s2_api_key
        self._session.headers.update(headers)

    # ------------------------------------------------------------------ #
    # Low-level request with retry/backoff
    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            self._limiter.wait()
            try:
                resp = self._session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Network error %s/%s for %s: %s",
                    attempt, self.max_retries, url, exc,
                )
                time.sleep(self._backoff(attempt))
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                delay = float(resp.headers.get("Retry-After", 0)) or self._backoff(attempt)
                logger.warning("429 rate-limited; backing off %.1fs", delay)
                time.sleep(delay)
                continue
            if 500 <= resp.status_code < 600:
                logger.warning(
                    "Server %s on %s (attempt %s/%s)",
                    resp.status_code, url, attempt, self.max_retries,
                )
                time.sleep(self._backoff(attempt))
                continue
            # Other 4xx are not retryable -- surface the API's explanation,
            # which names the offending field/param (e.g. unsupported fields).
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {url} :: {detail}",
                response=resp,
            )

        raise RuntimeError(f"{method} {url} failed after {self.max_retries} attempts")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return float(min(2 ** attempt, 30))

    # ------------------------------------------------------------------ #
    # Public endpoints
    # ------------------------------------------------------------------ #
    def search_bulk(
        self,
        query: str,
        *,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        year: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False,
        min_citation_count: int = 0,
        sort: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield papers from /paper/search/bulk, transparently paginating via the
        continuation `token`. Stops after `limit` papers (if given).

        `sort` accepts "<field>:<asc|desc>" where field is paperId,
        publicationDate or citationCount.
        """
        fields = fields or BULK_SEARCH_FIELDS
        params: Dict[str, Any] = {"query": query, "fields": ",".join(fields)}
        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            # Presence-only flag: restricts results to papers with an OA PDF.
            params["openAccessPdf"] = ""
        if min_citation_count:
            params["minCitationCount"] = min_citation_count
        if sort:
            params["sort"] = sort

        yielded = 0
        token: Optional[str] = None
        while True:
            if token:
                params["token"] = token
            data = self._request("GET", "/graph/v1/paper/search/bulk", params=params)
            page = data.get("data") or []
            if not page:
                return
            for paper in page:
                yield paper
                yielded += 1
                if limit and yielded >= limit:
                    return
            token = data.get("token")
            if not token:
                return

    def get_papers_batch(
        self, paper_ids: List[str], fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Resolve full records for known paper ids via /paper/batch."""
        fields = fields or DEFAULT_PAPER_FIELDS
        out: List[Dict[str, Any]] = []
        for i in range(0, len(paper_ids), _BATCH_MAX):
            batch = paper_ids[i : i + _BATCH_MAX]
            data = self._request(
                "POST",
                "/graph/v1/paper/batch",
                params={"fields": ",".join(fields)},
                json={"ids": batch},
            )
            out.extend(p for p in data if p)  # null entries == not found
        return out