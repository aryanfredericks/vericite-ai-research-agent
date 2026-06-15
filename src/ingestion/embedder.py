"""Embedding generation.

Defaults to a local SentenceTransformer model so there are no API costs. To use
a different provider (OpenAI, Cohere, a Claude-backed service, ...), subclass and
override `embed`, keeping the returned vector dimension equal to EMBEDDING_DIM.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .config import settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        expected_dim: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.expected_dim = expected_dim or settings.embedding_dim
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_embedding_dimension()
            if dim != self.expected_dim:
                raise ValueError(
                    f"Model dimension {dim} != configured EMBEDDING_DIM "
                    f"{self.expected_dim}. Set EMBEDDING_DIM={dim} and recreate "
                    "the vector column before ingesting."
                )
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors -> cosine distance ready
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
