"""Character-based recursive chunker with overlap.

Strategy: split on paragraph breaks, then sentences, then hard-wrap any unit
still larger than chunk_size, then greedily pack units into windows. Each window
carries an overlap tail from the previous one so context isn't lost at borders.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .schemas import Chunk


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initializes the text chunker

        Args:
            chunk_size (int): maximum charachter length of a single chunk
            chunk_overlap (int): maximum charachter length that can be overlapped between 2 seperate chunks

        Raises:
            ValueError: when 'chunk_overlap' is greater than 'chunk_size'
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        text: str,
        *,
        base_metadata: Optional[Dict[str, Any]] = None,
        section: Optional[str] = None,
    ) -> List[Chunk]:
        """Splits the input text into seperate chunks (using [Chunk] object)

        Args:
            text (str): Raw source string to be chunked
            base_metadata (Optional[Dict[str, Any]], optional): dictionary of metadata to be store alongside a chunk
            section (Optional[str], optional): string identifier that indicates the document section name

        Returns:
            List[Chunk]: A list of 'Chunk' objects, each of them containing contextual metadata and sequential indices.
        """
        base_metadata = base_metadata or {}
        units = self._split_units(text)
        chunks: List[Chunk] = []
        for i, content in enumerate(self._pack(units)):
            meta = dict(base_metadata)
            meta["chunk_index"] = i
            chunks.append(
                Chunk(chunk_index=i, content=content, section=section, metadata=meta)
            )
        return chunks

    # ------------------------------------------------------------------ #
    def _split_units(self, text: str) -> List[str]:
        """Recursively breaks text down into strings smaller than chunk_size.

        Progresses from paragraph breaks down to sentence boundaries, and falls back
        to hard character slicing if an element still violates the `chunk_size` limit.

        Args:
            text: The text segment to break into manageable pieces.

        Returns:
            A list of string units, where every single string is guaranteed to be 
            less than or equal to `self.chunk_size`.
        """
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        units: List[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                units.append(para)
            else:
                units.extend(self._split_sentences(para))

        final: List[str] = []
        for unit in units:
            if len(unit) <= self.chunk_size:
                final.append(unit)
            else:
                final.extend(self._hard_split(unit))
        return final

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Splits a text string into individual sentences using standard punctuation boundaries.

        Uses a positive lookbehind assertion to split on spaces immediately following
        periods, exclamation marks, or question marks.

        Args:
            text: The text block (typically a single paragraph) to split.

        Returns:
            A list of stripped sentence strings.
        """
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _hard_split(self, text: str) -> List[str]:
        """Force-splits an oversized text string into chunks using mathematical slicing.

        This acts as a failsafe when a single sentence exceeds `chunk_size`. It uses
        a stride pattern calculated by subtracting overlap from total chunk size.

        Args:
            text: The oversized string unit.

        Returns:
            A list of string slices adhering to the step boundaries.
        """
        step = self.chunk_size - self.chunk_overlap
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    def _pack(self, units: List[str]) -> List[str]:
        """Greedily aggregates text units into chunk windows up to chunk_size.

        When a window fills up, the method takes a snapshot of the current chunk,
        slices off its tail end matching `chunk_overlap`, and uses that tail as the
        starting prefix for the next chunk window to preserve contextual continuity.

        Args:
            units: A pre-split list of strings where each element is smaller 
                than `chunk_size`.

        Returns:
            A list of packed text strings with overlap boundaries applied.
        """
        chunks: List[str] = []
        current = ""
        for unit in units:
            if not current:
                current = unit
            elif len(current) + 1 + len(unit) <= self.chunk_size:
                current = f"{current} {unit}"
            else:
                chunks.append(current)
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{tail} {unit}".strip() if tail else unit
        if current:
            chunks.append(current)
        return chunks
