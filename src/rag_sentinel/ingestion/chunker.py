"""Structure-aware document chunker.

Uses LangChain's RecursiveCharacterTextSplitter under the hood but wraps it
so the rest of the pipeline only sees our internal Document type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .loaders import Document


@dataclass
class Chunk:
    """A single chunk ready for embedding and detection."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""          # Set by the vector store after hashing

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id!r}, chars={len(self.content)}, source={self.metadata.get('source', '?')!r})"


class Chunker:
    """Splits Documents into fixed-size overlapping chunks.

    Parameters
    ----------
    chunk_size:
        Approximate character count per chunk (not tokens — avoids tokenizer
        dependency; 400 chars ≈ 100 tokens for English prose).
    chunk_overlap:
        Character overlap between adjacent chunks (~15% of chunk_size).
    separators:
        Ordered list of separator strings tried in sequence.  Respects document
        structure (headings → paragraphs → sentences → words → chars).
    """

    DEFAULT_SEPARATORS = [
        "\n\n",   # paragraph boundary
        "\n",     # line break
        ". ",     # sentence end
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",       # character-level fallback
    ]

    def __init__(
        self,
        chunk_size: int = 1600,   # chars — approx 400 tokens
        chunk_overlap: int = 240,  # ~15% overlap
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, document: Document) -> list[Chunk]:
        """Split one Document into Chunks, carrying the parent metadata."""
        raw_chunks = self._recursive_split(document.content, self.separators)
        chunks: list[Chunk] = []

        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            meta = {**document.metadata, "chunk_index": i, "chunk_count": len(raw_chunks)}
            chunks.append(Chunk(content=text, metadata=meta))

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of Documents, assigning sequential IDs."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split(doc))

        # Assign deterministic IDs based on source + index
        for i, chunk in enumerate(all_chunks):
            source = chunk.metadata.get("source", "unknown")
            chunk.chunk_id = f"{source}::{chunk.metadata.get('chunk_index', i)}"

        logger.info(f"Chunker: produced {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text, trying each separator in order."""
        if len(text) <= self.chunk_size:
            return [text]

        # Find the first separator that appears in the text
        chosen_sep = ""
        remaining_seps: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                chosen_sep = sep
                remaining_seps = separators[i + 1:]
                break

        splits = text.split(chosen_sep) if chosen_sep else list(text)

        # Merge splits back up to chunk_size with overlap
        return self._merge_splits(splits, chosen_sep, remaining_seps)

    def _merge_splits(
        self,
        splits: list[str],
        separator: str,
        remaining_seps: list[str],
    ) -> list[str]:
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part) + len(separator)

            # If adding this part would exceed chunk_size, flush current buffer
            if current_len + part_len > self.chunk_size and current_parts:
                chunk_text = separator.join(current_parts)
                # Recursively split if still too large
                if len(chunk_text) > self.chunk_size and remaining_seps:
                    chunks.extend(self._recursive_split(chunk_text, remaining_seps))
                else:
                    chunks.append(chunk_text)

                # Keep overlap: drop parts from the front until under overlap limit
                while current_parts and current_len > self.chunk_overlap:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + len(separator)

            current_parts.append(part)
            current_len += part_len

        # Flush remaining
        if current_parts:
            chunk_text = separator.join(current_parts)
            if len(chunk_text) > self.chunk_size and remaining_seps:
                chunks.extend(self._recursive_split(chunk_text, remaining_seps))
            else:
                chunks.append(chunk_text)

        return chunks
