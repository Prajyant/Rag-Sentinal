"""Document loaders for PDF, HTML, plain text, and Markdown.

All loaders share the BaseLoader interface so the rest of the pipeline
is format-agnostic.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger


class Document:
    """Minimal document container passed between pipeline stages."""

    def __init__(self, content: str, metadata: dict[str, Any]) -> None:
        self.content = content
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"Document(source={self.metadata.get('source', '?')!r}, chars={len(self.content)})"


class BaseLoader(ABC):
    """Abstract base: every loader must implement `load`."""

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load a file and return a list of Documents (one per logical section)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _file_hash(path: Path) -> str:
        """SHA-256 of the raw file bytes — used for change detection."""
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and remove control characters."""
        # Collapse runs of whitespace (keep newlines for structure)
        text = re.sub(r"[ \t]+", " ", text)
        # Remove non-printable control chars (except \n \r \t)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]", "", text)
        return text.strip()


class PDFLoader(BaseLoader):
    """Load PDF files using pypdf."""

    def load(self, path: Path) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required: pip install pypdf") from exc

        path = Path(path)
        reader = PdfReader(str(path))
        pages = []

        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            text = self._clean_text(raw)
            if not text:
                continue
            pages.append(
                Document(
                    content=text,
                    metadata={
                        "source": str(path),
                        "file_type": "pdf",
                        "page": i + 1,
                        "file_hash": self._file_hash(path),
                    },
                )
            )

        logger.debug(f"PDFLoader: loaded {len(pages)} pages from {path.name}")
        return pages


class HTMLLoader(BaseLoader):
    """Load HTML files, stripping tags and scripts."""

    def load(self, path: Path) -> list[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4") from exc

        path = Path(path)
        raw_html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = self._clean_text(soup.get_text(separator="\n"))

        if not text:
            return []

        return [
            Document(
                content=text,
                metadata={
                    "source": str(path),
                    "file_type": "html",
                    "title": soup.title.string if soup.title else "",
                    "file_hash": self._file_hash(path),
                },
            )
        ]


class TextLoader(BaseLoader):
    """Load plain-text files."""

    def load(self, path: Path) -> list[Document]:
        path = Path(path)
        text = self._clean_text(path.read_text(encoding="utf-8", errors="replace"))

        if not text:
            return []

        return [
            Document(
                content=text,
                metadata={
                    "source": str(path),
                    "file_type": "txt",
                    "file_hash": self._file_hash(path),
                },
            )
        ]


class MarkdownLoader(BaseLoader):
    """Load Markdown files, preserving structure as plain text."""

    def load(self, path: Path) -> list[Document]:
        path = Path(path)
        raw = path.read_text(encoding="utf-8", errors="replace")

        # Strip Markdown syntax while keeping section structure
        # Remove code fences
        text = re.sub(r"```[\s\S]*?```", "", raw)
        # Remove inline code
        text = re.sub(r"`[^`]+`", "", text)
        # Remove image/link markdown
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Remove bold/italic markers
        text = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)
        # Remove HTML comments
        text = re.sub(r"<!--[\s\S]*?-->", "", text)

        text = self._clean_text(text)

        if not text:
            return []

        return [
            Document(
                content=text,
                metadata={
                    "source": str(path),
                    "file_type": "md",
                    "file_hash": self._file_hash(path),
                },
            )
        ]


class LoaderFactory:
    """Returns the right loader based on file extension."""

    _REGISTRY: dict[str, type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
    }

    @classmethod
    def get(cls, path: Path) -> BaseLoader:
        ext = Path(path).suffix.lower()
        loader_cls = cls._REGISTRY.get(ext)
        if loader_cls is None:
            raise ValueError(
                f"No loader for extension {ext!r}. "
                f"Supported: {list(cls._REGISTRY.keys())}"
            )
        return loader_cls()

    @classmethod
    def load_directory(cls, directory: Path, recursive: bool = True) -> list[Document]:
        """Load all supported documents from a directory."""
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        docs: list[Document] = []

        for file_path in directory.glob(pattern):
            if file_path.suffix.lower() not in cls._REGISTRY:
                continue
            try:
                loader = cls.get(file_path)
                docs.extend(loader.load(file_path))
            except Exception as exc:
                logger.warning(f"Failed to load {file_path}: {exc}")

        logger.info(f"LoaderFactory: loaded {len(docs)} documents from {directory}")
        return docs
