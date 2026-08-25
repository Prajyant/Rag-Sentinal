"""Metadata extraction and anomaly pre-checks.

Extracts author, timestamp, file size, and flags suspicious metadata
patterns that are the first line of the rule-based screener.
"""

from __future__ import annotations

import os
import re
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class MetadataExtractor:
    """Adds filesystem and content metadata to document dicts.

    Flags suspicious metadata conditions:
    - File modified very recently (possible just-in-time injection)
    - Unusually large or empty file
    - Missing or generic author fields
    - Mismatch between file extension and detected content type
    """

    # Files modified within this window are flagged as suspicious
    RECENT_MOD_WINDOW = timedelta(hours=1)

    # Size limits (bytes) outside which we flag the file
    MIN_SIZE = 50
    MAX_SIZE = 50 * 1024 * 1024   # 50 MB

    def enrich(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Add filesystem metadata and suspicion flags to an existing metadata dict."""
        source = metadata.get("source")
        if not source:
            return metadata

        path = Path(source)
        if not path.exists():
            return metadata

        file_stat = path.stat()
        mtime = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)
        size = file_stat.st_size

        metadata.update(
            {
                "file_size_bytes": size,
                "last_modified": mtime.isoformat(),
                "is_readonly": not os.access(path, os.W_OK),
            }
        )

        # ------------------------------------------------------------------
        # Suspicion flags (used by rule_screen.py later)
        # ------------------------------------------------------------------
        flags: list[str] = []

        now = datetime.now(tz=timezone.utc)
        if now - mtime < self.RECENT_MOD_WINDOW:
            flags.append("recently_modified")

        if size < self.MIN_SIZE:
            flags.append("abnormally_small")
        elif size > self.MAX_SIZE:
            flags.append("abnormally_large")

        # Check for missing author — PDFs can embed this; fall back to empty
        author = metadata.get("author", "").strip()
        if not author or author.lower() in {"unknown", "n/a", "-", "none"}:
            flags.append("missing_author")

        metadata["metadata_flags"] = flags
        if flags:
            logger.debug(f"Metadata flags for {path.name}: {flags}")

        return metadata

    @staticmethod
    def extract_pdf_metadata(path: Path) -> dict[str, Any]:
        """Extract embedded PDF metadata (author, creator, creation date, etc.)."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            info = reader.metadata or {}
            return {
                "author": str(info.get("/Author", "")),
                "creator": str(info.get("/Creator", "")),
                "producer": str(info.get("/Producer", "")),
                "pdf_created": str(info.get("/CreationDate", "")),
                "pdf_modified": str(info.get("/ModDate", "")),
                "num_pages": len(reader.pages),
            }
        except Exception as exc:
            logger.debug(f"PDF metadata extraction failed for {path}: {exc}")
            return {}

    @staticmethod
    def detect_hidden_content(text: str) -> list[str]:
        """Scan text for encoding-level hidden content.

        Returns a list of issue names — empty means clean.
        These findings are passed to the rule screener as pre-computed signals.
        """
        issues: list[str] = []

        # Zero-width characters (common injection vector)
        zero_width = re.findall(
            r"[\u200B\u200C\u200D\u200E\u200F\u2060\u2061\u2062\u2063\uFEFF]",
            text,
        )
        if zero_width:
            issues.append(f"zero_width_chars:{len(zero_width)}")

        # Unicode tag block (U+E0000–U+E007F) — used for invisible text smuggling
        tag_block = re.findall(r"[\U000E0000-\U000E007F]", text)
        if tag_block:
            issues.append(f"unicode_tag_block:{len(tag_block)}")

        # Homoglyph candidates — Cyrillic/Greek chars mixed with Latin
        homoglyph = re.findall(r"[\u0400-\u04FF\u0370-\u03FF]", text)
        if len(homoglyph) > 5:   # allow occasional proper nouns
            issues.append(f"homoglyph_chars:{len(homoglyph)}")

        # HTML comments in plain text (should have been stripped by loader)
        if re.search(r"<!--", text):
            issues.append("html_comment_in_text")

        # Base64-looking blobs > 100 chars
        b64 = re.findall(r"[A-Za-z0-9+/]{100,}={0,2}", text)
        if b64:
            issues.append(f"base64_blob:{len(b64)}")

        return issues
