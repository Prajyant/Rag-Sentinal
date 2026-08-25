"""Unit tests for ingestion components."""

import tempfile
from pathlib import Path

import pytest

from rag_sentinel.ingestion.loaders import TextLoader, MarkdownLoader, LoaderFactory
from rag_sentinel.ingestion.chunker import Chunker
from rag_sentinel.ingestion.metadata import MetadataExtractor


class TestLoaders:
    def test_text_loader(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world. This is test content for the loader.")
        loader = TextLoader()
        docs = loader.load(f)
        assert len(docs) == 1
        assert "Hello world" in docs[0].content
        assert docs[0].metadata["file_type"] == "txt"
        assert "file_hash" in docs[0].metadata

    def test_markdown_loader_strips_syntax(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Header\n\nThis is **bold** text with a [link](http://example.com).\n\n<!-- hidden comment -->")
        loader = MarkdownLoader()
        docs = loader.load(f)
        assert len(docs) == 1
        assert "**" not in docs[0].content
        assert "<!--" not in docs[0].content
        assert "bold" in docs[0].content

    def test_loader_factory_dispatch(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("Some content")
        loader = LoaderFactory.get(txt)
        assert isinstance(loader, TextLoader)

    def test_loader_factory_unsupported_raises(self, tmp_path):
        f = tmp_path / "doc.xyz"
        with pytest.raises(ValueError, match="No loader"):
            LoaderFactory.get(f)

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A content.")
        (tmp_path / "b.md").write_text("Document B content.")
        (tmp_path / "skip.xyz").write_text("Should be ignored.")
        docs = LoaderFactory.load_directory(tmp_path)
        assert len(docs) == 2


class TestChunker:
    def setup_method(self):
        self.chunker = Chunker(chunk_size=100, chunk_overlap=15)

    def test_short_text_is_single_chunk(self, tmp_path):
        from rag_sentinel.ingestion.loaders import Document
        doc = Document("Short text.", {"source": "test.txt"})
        chunks = self.chunker.split(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_long_text_splits_correctly(self, tmp_path):
        from rag_sentinel.ingestion.loaders import Document
        long_text = " ".join(["word"] * 200)  # 200 words ~ 1000 chars
        doc = Document(long_text, {"source": "test.txt"})
        chunks = self.chunker.split(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= self.chunker.chunk_size + 50  # allow slight overflow

    def test_metadata_inherited(self, tmp_path):
        from rag_sentinel.ingestion.loaders import Document
        doc = Document("Some text.", {"source": "myfile.txt", "author": "Tester"})
        chunks = self.chunker.split(doc)
        for chunk in chunks:
            assert chunk.metadata["source"] == "myfile.txt"
            assert chunk.metadata["author"] == "Tester"


class TestMetadataExtractor:
    def test_enrich_adds_file_size(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Content here.")
        extractor = MetadataExtractor()
        meta = extractor.enrich({"source": str(f)})
        assert "file_size_bytes" in meta
        assert meta["file_size_bytes"] > 0

    def test_detect_zero_width_chars(self):
        text = "Normal text\u200b\u200c hidden content"
        issues = MetadataExtractor.detect_hidden_content(text)
        assert any("zero_width" in i for i in issues)

    def test_detect_unicode_tag_block(self):
        text = "Text with \U000E0048\U000E0049 tag block chars"
        issues = MetadataExtractor.detect_hidden_content(text)
        assert any("unicode_tag_block" in i for i in issues)

    def test_clean_text_no_issues(self):
        text = "This is perfectly normal text with no hidden content."
        issues = MetadataExtractor.detect_hidden_content(text)
        assert issues == []
