"""Tests for document chunking, hashing, and mock embedding."""

from pathlib import Path

import pytest

from hysearch.core.chunker import (
    chunk_text,
    compute_file_hash,
    compute_text_hash,
)
from hysearch.core.config import DEFAULT_EMBED_DIM
from hysearch.core.embedder import DummyEmbedder


def test_compute_text_and_file_hash(tmp_path: Path) -> None:
    """Verify SHA-256 hash calculation is deterministic for text and file."""
    sample_text = "This is a test content for hashing verification."
    hash1 = compute_text_hash(sample_text)
    hash2 = compute_text_hash(sample_text)
    assert hash1 == hash2
    assert len(hash1) == 64

    # ファイル経由での検証
    file_path = tmp_path / "test.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    file_hash = compute_file_hash(file_path)
    assert file_hash == hash1


def test_chunk_text_empty_and_short() -> None:
    """Verify chunking behavior on empty and short texts."""
    assert chunk_text("") == []

    short = "Short document."
    chunks = chunk_text(short, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == short
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(short)


def test_chunk_text_with_markdown_headings() -> None:
    """Verify that chunks split cleanly at Markdown heading boundaries."""
    doc = (
        "# Title 1\n"
        "This is paragraph one.\n\n"
        "## Title 2\n"
        "This is paragraph two."
    )
    chunks = chunk_text(doc, chunk_size=35, overlap=10)
    assert len(chunks) >= 2
    # 各チャンクの連続性とインデックスを確認
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert len(c.content) > 0


def test_chunk_text_overlap_and_sliding_window() -> None:
    """Verify sliding window with overlap produces overlapping text."""
    long_text = "abcdefghij" * 20  # 200 chars
    chunks = chunk_text(long_text, chunk_size=50, overlap=15)
    assert len(chunks) > 1

    # 前のチャンクの末尾と次のチャンクの先頭がオーバーラップしているか確認
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        assert c2.start_char < c1.end_char


def test_chunk_text_invalid_arguments() -> None:
    """Verify that invalid chunk_size or overlap raises ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("abc", chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="overlap must satisfy"):
        chunk_text("abc", chunk_size=50, overlap=50)

    with pytest.raises(ValueError, match="overlap must satisfy"):
        chunk_text("abc", chunk_size=50, overlap=-5)


def test_dummy_embedder_determinism_and_dimensions() -> None:
    """Verify DummyEmbedder dimensions and vector determinism."""
    embedder = DummyEmbedder(dimension=DEFAULT_EMBED_DIM)
    assert embedder.dimension == DEFAULT_EMBED_DIM

    texts = ["apple", "banana", "apple"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    assert len(vectors[0]) == DEFAULT_EMBED_DIM

    # 同一文字列は同一ベクトル
    assert vectors[0] == vectors[2]
    # 異なる文字列は異なるベクトル
    assert vectors[0] != vectors[1]

    # embed_query の動作確認
    query_vec = embedder.embed_query("apple")
    assert query_vec == vectors[0]
