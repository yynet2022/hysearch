"""Tests for document chunking, hashing, and mock embedding."""

from pathlib import Path

import pytest

from hysearch.core.chunker import (
    _find_split_point,
    chunk_text,
    compute_file_hash,
    compute_text_hash,
)
from hysearch.core.config import (
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    EMBED_MODEL_GEMMA,
    EMBED_MODEL_MPNET,
    EMBED_MODEL_QWEN3,
)
from hysearch.core.embedder import (
    DummyEmbedder,
    FastEmbedder,
    resolve_embed_model,
)


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


def test_dummy_embedder_fixed_value() -> None:
    """Verify DummyEmbedder with fixed_value fills constant vectors."""
    embedder = DummyEmbedder(dimension=4, fixed_value=0.5)
    vec = embedder.embed_query("sample")
    assert vec == [0.5, 0.5, 0.5, 0.5]


def test_fastembedder_empty_list() -> None:
    """Verify FastEmbedder returns empty list when given no texts."""
    embedder = FastEmbedder()
    assert embedder.dimension == DEFAULT_EMBED_DIM
    assert embedder.embed([]) == []


def test_resolve_embed_model() -> None:
    """Verify resolve_embed_model validates and resolves model and dim."""
    assert resolve_embed_model(None) == (DEFAULT_EMBED_MODEL, 384)
    assert resolve_embed_model(EMBED_MODEL_MPNET) == (EMBED_MODEL_MPNET, 768)
    assert resolve_embed_model(EMBED_MODEL_QWEN3.lower()) == (
        EMBED_MODEL_QWEN3,
        1024,
    )
    assert resolve_embed_model(f"  {EMBED_MODEL_GEMMA}  ") == (
        EMBED_MODEL_GEMMA,
        768,
    )

    with pytest.raises(ValueError, match="Unsupported embedding model"):
        resolve_embed_model("unknown/model-v1")


def test_fastembedder_mocked_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify FastEmbedder lazy initialization and passage/query methods."""

    class MockArray:
        def __init__(self, vals: list[float]) -> None:
            self.vals = vals

        def tolist(self) -> list[float]:
            return self.vals

    class MockTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def passage_embed(self, texts: list[str]) -> list[object]:
            return [MockArray([0.1, 0.2, 0.3]) for _ in texts]

        def query_embed(self, query: str) -> list[object]:
            return [MockArray([0.4, 0.5, 0.6])]

    monkeypatch.setattr("fastembed.TextEmbedding", MockTextEmbedding)

    embedder = FastEmbedder(model_name="mock-model", dimension=3)
    assert embedder.dimension == 3
    assert embedder._model_instance is None

    # embed の呼び出しで passage_embed が呼ばれ結果が返ること
    vectors = embedder.embed(["hello", "world"])
    assert embedder._model_instance is not None
    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    # embed_query の呼び出しで query_embed が呼ばれ結果が返ること
    query_vec = embedder.embed_query("query")
    assert query_vec == [0.4, 0.5, 0.6]


def test_find_split_point_boundaries() -> None:
    """Verify _find_split_point handles various split patterns."""
    # 1. 空の探索範囲 (78行目)
    assert _find_split_point("sample", 3, 3) == 3

    # 2. Markdown 見出しの前 (83行目)
    text_heading = "intro text\n# Heading 1\nbody text"
    pt = _find_split_point(text_heading, 5, 20)
    assert pt == 11

    # 3. 単一改行 (93行目)
    text_newline = "line1\nline2"
    pt = _find_split_point(text_newline, 2, 8)
    assert pt == 6

    # 4. 句読点 (99行目)
    text_punct = "文その一。文その二"
    pt = _find_split_point(text_punct, 2, 7)
    assert pt == 5  # '。' の直後


def test_chunk_text_split_point_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback to ideal_end when split point is <= start."""
    # 160行目: end <= start となるようモック
    monkeypatch.setattr(
        "hysearch.core.chunker._find_split_point",
        lambda text, search_start, search_end: 0,
    )
    chunks = chunk_text("01234567890123456789", chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    assert chunks[0].end_char == 10
