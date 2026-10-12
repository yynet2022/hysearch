"""Tests for searcher engine, RRF scoring, and query fallbacks."""

import sqlite3
from pathlib import Path

import pytest

from hysearch.core.embedder import DummyEmbedder
from hysearch.core.indexer import index_file
from hysearch.core.searcher import compute_rrf_scores, search, search_bm25
from hysearch.utils.text import clean_text, extract_snippet


def test_clean_text_and_extract_snippet() -> None:
    """Verify text cleaning and snippet generation with keywords."""
    raw = "Line 1\n\nLine 2   with   spaces\r\nand tabs\t."
    cleaned = clean_text(raw)
    assert cleaned == "Line 1 Line 2 with spaces and tabs ."

    content = "Prefix words. The target keyword appears here. Suffix words."
    snippet = extract_snippet(content, query="keyword", max_length=40)
    assert "keyword" in snippet
    assert "..." in snippet


def test_clean_text_and_extract_snippet_empty() -> None:
    """Verify empty input handling for clean_text and extract_snippet."""
    assert clean_text("") == ""
    assert extract_snippet("") == ""


def test_extract_snippet_fallback_no_match() -> None:
    """Verify snippet fallback trims prefix with ellipsis when no match."""
    long_content = "Word " * 50  # 250 chars
    snippet = extract_snippet(long_content, query="nonexistent", max_length=30)
    assert snippet.endswith("...")
    assert len(snippet) <= 30


def test_extract_snippet_match_at_end_of_text() -> None:
    """Verify snippet window adjustment when match is near text end."""
    content = ("prefix " * 20) + "endtarget"
    snippet = extract_snippet(content, query="endtarget", max_length=35)
    assert snippet.startswith("...")
    assert not snippet.endswith("...")
    assert "endtarget" in snippet


def test_compute_rrf_scores() -> None:
    """Verify reciprocal rank fusion formula calculation."""
    bm25_ids = [10, 20, 30]
    vec_ids = [20, 10, 40]
    scores = compute_rrf_scores(bm25_ids, vec_ids, rrf_k=60)

    # 10番: BM25 rank 1 (1/61), Vec rank 2 (1/62)
    # 20番: BM25 rank 2 (1/62), Vec rank 1 (1/61) -> 同スコア
    score_10 = scores[10][0]
    score_20 = scores[20][0]
    assert abs(score_10 - score_20) < 1e-6

    # 30番: BM25 rank 3 (1/63), Vec なし (0)
    score_30 = scores[30][0]
    assert abs(score_30 - (1.0 / 63)) < 1e-6
    assert scores[30][1] == 3
    assert scores[30][2] is None


def test_hybrid_search_with_japanese(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify hybrid search on Japanese content."""
    doc1 = tmp_path / "doc1.md"
    doc1.write_text(
        "SQLiteとfastembedを用いた軽量ハイブリッド検索CLIツール。",
        encoding="utf-8",
    )
    doc2 = tmp_path / "doc2.md"
    doc2.write_text(
        "機械学習とベクトル埋め込みによるセマンティック検索の実装。",
        encoding="utf-8",
    )

    index_file(memory_db, doc1, dummy_embedder)
    index_file(memory_db, doc2, dummy_embedder)

    response = search(
        conn=memory_db,
        query="ハイブリッド検索",
        embedder=dummy_embedder,
        mode="hybrid",
        top_k=5,
    )
    assert response.total_hits > 0
    assert response.hits[0].file_path == str(doc1.resolve())
    assert response.hits[0].bm25_rank is not None
    assert response.hits[0].snippet is not None


def test_short_query_fallback(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify queries under 3 characters fallback to vector search safely."""
    doc1 = tmp_path / "doc1.md"
    doc1.write_text("AIの進化と未来の設計仕様書。", encoding="utf-8")
    index_file(memory_db, doc1, dummy_embedder)

    # 2文字の短文字クエリ（FTS5 trigram では単独マッチ不可）
    response = search(
        conn=memory_db,
        query="AI",
        embedder=dummy_embedder,
        mode="hybrid",
    )
    assert response.total_hits == 1
    # BM25 はスキップされ、ベクトル検索によってヒットする
    assert response.hits[0].bm25_rank is None
    assert response.hits[0].vec_rank == 1


def test_search_modes(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify switching between hybrid, bm25, and vector search modes."""
    doc = tmp_path / "doc.md"
    doc.write_text("Pythonプログラミングとデータベース検索", encoding="utf-8")
    index_file(memory_db, doc, dummy_embedder)

    res_bm25 = search(
        conn=memory_db,
        query="プログラミング",
        embedder=dummy_embedder,
        mode="bm25",
    )
    assert res_bm25.total_hits == 1
    assert res_bm25.hits[0].bm25_rank == 1
    assert res_bm25.hits[0].vec_rank is None

    res_vec = search(
        conn=memory_db,
        query="プログラミング",
        embedder=dummy_embedder,
        mode="vector",
    )
    assert res_vec.total_hits == 1
    assert res_vec.hits[0].bm25_rank is None
    assert res_vec.hits[0].vec_rank == 1


def test_search_bm25_operational_error() -> None:
    """Verify search_bm25 returns empty list when table is missing."""
    conn = sqlite3.connect(":memory:")
    # fts_chunks 未作成のため OperationalError が発生し [] が返る
    results = search_bm25(conn, "keyword")
    assert results == []
    conn.close()


def test_search_skips_missing_chunk_in_data(
    memory_db: sqlite3.Connection,
    dummy_embedder: DummyEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify search gracefully skips chunk ID missing from chunks table."""
    # 存在しない ID 99999 を返すよう search_bm25 をモック
    monkeypatch.setattr(
        "hysearch.core.searcher.search_bm25",
        lambda conn, query, limit: [99999],
    )
    res = search(
        conn=memory_db,
        query="test",
        embedder=dummy_embedder,
        mode="bm25",
    )
    assert res.total_hits == 0
    assert res.hits == []


def test_search_on_empty_db(
    memory_db: sqlite3.Connection,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify search on empty database returns zero hits response."""
    res = search(
        conn=memory_db,
        query="empty query",
        embedder=dummy_embedder,
    )
    assert res.total_hits == 0
    assert res.hits == []
