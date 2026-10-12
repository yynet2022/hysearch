"""Hybrid search engine combining BM25 full-text and vector similarity.

This module executes keyword search on FTS5 trigram indexes, vector KNN
search on sqlite-vec, and merges the results using Reciprocal Rank
Fusion (RRF).
"""

import sqlite3
import time
from collections import defaultdict
from typing import Literal

import sqlite_vec

from hysearch.core.config import (
    DEFAULT_RRF_K,
    DEFAULT_WEIGHT_BM25,
    DEFAULT_WEIGHT_VEC,
)
from hysearch.core.embedder import EmbedderProtocol, FastEmbedder
from hysearch.core.models import SearchHit, SearchResponse
from hysearch.utils.text import extract_snippet

SearchMode = Literal["hybrid", "bm25", "vector"]


def _clean_fts_query(query: str) -> str | None:
    """Prepare a safe query string for SQLite FTS5 trigram tokenizer.

    FTS5 trigram requires tokens of length 3 or more characters.
    If the effective query string is shorter than 3 characters, return None
    to indicate BM25 matching should be skipped.

    Args:
        query: Raw user query string.

    Returns:
        str | None: Formatted FTS query, or None if too short/invalid.
    """
    stripped = query.strip()
    # 空白を除いた長さが3文字未満なら trigram 検索不可
    compact = "".join(stripped.split())
    if len(compact) < 3:
        return None

    # ダブルクォートをエスケープしてフレーズクエリ化
    escaped = stripped.replace('"', '""')
    return f'"{escaped}"'


def search_bm25(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 50,
) -> list[int]:
    """Execute BM25 keyword search using SQLite FTS5 trigram virtual table.

    Args:
        conn: Open SQLite connection.
        query: Raw search query text.
        limit: Maximum number of ranked results to return.

    Returns:
        list[int]: Ordered list of chunk rowids from top to bottom rank.
    """
    fts_query = _clean_fts_query(query)
    if not fts_query:
        return []

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT rowid
            FROM fts_chunks
            WHERE fts_chunks MATCH ?
            ORDER BY rank
            LIMIT ?;
            """,
            (fts_query, limit),
        )
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # 構文エラー等の場合は空リストを返しベクトル検索に委ねる
        return []


def search_vector(
    conn: sqlite3.Connection,
    query: str,
    embedder: EmbedderProtocol,
    limit: int = 50,
) -> list[int]:
    """Execute vector approximate nearest neighbors search using sqlite-vec.

    Args:
        conn: Open SQLite connection.
        query: Search query text.
        embedder: Vector embedding instance.
        limit: Maximum number of nearest results to return.

    Returns:
        list[int]: Ordered list of chunk IDs sorted by increasing distance.
    """
    # 登録データが存在するか確認
    count = conn.execute("SELECT COUNT(*) FROM vec_chunks;").fetchone()[0]
    if count == 0:
        return []

    k_val = min(limit, count)
    query_vector = embedder.embed_query(query)
    serialized = sqlite_vec.serialize_float32(query_vector)

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id
        FROM vec_chunks
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance;
        """,
        (serialized, k_val),
    )
    return [row[0] for row in cursor.fetchall()]


def compute_rrf_scores(
    bm25_ids: list[int],
    vec_ids: list[int],
    rrf_k: int = DEFAULT_RRF_K,
    weight_bm25: float = DEFAULT_WEIGHT_BM25,
    weight_vec: float = DEFAULT_WEIGHT_VEC,
) -> dict[int, tuple[float, int | None, int | None]]:
    """Calculate Reciprocal Rank Fusion (RRF) scores across rankings.

    Args:
        bm25_ids: Ranked chunk IDs from BM25 (1-based order).
        vec_ids: Ranked chunk IDs from vector search (1-based order).
        rrf_k: Smoothing constant (default: 60).
        weight_bm25: Weight multiplier for BM25 ranking.
        weight_vec: Weight multiplier for vector ranking.

    Returns:
        dict[int, tuple[float, int | None, int | None]]:
            Mapping from chunk_id to (rrf_score, bm25_rank, vec_rank).
    """
    scores: dict[int, float] = defaultdict(float)
    bm25_ranks: dict[int, int] = {}
    vec_ranks: dict[int, int] = {}

    for rank, cid in enumerate(bm25_ids, start=1):
        bm25_ranks[cid] = rank
        scores[cid] += weight_bm25 / (rrf_k + rank)

    for rank, cid in enumerate(vec_ids, start=1):
        vec_ranks[cid] = rank
        scores[cid] += weight_vec / (rrf_k + rank)

    results: dict[int, tuple[float, int | None, int | None]] = {}
    for cid, total_score in scores.items():
        results[cid] = (
            total_score,
            bm25_ranks.get(cid),
            vec_ranks.get(cid),
        )

    return results


def search(
    conn: sqlite3.Connection,
    query: str,
    embedder: EmbedderProtocol | None = None,
    mode: SearchMode = "hybrid",
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
    weight_bm25: float = DEFAULT_WEIGHT_BM25,
    weight_vec: float = DEFAULT_WEIGHT_VEC,
) -> SearchResponse:
    """Perform hybrid, BM25, or vector search on the database.

    Args:
        conn: Open SQLite connection.
        query: User search query.
        embedder: Embedding model instance (defaults to FastEmbedder).
        mode: Search mode ('hybrid', 'bm25', or 'vector').
        top_k: Number of top results to return.
        rrf_k: Smoothing parameter for RRF formula.
        weight_bm25: BM25 score weight.
        weight_vec: Vector score weight.

    Returns:
        SearchResponse: Structured search response containing ranked hits.
    """
    start_time = time.perf_counter()
    active_embedder: EmbedderProtocol = (
        embedder if embedder is not None else FastEmbedder()
    )

    candidate_limit = max(top_k * 5, 50)
    bm25_ids: list[int] = []
    vec_ids: list[int] = []

    if mode in ("hybrid", "bm25"):
        bm25_ids = search_bm25(conn, query, limit=candidate_limit)

    if mode in ("hybrid", "vector"):
        vec_ids = search_vector(
            conn, query, embedder=active_embedder, limit=candidate_limit
        )

    rrf_map = compute_rrf_scores(
        bm25_ids=bm25_ids,
        vec_ids=vec_ids,
        rrf_k=rrf_k,
        weight_bm25=weight_bm25 if mode != "vector" else 0.0,
        weight_vec=weight_vec if mode != "bm25" else 0.0,
    )

    if not rrf_map:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return SearchResponse(
            query=query,
            total_hits=0,
            hits=[],
            execution_time_ms=round(elapsed, 2),
        )

    # RRF スコアの高い順にソート
    sorted_items = sorted(
        rrf_map.items(), key=lambda item: item[1][0], reverse=True
    )[:top_k]

    selected_ids = [item[0] for item in sorted_items]
    placeholders = ",".join("?" for _ in selected_ids)

    # チャンク情報とファイルパスを取得
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT c.id, f.path, c.chunk_index, c.content
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE c.id IN ({placeholders});
        """,
        selected_ids,
    )
    chunk_data = {row[0]: row for row in cursor.fetchall()}

    hits: list[SearchHit] = []
    for cid in selected_ids:
        if cid not in chunk_data:
            continue
        row = chunk_data[cid]
        score, b_rank, v_rank = rrf_map[cid]
        content = row[3]
        snippet = extract_snippet(content, query=query)

        hits.append(
            SearchHit(
                chunk_id=cid,
                file_path=row[1],
                chunk_index=row[2],
                content=content,
                rrf_score=round(score, 6),
                bm25_rank=b_rank,
                vec_rank=v_rank,
                snippet=snippet,
            )
        )

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return SearchResponse(
        query=query,
        total_hits=len(hits),
        hits=hits,
        execution_time_ms=round(elapsed, 2),
    )
