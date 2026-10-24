"""Hybrid search engine combining BM25 full-text and vector similarity.

This module executes keyword search on FTS5 trigram indexes, vector KNN
search on sqlite-vec, and merges the results using Reciprocal Rank
Fusion (RRF).
"""

import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import sqlite_vec

from hysearch.core.config import (
    DEFAULT_RRF_K,
    DEFAULT_WEIGHT_BM25,
    DEFAULT_WEIGHT_VEC,
    TOKENIZER_TRIGRAM,
)
from hysearch.core.database import (
    get_db_embed_dim,
    get_db_embed_model,
    get_db_tokenizer,
)
from hysearch.core.embedder import EmbedderProtocol, FastEmbedder
from hysearch.core.models import SearchHit, SearchResponse
from hysearch.core.tokenizer import build_fts_query, resolve_tokenizer
from hysearch.utils.text import extract_snippet

SearchMode = Literal["hybrid", "bm25", "vector"]


def _match_path(file_path_str: str, path_filter: str) -> bool:
    """Check if a file path matches the path filter prefix or folder."""
    target = Path(path_filter).resolve()
    file_p = Path(file_path_str)
    try:
        file_p.relative_to(target)
        return True
    except ValueError:
        pass
    norm_file = file_path_str.replace("\\", "/")
    norm_filter = path_filter.replace("\\", "/").rstrip("/")
    return norm_filter in norm_file


def _match_glob(file_path_str: str, glob_pattern: str) -> bool:
    """Check if a file path matches the given glob pattern."""
    p = Path(file_path_str)
    norm_file = file_path_str.replace("\\", "/")
    norm_glob = glob_pattern.replace("\\", "/")
    return (
        fnmatch(norm_file, norm_glob)
        or fnmatch(p.name, norm_glob)
        or fnmatch(str(p), glob_pattern)
    )


def get_filtered_chunk_ids(
    conn: sqlite3.Connection,
    path_filter: str | None = None,
    glob_filter: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[int] | None:
    """Retrieve chunk IDs matching path, glob, and modification filters.

    Args:
        conn: Open SQLite connection.
        path_filter: Optional directory or path substring.
        glob_filter: Optional file glob pattern.
        since: Optional minimum modification datetime.
        until: Optional maximum modification datetime.

    Returns:
        list[int] | None: Allowed chunk IDs, or None if no filter applied.
    """
    if (
        path_filter is None
        and glob_filter is None
        and since is None
        and until is None
    ):
        return None

    cursor = conn.cursor()
    cursor.execute("SELECT id, path, mtime FROM files;")
    rows = cursor.fetchall()

    since_ts = since.timestamp() if since is not None else None
    until_ts = until.timestamp() if until is not None else None
    matching_file_ids: list[int] = []

    for fid, fpath, fmtime in rows:
        if since_ts is not None and fmtime < since_ts:
            continue
        if until_ts is not None and fmtime > until_ts:
            continue
        if path_filter is not None and not _match_path(fpath, path_filter):
            continue
        if glob_filter is not None and not _match_glob(fpath, glob_filter):
            continue
        matching_file_ids.append(fid)

    if not matching_file_ids:
        return []

    placeholders = ",".join("?" for _ in matching_file_ids)
    cursor.execute(
        f"SELECT id FROM chunks WHERE file_id IN ({placeholders});",
        matching_file_ids,
    )
    return [row[0] for row in cursor.fetchall()]


def _clean_fts_query(
    query: str, tokenizer_name: str = TOKENIZER_TRIGRAM
) -> str | None:
    """Prepare a safe query string for SQLite FTS5 virtual table.

    Args:
        query: Raw user query string.
        tokenizer_name: Active tokenizer name stored in the database.

    Returns:
        str | None: Formatted FTS query, or None if too short/invalid.
    """
    return build_fts_query(query, tokenizer_name=tokenizer_name)


def search_bm25(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 50,
    target_ids: list[int] | None = None,
) -> list[int]:
    """Execute BM25 keyword search using SQLite FTS5 virtual table.

    Args:
        conn: Open SQLite connection.
        query: Raw search query text.
        limit: Maximum number of ranked results to return.
        target_ids: Optional list of chunk IDs to restrict the search.

    Returns:
        list[int]: Ordered list of chunk rowids from top to bottom rank.
    """
    if target_ids is not None and not target_ids:
        return []

    tokenizer_name = resolve_tokenizer(get_db_tokenizer(conn))
    fts_query = _clean_fts_query(query, tokenizer_name=tokenizer_name)
    if not fts_query:
        return []

    cursor = conn.cursor()
    try:
        if target_ids is not None:
            placeholders = ",".join("?" for _ in target_ids)
            cursor.execute(
                f"""
                SELECT rowid
                FROM fts_chunks
                WHERE fts_chunks MATCH ? AND rowid IN ({placeholders})
                ORDER BY rank
                LIMIT ?;
                """,
                [fts_query, *target_ids, limit],
            )
        else:
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
    target_ids: list[int] | None = None,
) -> list[int]:
    """Execute vector approximate nearest neighbors search using sqlite-vec.

    Args:
        conn: Open SQLite connection.
        query: Search query text.
        embedder: Vector embedding instance.
        limit: Maximum number of nearest results to return.
        target_ids: Optional list of chunk IDs to restrict the search.

    Returns:
        list[int]: Ordered list of chunk IDs sorted by increasing distance.
    """
    if target_ids is not None and not target_ids:
        return []

    # 登録データが存在するか確認
    count = conn.execute("SELECT COUNT(*) FROM vec_chunks;").fetchone()[0]
    if count == 0:
        return []

    max_available = len(target_ids) if target_ids is not None else count
    k_val = min(limit, max_available)
    if k_val <= 0:
        return []

    query_vector = embedder.embed_query(query)
    serialized = sqlite_vec.serialize_float32(query_vector)

    cursor = conn.cursor()
    if target_ids is not None:
        placeholders = ",".join("?" for _ in target_ids)
        cursor.execute(
            f"""
            SELECT id
            FROM vec_chunks
            WHERE embedding MATCH ? AND k = ? AND id IN ({placeholders})
            ORDER BY distance;
            """,
            [serialized, k_val, *target_ids],
        )
    else:
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
    path_filter: str | None = None,
    glob_filter: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
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
        path_filter: Optional directory or path substring.
        glob_filter: Optional file glob pattern.
        since: Optional minimum modification datetime.
        until: Optional maximum modification datetime.

    Returns:
        SearchResponse: Structured search response containing ranked hits.
    """
    start_time = time.perf_counter()
    target_ids = get_filtered_chunk_ids(
        conn,
        path_filter=path_filter,
        glob_filter=glob_filter,
        since=since,
        until=until,
    )
    if target_ids is not None and not target_ids:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return SearchResponse(
            query=query,
            total_hits=0,
            hits=[],
            execution_time_ms=round(elapsed, 2),
        )

    candidate_limit = max(top_k * 5, 50)
    bm25_ids: list[int] = []
    vec_ids: list[int] = []

    if mode in ("hybrid", "bm25"):
        bm25_ids = search_bm25(
            conn, query, limit=candidate_limit, target_ids=target_ids
        )

    if mode in ("hybrid", "vector"):
        active_embedder: EmbedderProtocol = (
            embedder
            if embedder is not None
            else FastEmbedder(
                model_name=get_db_embed_model(conn),
                dimension=get_db_embed_dim(conn),
            )
        )
        vec_ids = search_vector(
            conn,
            query,
            embedder=active_embedder,
            limit=candidate_limit,
            target_ids=target_ids,
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
