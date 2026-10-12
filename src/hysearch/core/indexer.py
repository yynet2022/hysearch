"""Indexing pipeline for scanning, chunking, and vectorizing documents.

This module crawls markdown and text files, calculates SHA-256 digests
to detect changes, splits content into chunks, and stores full-text and
vector representations in SQLite.
"""

import sqlite3
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import sqlite_vec

from hysearch.core.chunker import chunk_text, compute_file_hash
from hysearch.core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
)
from hysearch.core.embedder import EmbedderProtocol, FastEmbedder
from hysearch.core.models import IndexingStats

PathLike = str | Path
ProgressCallback = Callable[[int, int, Path], None]
PruneCallback = Callable[[str, bool], None]


def scan_directory(
    target_dir: PathLike,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    """Recursively discover files matching supported extensions.

    Args:
        target_dir: Root directory path to crawl.
        extensions: Sequence of file extensions to include (e.g. ['.md']).

    Returns:
        list[Path]: List of resolved matching file paths, sorted by path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    path = Path(target_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {path}")

    ext_set = {e.lower() for e in extensions}
    matched: list[Path] = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_set:
            matched.append(p)

    matched.sort()
    return matched


def get_stored_file_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """Retrieve mapping of file paths to their stored SHA-256 hashes.

    Args:
        conn: Open SQLite connection.

    Returns:
        dict[str, str]: Mapping from file path string to hex hash string.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT path, hash FROM files;")
    return {row[0]: row[1] for row in cursor.fetchall()}


def index_file(
    conn: sqlite3.Connection,
    file_path: Path,
    embedder: EmbedderProtocol,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    current_hash: str | None = None,
) -> int:
    """Index or re-index a single document file.

    Args:
        conn: Open SQLite connection.
        file_path: Path to the target file.
        embedder: Vector embedding instance.
        chunk_size: Target character count per chunk.
        overlap: Character overlap count.
        current_hash: Precomputed SHA-256 hex digest, or None to compute.

    Returns:
        int: Number of chunks created and stored for this file.
    """
    path_str = str(file_path.resolve())
    h = current_hash or compute_file_hash(file_path)
    mtime = file_path.stat().st_mtime

    text = file_path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    with conn:
        # 既存レコードが存在する場合は削除（CASCADE で chunks も消去）
        conn.execute("DELETE FROM files WHERE path = ?;", (path_str,))

        if not chunks:
            # 空ファイルの場合もファイルメタデータだけ記録
            conn.execute(
                """
                INSERT INTO files (path, hash, chunk_count, mtime)
                VALUES (?, ?, ?, ?);
                """,
                (path_str, h, 0, mtime),
            )
            return 0

        # files テーブルに新規登録
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO files (path, hash, chunk_count, mtime)
            VALUES (?, ?, ?, ?);
            """,
            (path_str, h, len(chunks), mtime),
        )
        file_id = cursor.lastrowid
        assert file_id is not None

        # チャンク本文の埋め込みベクトルを一括生成
        chunk_texts = [c.content for c in chunks]
        embeddings = embedder.embed(chunk_texts)

        # chunks テーブルおよび vec_chunks テーブルに挿入
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cursor.execute(
                """
                INSERT INTO chunks (file_id, chunk_index, content, start_char,
                                    end_char)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    file_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.start_char,
                    chunk.end_char,
                ),
            )
            chunk_id = cursor.lastrowid
            assert chunk_id is not None

            # sqlite-vec に float32 ベクトルを登録
            serialized_vec = sqlite_vec.serialize_float32(embedding)
            cursor.execute(
                "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?);",
                (chunk_id, serialized_vec),
            )

    return len(chunks)


def index_directory(
    conn: sqlite3.Connection,
    target_dir: PathLike,
    embedder: EmbedderProtocol | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
    extensions: Sequence[str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> IndexingStats:
    """Scan and index all matching documents within the target directory.

    Args:
        conn: Open SQLite connection.
        target_dir: Path to the target root directory.
        embedder: Vector embedding provider (defaults to FastEmbedder).
        chunk_size: Target characters per chunk.
        overlap: Overlap characters between chunks.
        force: If True, re-index all files ignoring hash equality.
        extensions: Optional file extensions sequence to filter targets.
        on_progress: Optional callback receiving (current_index, total_count,
            current_file_path).

    Returns:
        IndexingStats: Summary statistics of the indexing process.
    """
    start_time = time.perf_counter()
    active_embedder: EmbedderProtocol = (
        embedder if embedder is not None else FastEmbedder()
    )

    scan_exts = extensions if extensions is not None else SUPPORTED_EXTENSIONS
    files = scan_directory(target_dir, extensions=scan_exts)
    stored_hashes = get_stored_file_hashes(conn)

    total_files = len(files)
    stats = IndexingStats(files_scanned=total_files)

    for idx, file_path in enumerate(files, start=1):
        if on_progress is not None:
            on_progress(idx, total_files, file_path)

        resolved_str = str(file_path.resolve())
        current_hash = compute_file_hash(file_path)

        if (
            not force
            and resolved_str in stored_hashes
            and stored_hashes[resolved_str] == current_hash
        ):
            stats.files_skipped += 1
            continue

        created = index_file(
            conn=conn,
            file_path=file_path,
            embedder=active_embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            current_hash=current_hash,
        )
        stats.files_indexed += 1
        stats.chunks_created += created

    stats.elapsed_seconds = round(time.perf_counter() - start_time, 4)
    return stats


def prune_missing_files(
    conn: sqlite3.Connection,
    dry_run: bool = False,
    on_missing: PruneCallback | None = None,
) -> tuple[int, list[str]]:
    """Remove database records for files that no longer exist on disk.

    Args:
        conn: Open SQLite connection.
        dry_run: If True, do not actually delete rows from the database.
        on_missing: Optional callback receiving (file_path_str, is_dry_run)
            called immediately when each missing file is detected.

    Returns:
        tuple[int, list[str]]: (total_files_checked, pruned_file_paths).
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, path FROM files ORDER BY id;")
    rows = cursor.fetchall()

    total_checked = len(rows)
    pruned: list[str] = []

    for fid, fpath in rows:
        if not Path(fpath).exists():
            if on_missing is not None:
                on_missing(fpath, dry_run)
            if not dry_run:
                with conn:
                    conn.execute("DELETE FROM files WHERE id = ?;", (fid,))
            pruned.append(fpath)

    return total_checked, pruned
