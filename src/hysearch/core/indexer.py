"""Indexing pipeline for scanning, chunking, and vectorizing documents.

This module crawls markdown and text files, calculates SHA-256 digests
to detect changes, splits content into chunks, and stores full-text and
vector representations in SQLite.
"""

import os
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
    TOKENIZER_TRIGRAM,
)
from hysearch.core.database import (
    get_db_embed_dim,
    get_db_embed_model,
    get_db_tokenizer,
)
from hysearch.core.embedder import EmbedderProtocol, FastEmbedder
from hysearch.core.models import IndexingStats
from hysearch.core.tokenizer import format_fts_content, resolve_tokenizer

PathLike = str | Path
ProgressCallback = Callable[[int, int, Path], None]
FileActionCallback = Callable[[Path, bool], None]
PruneCallback = Callable[[str, bool], None]


def _normalize_exclude_dirs(
    exclude_dirs: Sequence[str] | None,
) -> tuple[set[str], list[str]]:
    """Normalize excluded directory patterns into names and relative paths."""
    names: set[str] = set()
    rel_paths: list[str] = []
    if not exclude_dirs:
        return names, rel_paths

    for raw in exclude_dirs:
        clean = raw.strip().replace("\\", "/")
        while clean.startswith("./"):
            clean = clean[2:]
        clean = clean.strip("/")
        if not clean:
            continue
        if "/" in clean:
            rel_paths.append(clean)
        else:
            names.add(clean)
    return names, rel_paths


def _is_dir_excluded(
    dir_path: Path,
    root_dir: Path,
    exclude_names: set[str],
    exclude_rel_paths: list[str],
) -> bool:
    """Check whether a directory matches any exclusion rule."""
    if dir_path.name in exclude_names:
        return True
    rel_posix = dir_path.relative_to(root_dir).as_posix()
    return any(
        rel_posix == pat or rel_posix.startswith(f"{pat}/")
        for pat in exclude_rel_paths
    )


def scan_directory(
    target_dir: PathLike,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
    exclude_dirs: Sequence[str] | None = None,
) -> list[Path]:
    """Recursively discover files matching supported extensions.

    Args:
        target_dir: Root directory path to crawl.
        extensions: Sequence of file extensions to include (e.g. ['.md']).
        exclude_dirs: Optional directory names or relative paths to exclude.

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
    exclude_names, exclude_rel_paths = _normalize_exclude_dirs(exclude_dirs)
    matched: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(path):
        current_dir = Path(dirpath)
        if exclude_names or exclude_rel_paths:
            dirnames[:] = [
                d
                for d in dirnames
                if not _is_dir_excluded(
                    current_dir / d, path, exclude_names, exclude_rel_paths
                )
            ]
        for fname in filenames:
            file_p = current_dir / fname
            if file_p.suffix.lower() in ext_set:
                matched.append(file_p)

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
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='files';"
    )
    if cursor.fetchone() is None:
        return {}
    cursor.execute("SELECT path, hash FROM files;")
    return {row[0]: row[1] for row in cursor.fetchall()}


def index_file(
    conn: sqlite3.Connection,
    file_path: Path,
    embedder: EmbedderProtocol,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    current_hash: str | None = None,
    tokenizer: str | None = None,
) -> int:
    """Index or re-index a single document file.

    Args:
        conn: Open SQLite connection.
        file_path: Path to the target file.
        embedder: Vector embedding instance.
        chunk_size: Target character count per chunk.
        overlap: Character overlap count.
        current_hash: Precomputed SHA-256 hex digest, or None to compute.
        tokenizer: Optional active tokenizer name.

    Returns:
        int: Number of chunks created and stored for this file.
    """
    active_tokenizer = tokenizer or get_db_tokenizer(conn)
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

        # chunks テーブル、fts_chunks、および vec_chunks テーブルに挿入
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

            # Sudachi 等の場合は分かち書き済みテキストを fts_chunks に登録
            if active_tokenizer != TOKENIZER_TRIGRAM:
                fts_text = format_fts_content(chunk.content, active_tokenizer)
                cursor.execute(
                    "INSERT INTO fts_chunks (rowid, content) VALUES (?, ?);",
                    (chunk_id, fts_text),
                )

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
    exclude_dirs: Sequence[str] | None = None,
    dry_run: bool = False,
    tokenizer: str | None = None,
    embed_model: str | None = None,
    embed_dim: int | None = None,
    on_progress: ProgressCallback | None = None,
    on_file: FileActionCallback | None = None,
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
        exclude_dirs: Optional directory names or relative paths to exclude.
        dry_run: If True, scan and calculate stats without modifying DB.
        tokenizer: Optional explicit tokenizer name for dry-run reporting.
        embed_model: Optional explicit embedding model name.
        embed_dim: Optional explicit embedding vector dimension.
        on_progress: Optional callback receiving (current_index, total_count,
            current_file_path).
        on_file: Optional callback receiving (file_path, is_skipped) for each
            discovered file.

    Returns:
        IndexingStats: Summary statistics of the indexing process.
    """
    start_time = time.perf_counter()
    active_tokenizer = resolve_tokenizer(
        tokenizer if tokenizer is not None else get_db_tokenizer(conn)
    )
    active_model = (
        embed_model if embed_model is not None else get_db_embed_model(conn)
    )
    active_dim = embed_dim if embed_dim is not None else get_db_embed_dim(conn)
    active_embedder: EmbedderProtocol | None = (
        embedder
        if embedder is not None
        else (
            None
            if dry_run
            else FastEmbedder(model_name=active_model, dimension=active_dim)
        )
    )
    effective_dim = (
        active_embedder.dimension
        if active_embedder is not None
        else active_dim
    )

    scan_exts = extensions if extensions is not None else SUPPORTED_EXTENSIONS
    files = scan_directory(
        target_dir, extensions=scan_exts, exclude_dirs=exclude_dirs
    )
    stored_hashes = get_stored_file_hashes(conn)

    total_files = len(files)
    stats = IndexingStats(
        files_scanned=total_files,
        tokenizer=active_tokenizer,
        embed_model=active_model,
        embed_dim=effective_dim,
    )

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
            if on_file is not None:
                on_file(file_path, True)
            continue

        if on_file is not None:
            on_file(file_path, False)

        if dry_run:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            stats.files_indexed += 1
            stats.chunks_created += len(chunks)
            continue

        assert active_embedder is not None
        created = index_file(
            conn=conn,
            file_path=file_path,
            embedder=active_embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            current_hash=current_hash,
            tokenizer=active_tokenizer,
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
