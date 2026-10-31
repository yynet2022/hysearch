"""Tests for indexing pipeline and incremental change detection."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from hysearch.core.database import create_connection, init_db
from hysearch.core.embedder import DummyEmbedder
from hysearch.core.indexer import (
    get_stored_file_hashes,
    index_directory,
    index_file,
    prune_missing_files,
    scan_directory,
)


def test_scan_directory_filters_and_recursion(tmp_path: Path) -> None:
    """Verify scan_directory only collects supported file extensions."""
    (tmp_path / "doc1.md").write_text("Markdown 1", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Text note", encoding="utf-8")
    (tmp_path / "code.py").write_text("print()", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "guide.markdown").write_text("Sub guide", encoding="utf-8")
    (sub_dir / "ignored.bin").write_bytes(b"\x00\x01")

    files = scan_directory(tmp_path)
    names = [f.name for f in files]
    assert "doc1.md" in names
    assert "notes.txt" in names
    assert "code.py" in names
    assert "guide.markdown" in names
    assert "image.png" not in names
    assert "ignored.bin" not in names


def test_index_directory_incremental_and_force(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify incremental indexing skips unchanged files and force works."""
    file1 = tmp_path / "doc1.md"
    file2 = tmp_path / "doc2.txt"
    file1.write_text("First document content.", encoding="utf-8")
    file2.write_text("Second document content.", encoding="utf-8")

    # 初回インデックス
    stats1 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
    )
    assert stats1.files_scanned == 2
    assert stats1.files_indexed == 2
    assert stats1.files_skipped == 0
    assert stats1.chunks_created == 2

    # 変更なしでの2回目インデックス（スキップされること）
    stats2 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
    )
    assert stats2.files_scanned == 2
    assert stats2.files_indexed == 0
    assert stats2.files_skipped == 2
    assert stats2.chunks_created == 0

    # 1ファイルのみ更新
    file1.write_text("Updated first document content.", encoding="utf-8")
    stats3 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
    )
    assert stats3.files_scanned == 2
    assert stats3.files_indexed == 1
    assert stats3.files_skipped == 1

    # force=True で全再構築
    stats4 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
        force=True,
    )
    assert stats4.files_scanned == 2
    assert stats4.files_indexed == 2
    assert stats4.files_skipped == 0


def test_get_stored_file_hashes(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify retrieving hash map from database."""
    file1 = tmp_path / "test.md"
    file1.write_text("Sample", encoding="utf-8")

    index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
    )
    hashes = get_stored_file_hashes(memory_db)
    assert len(hashes) == 1
    assert str(file1.resolve()) in hashes


def test_index_directory_on_progress_callback(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify on_progress callback receives correct indices and file paths."""
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "b.md").write_text("B", encoding="utf-8")

    calls: list[tuple[int, int, str]] = []

    def callback(current: int, total: int, file_path: Path) -> None:
        calls.append((current, total, file_path.name))

    index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        on_progress=callback,
    )

    assert len(calls) == 2
    assert calls[0] == (1, 2, "a.md")
    assert calls[1] == (2, 2, "b.md")


def test_scan_directory_errors(tmp_path: Path) -> None:
    """Verify scan_directory raises appropriate errors for invalid paths."""
    missing = tmp_path / "nonexistent_dir"
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        scan_directory(missing)

    regular_file = tmp_path / "file.txt"
    regular_file.write_text("content", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        scan_directory(regular_file)


def test_index_file_empty(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify indexing empty file creates file metadata with 0 chunks."""
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("", encoding="utf-8")

    count = index_file(
        conn=memory_db,
        file_path=empty_file,
        embedder=dummy_embedder,
    )
    assert count == 0

    hashes = get_stored_file_hashes(memory_db)
    assert str(empty_file.resolve()) in hashes


def test_index_directory_with_extensions_filter(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify index_directory with custom extensions filters files."""
    (tmp_path / "doc.md").write_text("Markdown doc", encoding="utf-8")
    (tmp_path / "script.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "style.css").write_text("body {}", encoding="utf-8")

    # 1. .py のみインデックス
    stats_py = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        extensions=[".py"],
    )
    assert stats_py.files_scanned == 1
    assert stats_py.files_indexed == 1
    hashes = get_stored_file_hashes(memory_db)
    assert len(hashes) == 1
    assert str((tmp_path / "script.py").resolve()) in hashes

    # 2. 続いて .md をインデックス（既存の .py はそのまま残る）
    stats_md = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        extensions=[".md"],
    )
    assert stats_md.files_scanned == 1
    assert stats_md.files_indexed == 1
    hashes_after = get_stored_file_hashes(memory_db)
    assert len(hashes_after) == 2
    assert str((tmp_path / "script.py").resolve()) in hashes_after
    assert str((tmp_path / "doc.md").resolve()) in hashes_after


def test_prune_missing_files(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify prune_missing_files removes orphaned files and cascades."""
    file_keep = tmp_path / "keep.md"
    file_keep.write_text("Keep me", encoding="utf-8")
    file_remove = tmp_path / "remove.md"
    file_remove.write_text("Delete me", encoding="utf-8")

    index_directory(memory_db, tmp_path, dummy_embedder)

    assert memory_db.execute("SELECT COUNT(*) FROM files;").fetchone()[0] == 2
    assert memory_db.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 2

    # 物理ファイルを削除
    file_remove.unlink()

    # 1. dry-run モードの検証（on_missing 省略時および指定時）
    total_no_cb, dry_no_cb = prune_missing_files(memory_db, dry_run=True)
    assert total_no_cb == 2
    assert len(dry_no_cb) == 1

    callback_calls: list[tuple[str, bool]] = []

    def on_missing(path_str: str, dry_run: bool) -> None:
        callback_calls.append((path_str, dry_run))

    total, dry_pruned = prune_missing_files(
        memory_db, dry_run=True, on_missing=on_missing
    )
    assert total == 2
    assert len(dry_pruned) == 1
    assert str(file_remove.resolve()) in dry_pruned
    assert len(callback_calls) == 1
    assert callback_calls[0] == (str(file_remove.resolve()), True)
    # DBからはまだ消えていない
    assert memory_db.execute("SELECT COUNT(*) FROM files;").fetchone()[0] == 2

    # 2. 通常実行モードの検証
    callback_calls.clear()
    total, pruned = prune_missing_files(
        memory_db, dry_run=False, on_missing=on_missing
    )
    assert total == 2
    assert len(pruned) == 1
    assert str(file_remove.resolve()) in pruned
    assert len(callback_calls) == 1
    assert callback_calls[0] == (str(file_remove.resolve()), False)

    # DBから files, chunks, fts, vec がカスケード削除されていることを確認
    assert memory_db.execute("SELECT COUNT(*) FROM files;").fetchone()[0] == 1
    assert memory_db.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 1
    assert (
        memory_db.execute("SELECT COUNT(*) FROM fts_chunks;").fetchone()[0]
        == 1
    )
    assert (
        memory_db.execute("SELECT COUNT(*) FROM vec_chunks;").fetchone()[0]
        == 1
    )

    # 3. 再度 prune 実行（欠落ファイル0件）
    total2, pruned2 = prune_missing_files(memory_db, dry_run=False)
    assert total2 == 1
    assert pruned2 == []


def test_index_with_sudachi_tokenizer(
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify indexing inserts space-separated tokens when using Sudachi."""
    conn = create_connection(":memory:")
    init_db(conn, tokenizer="sudachi-core")

    doc = tmp_path / "jp.md"
    doc.write_text("京都で検索する。", encoding="utf-8")

    with (
        patch(
            "hysearch.core.indexer.resolve_tokenizer",
            return_value="sudachi-core",
        ),
        patch(
            "hysearch.core.indexer.format_fts_content",
            return_value="京都 で 検索 する",
        ),
    ):
        stats = index_directory(conn, tmp_path, dummy_embedder)

    assert stats.tokenizer == "sudachi-core"
    assert stats.files_indexed == 1

    # chunks holds raw text, fts_chunks holds segmented text
    raw_content = conn.execute("SELECT content FROM chunks;").fetchone()[0]
    fts_content = conn.execute("SELECT content FROM fts_chunks;").fetchone()[0]
    assert raw_content == "京都で検索する。"
    assert fts_content == "京都 で 検索 する"
    conn.close()


def test_scan_and_index_with_exclude_dirs_and_dry_run(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify exclude_dirs prunes directories and dry_run does not write DB."""
    (tmp_path / "keep.md").write_text("Main document.", encoding="utf-8")

    venv_dir = tmp_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "pkg.py").write_text("x = 1", encoding="utf-8")

    sub_ignored = tmp_path / "docs" / "drafts"
    sub_ignored.mkdir(parents=True)
    (sub_ignored / "secret.md").write_text("Draft", encoding="utf-8")

    sub_active = tmp_path / "docs" / "public"
    sub_active.mkdir(parents=True)
    (sub_active / "guide.md").write_text("Public guide", encoding="utf-8")

    # 1. scan_directory with directory name and relative path exclusions
    scanned = scan_directory(
        tmp_path,
        exclude_dirs=["", "./.venv/", "docs/drafts"],
    )
    scanned_names = [p.name for p in scanned]
    assert scanned_names == ["guide.md", "keep.md"]

    # Also test when only a relative path with slash is excluded
    scanned_rel_only = scan_directory(
        tmp_path,
        exclude_dirs=["docs/drafts"],
    )
    assert "secret.md" not in [p.name for p in scanned_rel_only]
    assert "pkg.py" in [p.name for p in scanned_rel_only]

    # 2. Dry-run on uninitialized ":memory:" connection
    uninit_conn = create_connection(":memory:")
    file_events: list[tuple[str, bool]] = []

    def on_file(fpath: Path, skipped: bool) -> None:
        file_events.append((fpath.name, skipped))

    dry_stats = index_directory(
        conn=uninit_conn,
        target_dir=tmp_path,
        exclude_dirs=[".venv", "docs/drafts"],
        dry_run=True,
        on_file=on_file,
    )
    assert dry_stats.files_scanned == 2
    assert dry_stats.files_indexed == 2
    assert dry_stats.files_skipped == 0
    assert dry_stats.chunks_created == 2
    assert file_events == [("guide.md", False), ("keep.md", False)]
    uninit_conn.close()

    # 3. Actual index followed by dry-run to verify skipped file reporting
    index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        exclude_dirs=[".venv", "docs/drafts"],
    )
    assert memory_db.execute("SELECT COUNT(*) FROM files;").fetchone()[0] == 2

    file_events.clear()
    dry_stats2 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        exclude_dirs=[".venv", "docs/drafts"],
        dry_run=True,
        on_file=on_file,
    )
    assert dry_stats2.files_scanned == 2
    assert dry_stats2.files_indexed == 0
    assert dry_stats2.files_skipped == 2
    assert file_events == [("guide.md", True), ("keep.md", True)]
