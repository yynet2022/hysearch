"""Tests for indexing pipeline and incremental change detection."""

import sqlite3
from pathlib import Path

import pytest

from hysearch.core.embedder import DummyEmbedder
from hysearch.core.indexer import (
    get_stored_file_hashes,
    index_directory,
    index_file,
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
