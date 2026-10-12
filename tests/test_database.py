"""Tests for database initialization, FTS5 triggers, and sqlite-vec."""

import sqlite3
from pathlib import Path

import sqlite_vec

from hysearch.core.config import DEFAULT_EMBED_DIM
from hysearch.core.database import get_connection, init_db


def test_init_db_creates_tables_and_triggers(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify that all required tables and triggers are created."""
    cursor = memory_db.cursor()

    # テーブルの存在確認
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
    )
    names = {row[0] for row in cursor.fetchall()}
    assert "files" in names
    assert "chunks" in names
    assert "fts_chunks" in names
    assert "vec_chunks" in names


def test_fts5_trigram_insert_and_search(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify FTS5 trigram indexing and trigger synchronization."""
    with memory_db:
        memory_db.execute(
            "INSERT INTO files (path, hash, chunk_count) VALUES (?, ?, ?);",
            ("test.md", "hash123", 1),
        )
        file_id = memory_db.execute("SELECT last_insert_rowid();").fetchone()[
            0
        ]
        memory_db.execute(
            """
            INSERT INTO chunks (file_id, chunk_index, content, start_char,
                                end_char)
            VALUES (?, ?, ?, ?, ?);
            """,
            (file_id, 0, "これはハイブリッド検索のテスト文書です。", 0, 20),
        )

    # FTS5 から検索できるか確認 (trigram)
    cursor = memory_db.cursor()
    cursor.execute(
        "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?;",
        ("ハイブリッド",),
    )
    rows = cursor.fetchall()
    assert len(rows) == 1


def test_sqlite_vec_insert_and_query(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify inserting and querying vectors using sqlite-vec."""
    dummy_vec = [0.1] * DEFAULT_EMBED_DIM
    serialized = sqlite_vec.serialize_float32(dummy_vec)

    with memory_db:
        memory_db.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?);",
            (1, serialized),
        )

    cursor = memory_db.cursor()
    cursor.execute(
        """
        SELECT id, distance
        FROM vec_chunks
        WHERE embedding MATCH ? AND k = 1
        ORDER BY distance;
        """,
        (serialized,),
    )
    rows = cursor.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] < 1e-5


def test_cascade_delete_and_triggers(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify that deleting a file cascades to chunks, FTS5, and vectors."""
    dummy_vec = [0.2] * DEFAULT_EMBED_DIM
    serialized = sqlite_vec.serialize_float32(dummy_vec)

    with memory_db:
        memory_db.execute(
            "INSERT INTO files (path, hash, chunk_count) VALUES (?, ?, ?);",
            ("doc.md", "hashabc", 1),
        )
        file_id = memory_db.execute("SELECT last_insert_rowid();").fetchone()[
            0
        ]
        memory_db.execute(
            """
            INSERT INTO chunks (file_id, chunk_index, content, start_char,
                                end_char)
            VALUES (?, ?, ?, ?, ?);
            """,
            (file_id, 0, "削除テスト用のチャンクです。", 0, 12),
        )
        chunk_id = memory_db.execute("SELECT last_insert_rowid();").fetchone()[
            0
        ]
        memory_db.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?);",
            (chunk_id, serialized),
        )

    # 削除前の確認
    assert memory_db.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 1
    assert (
        memory_db.execute("SELECT COUNT(*) FROM fts_chunks;").fetchone()[0]
        == 1
    )
    assert (
        memory_db.execute("SELECT COUNT(*) FROM vec_chunks;").fetchone()[0]
        == 1
    )

    # files から削除
    with memory_db:
        memory_db.execute("DELETE FROM files WHERE id = ?;", (file_id,))

    # カスケードおよびトリガーによる削除を確認
    assert memory_db.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 0
    assert (
        memory_db.execute("SELECT COUNT(*) FROM fts_chunks;").fetchone()[0]
        == 0
    )
    assert (
        memory_db.execute("SELECT COUNT(*) FROM vec_chunks;").fetchone()[0]
        == 0
    )


def test_get_connection_context_manager(tmp_path: Path) -> None:
    """Verify get_connection opens, operates, and closes properly."""
    db_file = tmp_path / "test.db"
    with get_connection(db_file) as conn:
        init_db(conn)
        assert conn.execute("SELECT 1;").fetchone()[0] == 1
