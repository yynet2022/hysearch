"""SQLite database connection and schema initialization for hysearch.

This module provides functions to create database connections, load the
sqlite-vec extension, initialize schema tables (metadata, chunks, FTS5,
and vector index), and reset the database.
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from hysearch.core.config import DEFAULT_EMBED_DIM

PathLike = str | Path

# データベース初期化用 SQL スクリプト
INIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
    UNIQUE(file_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content)
        VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content)
        VALUES('delete', old.id, old.content);
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;
"""


def create_connection(db_path: PathLike) -> sqlite3.Connection:
    """Create a new SQLite connection with foreign keys and sqlite-vec loaded.

    Args:
        db_path: Path to the SQLite database file, or ':memory:'.

    Returns:
        sqlite3.Connection: Configured connection with vector extension loaded.

    Raises:
        sqlite3.OperationalError: If extension loading fails or DB is locked.
    """
    path_str = str(db_path)
    conn = sqlite3.connect(path_str)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


@contextmanager
def get_connection(
    db_path: PathLike,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connection to ensure proper closure.

    Args:
        db_path: Path to the SQLite database file or ':memory:'.

    Yields:
        sqlite3.Connection: Open database connection.
    """
    conn = create_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(
    conn: sqlite3.Connection, embed_dim: int = DEFAULT_EMBED_DIM
) -> None:
    """Initialize the required database tables, triggers, and vector tables.

    Args:
        conn: Open SQLite connection.
        embed_dim: Dimension of embeddings for the vector table.
    """
    with conn:
        conn.executescript(INIT_SCHEMA_SQL)
        vec_sql = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\n"
            f"    id INTEGER PRIMARY KEY,\n"
            f"    embedding float[{embed_dim}]\n"
            f");"
        )
        conn.execute(vec_sql)
        # vec_chunks 同期用 DELETE トリガー
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ad_vec
            AFTER DELETE ON chunks BEGIN
                DELETE FROM vec_chunks WHERE id = old.id;
            END;
            """
        )
