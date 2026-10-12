"""Pytest shared fixtures for hysearch tests."""

import sqlite3
from collections.abc import Generator

import pytest

from hysearch.core.config import DEFAULT_EMBED_DIM
from hysearch.core.database import create_connection, init_db
from hysearch.core.embedder import DummyEmbedder


@pytest.fixture
def memory_db() -> Generator[sqlite3.Connection, None, None]:
    """Provide an in-memory SQLite database initialized with full schema.

    Yields:
        sqlite3.Connection: Initialized in-memory SQLite connection.
    """
    conn = create_connection(":memory:")
    init_db(conn, embed_dim=DEFAULT_EMBED_DIM)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def dummy_embedder() -> DummyEmbedder:
    """Provide a deterministic DummyEmbedder instance for fast testing.

    Returns:
        DummyEmbedder: Mock embedder producing 384-dim vectors.
    """
    return DummyEmbedder(dimension=DEFAULT_EMBED_DIM)
