"""Tests for the MCP server tools (status, query) and CLI mcp command."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from typer.testing import CliRunner

from hysearch.cli.main import app as cli_app
from hysearch.core.database import get_connection, init_db
from hysearch.core.embedder import DummyEmbedder
from hysearch.core.indexer import index_directory
from hysearch.mcp.server import (
    READ_ONLY_ANNOTATIONS,
    create_mcp_server,
    run_mcp_server,
)

runner = CliRunner()


@pytest.fixture()
def indexed_db(tmp_path: Path, dummy_embedder: DummyEmbedder) -> Path:
    """Create and populate a temporary SQLite database for MCP tests."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text(
        "# Python Guide\nPython is a great programming language for search.",
        encoding="utf-8",
    )
    (docs_dir / "notes.txt").write_text(
        "SQLite FTS5 and sqlite-vec enable fast local hybrid search.",
        encoding="utf-8",
    )
    db_file = tmp_path / "mcp_test.db"
    with get_connection(db_file) as conn:
        init_db(conn)
        index_directory(conn, docs_dir, embedder=dummy_embedder)
    return db_file


def _extract_tool_text(call_result: Any) -> str:
    """Extract the first TextContent string from an MCP CallToolResult."""
    return str(call_result.content[0].text)


def test_mcp_tools_registration_and_annotations(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify tools are registered with READ_ONLY_ANNOTATIONS."""
    mcp = create_mcp_server(db_path=indexed_db, embedder=dummy_embedder)
    tools = asyncio.run(mcp.list_tools())
    tool_map = {t.name: t for t in tools}

    assert set(tool_map.keys()) == {"status", "query"}
    for name in ("status", "query"):
        ann = tool_map[name].annotations
        assert ann is not None
        assert ann.read_only_hint is True
        assert ann.destructive_hint is False
        assert ann.idempotent_hint is True
        assert ann.open_world_hint is False
        assert ann == READ_ONLY_ANNOTATIONS


def test_mcp_status_and_query_tools(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify MCP status and query tools return valid JSON responses."""
    mcp = create_mcp_server(db_path=indexed_db, embedder=dummy_embedder)

    # 1. status tool
    res_status = asyncio.run(mcp.call_tool("status", {}))
    status_data = json.loads(_extract_tool_text(res_status))
    assert status_data["Files Indexed"] == 2
    assert status_data["Total Chunks"] >= 2
    assert status_data["Tokenizer"] == "trigram"
    assert "Embedding Model" in status_data
    assert status_data["Embedding Dim"] == 384

    # 2. query tool (hybrid mode)
    res_hyb = asyncio.run(
        mcp.call_tool(
            "query",
            {
                "query": "Python",
                "top": 3,
                "mode": "hybrid",
                "rrf_k": 60,
            },
        )
    )
    hyb_data = json.loads(_extract_tool_text(res_hyb))
    assert hyb_data["query"] == "Python"
    assert hyb_data["total_hits"] >= 1
    assert len(hyb_data["hits"]) >= 1

    # 3. query tool (bm25 mode with path, glob, and date filters)
    res_bm25 = asyncio.run(
        mcp.call_tool(
            "query",
            {
                "query": "SQLite",
                "mode": "bm25",
                "path": "docs",
                "glob": "*.txt",
                "since": "2020-01-01",
                "until": "2099-12-31",
            },
        )
    )
    bm25_data = json.loads(_extract_tool_text(res_bm25))
    assert bm25_data["total_hits"] == 1
    assert "notes.txt" in bm25_data["hits"][0]["file_path"]


def test_mcp_lazy_embedder_caching(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify FastEmbedder is lazily initialized and cached across queries."""
    with patch(
        "hysearch.mcp.server.FastEmbedder", return_value=dummy_embedder
    ) as mock_cls:
        mcp = create_mcp_server(db_path=indexed_db, embedder=None)

        asyncio.run(
            mcp.call_tool("query", {"query": "Python", "mode": "vector"})
        )
        asyncio.run(
            mcp.call_tool("query", {"query": "SQLite", "mode": "hybrid"})
        )
        mock_cls.assert_called_once()


def test_mcp_tools_validation_and_missing_db_errors(
    indexed_db: Path, tmp_path: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify MCP tools raise ToolError with descriptive messages on error."""
    missing_db = tmp_path / "missing.db"
    mcp_missing = create_mcp_server(
        db_path=missing_db, embedder=dummy_embedder
    )

    with pytest.raises(ToolError, match="does not exist"):
        asyncio.run(mcp_missing.call_tool("status", {}))

    with pytest.raises(ToolError, match="does not exist"):
        asyncio.run(mcp_missing.call_tool("query", {"query": "Python"}))

    mcp = create_mcp_server(db_path=indexed_db, embedder=dummy_embedder)

    # Empty query
    with pytest.raises(ToolError, match="must not be empty"):
        asyncio.run(mcp.call_tool("query", {"query": "   "}))

    # Invalid top (< 1)
    with pytest.raises(ToolError, match="'top' must be at least 1"):
        asyncio.run(mcp.call_tool("query", {"query": "Python", "top": 0}))

    # Invalid rrf_k (< 1)
    with pytest.raises(ToolError, match="'rrf_k' must be at least 1"):
        asyncio.run(mcp.call_tool("query", {"query": "Python", "rrf_k": 0}))

    # Invalid mode
    with pytest.raises(ToolError, match="Invalid mode"):
        asyncio.run(
            mcp.call_tool("query", {"query": "Python", "mode": "invalid"})
        )

    # Conflicting since and after
    with pytest.raises(ToolError, match="Conflicting values"):
        asyncio.run(
            mcp.call_tool(
                "query",
                {
                    "query": "Python",
                    "since": "2026-01-01",
                    "after": "2026-02-01",
                },
            )
        )

    # Conflicting until and before
    with pytest.raises(ToolError, match="Conflicting values"):
        asyncio.run(
            mcp.call_tool(
                "query",
                {
                    "query": "Python",
                    "until": "2026-01-01",
                    "before": "2026-02-01",
                },
            )
        )

    # Invalid date format
    with pytest.raises(ToolError, match="Invalid date format"):
        asyncio.run(
            mcp.call_tool("query", {"query": "Python", "since": "bad-date"})
        )

    # RuntimeError during search (e.g. missing Sudachi dict)
    with (
        patch(
            "hysearch.mcp.server.search",
            side_effect=RuntimeError("Tokenizer unavailable"),
        ),
        pytest.raises(ToolError, match="Tokenizer unavailable"),
    ):
        asyncio.run(mcp.call_tool("query", {"query": "Python"}))


def test_run_mcp_server_and_cli_command(
    indexed_db: Path, tmp_path: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify run_mcp_server and 'hysearch mcp' CLI command execution."""
    missing_db = tmp_path / "nonexistent.db"

    # 1. run_mcp_server with missing DB raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_mcp_server(db_path=missing_db, embedder=dummy_embedder)

    # 2. run_mcp_server with existing DB calls mcp.run(transport="stdio")
    mock_mcp = MagicMock()
    with patch("hysearch.mcp.server.create_mcp_server", return_value=mock_mcp):
        run_mcp_server(db_path=indexed_db, embedder=dummy_embedder)
        mock_mcp.run.assert_called_once_with(transport="stdio")

    # 3. CLI 'hysearch mcp' with missing DB -> exit code 1
    res_missing = runner.invoke(cli_app, ["mcp", "--db", str(missing_db)])
    assert res_missing.exit_code == 1
    assert "does not exist" in (res_missing.stdout + res_missing.stderr)

    # 4. CLI 'hysearch mcp' with existing DB -> invokes run_mcp_server
    with patch("hysearch.cli.main.run_mcp_server") as mock_run:
        res_ok = runner.invoke(cli_app, ["mcp", "--db", str(indexed_db)])
        assert res_ok.exit_code == 0
        mock_run.assert_called_once_with(db_path=indexed_db)

    # 5. CLI 'hysearch mcp' handles RuntimeError -> exit code 1
    with patch(
        "hysearch.cli.main.run_mcp_server",
        side_effect=RuntimeError("MCP stdio failure"),
    ):
        res_err = runner.invoke(cli_app, ["mcp", "--db", str(indexed_db)])
        assert res_err.exit_code == 1
        assert "MCP stdio failure" in (res_err.stdout + res_err.stderr)


def test_mcp_read_only_db(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify MCP status and query tools succeed on chmod 444 database."""
    indexed_db.chmod(0o444)
    try:
        mcp = create_mcp_server(db_path=indexed_db, embedder=dummy_embedder)

        res_status = asyncio.run(mcp.call_tool("status", {}))
        status_data = json.loads(_extract_tool_text(res_status))
        assert status_data["Files Indexed"] == 2

        res_query = asyncio.run(mcp.call_tool("query", {"query": "Python"}))
        query_data = json.loads(_extract_tool_text(res_query))
        assert query_data["total_hits"] >= 1
    finally:
        indexed_db.chmod(0o666)
