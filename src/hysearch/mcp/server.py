"""MCP server implementation providing read-only query and status tools.

This module configures an MCPServer (FastMCP v2) over stdio transport,
exposing read-only tools for checking index status and running hybrid,
BM25, or vector search queries with full metadata filtering.
"""

import json
import sqlite3
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from hysearch.core.config import DEFAULT_DB_PATH, DEFAULT_RRF_K
from hysearch.core.database import (
    PathLike,
    get_connection,
    get_db_embed_dim,
    get_db_embed_model,
    get_db_status,
    init_db,
)
from hysearch.core.embedder import EmbedderProtocol, FastEmbedder
from hysearch.core.searcher import search, validate_and_parse_search_params

# 読み取り専用ツールのMCPアノテーション定義
READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def create_mcp_server(
    db_path: PathLike = DEFAULT_DB_PATH,
    embedder: EmbedderProtocol | None = None,
) -> MCPServer:
    """Create and configure the MCPServer instance with status and query tools.

    Args:
        db_path: Path to the SQLite database file.
        embedder: Optional pre-initialized embedder instance. If None,
            FastEmbedder is lazily initialized and cached on first query.

    Returns:
        MCPServer: Configured MCP server instance.
    """
    mcp = MCPServer(
        name="hysearch",
        instructions=(
            "Local hybrid search engine (BM25 + vector similarity) "
            "powered by SQLite."
        ),
    )
    resolved_db = Path(db_path)
    state: dict[str, EmbedderProtocol | None] = {"embedder": embedder}

    def get_cached_embedder(conn: sqlite3.Connection) -> EmbedderProtocol:
        cached = state["embedder"]
        if cached is None:
            cached = FastEmbedder(
                model_name=get_db_embed_model(conn),
                dimension=get_db_embed_dim(conn),
            )
            state["embedder"] = cached
        return cached

    @mcp.tool(
        name="status",
        description=(
            "Return index statistics and database metadata "
            "(tokenizer, embedding model, file/chunk counts) as JSON."
        ),
        annotations=READ_ONLY_ANNOTATIONS,
    )
    def status_tool() -> str:
        """Retrieve index statistics and database metadata as JSON."""
        if not resolved_db.exists():
            raise ToolError(
                f"Database file '{resolved_db}' does not exist. "
                "Run 'hysearch index' first."
            )
        status_data = get_db_status(resolved_db)
        return json.dumps(status_data, ensure_ascii=False, indent=2)

    @mcp.tool(
        name="query",
        description=(
            "Execute a hybrid, BM25, or vector search query against the "
            "indexed SQLite database and return ranked hits as JSON."
        ),
        annotations=READ_ONLY_ANNOTATIONS,
    )
    def query_tool(
        query: str,
        top: int = 5,
        mode: str = "hybrid",
        rrf_k: int = DEFAULT_RRF_K,
        path: str | None = None,
        glob: str | None = None,
        since: str | None = None,
        after: str | None = None,
        until: str | None = None,
        before: str | None = None,
    ) -> str:
        """Execute a search query with optional filters and return JSON."""
        if not query.strip():
            raise ToolError("Search query 'query' must not be empty.")
        if top < 1:
            raise ToolError("Parameter 'top' must be at least 1.")
        if rrf_k < 1:
            raise ToolError("Parameter 'rrf_k' must be at least 1.")

        try:
            search_mode, since_dt, until_dt = validate_and_parse_search_params(
                mode=mode,
                since=since,
                after=after,
                until=until,
                before=before,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        if not resolved_db.exists():
            raise ToolError(
                f"Database file '{resolved_db}' does not exist. "
                "Run 'hysearch index' first."
            )

        with get_connection(resolved_db) as conn:
            init_db(conn)
            active_embedder = (
                get_cached_embedder(conn) if search_mode != "bm25" else None
            )
            try:
                response = search(
                    conn=conn,
                    query=query,
                    embedder=active_embedder,
                    mode=search_mode,
                    top_k=top,
                    rrf_k=rrf_k,
                    path_filter=path,
                    glob_filter=glob,
                    since=since_dt,
                    until=until_dt,
                )
            except (ValueError, RuntimeError) as exc:
                raise ToolError(str(exc)) from exc

        return response.model_dump_json(indent=2)

    return mcp


def run_mcp_server(
    db_path: PathLike = DEFAULT_DB_PATH,
    embedder: EmbedderProtocol | None = None,
) -> None:
    """Start the MCP server over stdio transport.

    Args:
        db_path: Path to the SQLite database file.
        embedder: Optional pre-initialized embedder instance.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    resolved_db = Path(db_path)
    if not resolved_db.exists():
        raise FileNotFoundError(
            f"Database file '{resolved_db}' does not exist. "
            "Run 'hysearch index' first."
        )
    mcp = create_mcp_server(db_path=resolved_db, embedder=embedder)
    mcp.run(transport="stdio")
