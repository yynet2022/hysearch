"""FastAPI application for hysearch web interface and REST API.

This module provides endpoints for serving the single-page browser UI,
inspecting database status, and running hybrid/BM25/vector search queries.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from hysearch.core.config import DEFAULT_DB_PATH, DEFAULT_RRF_K
from hysearch.core.database import (
    PathLike,
    get_connection,
    get_db_status,
    init_db,
)
from hysearch.core.embedder import EmbedderProtocol, FastEmbedder
from hysearch.core.models import SearchResponse
from hysearch.core.searcher import SearchMode, search
from hysearch.utils.text import parse_datetime_str

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


def create_app(
    db_path: PathLike = DEFAULT_DB_PATH,
    embedder: EmbedderProtocol | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        db_path: Path to the SQLite database file.
        embedder: Optional pre-initialized embedder instance. If None,
            FastEmbedder will be lazily initialized and cached on first query.

    Returns:
        FastAPI: Configured FastAPI web application.
    """
    app = FastAPI(
        title="hysearch Web Server",
        description="Local Hybrid Search Engine powered by SQLite.",
    )
    resolved_db = Path(db_path)
    state: dict[str, EmbedderProtocol | None] = {"embedder": embedder}

    def get_cached_embedder() -> EmbedderProtocol:
        cached = state["embedder"]
        if cached is None:
            cached = FastEmbedder()
            state["embedder"] = cached
        return cached

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> HTMLResponse:
        """Serve the interactive single-page web UI."""
        html_content = _TEMPLATE_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        """Return index statistics and database metadata."""
        if not resolved_db.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Database file '{resolved_db}' does not exist. "
                    "Run 'hysearch index' first."
                ),
            )
        return get_db_status(resolved_db)

    @app.get("/api/query", response_model=SearchResponse)
    def api_query(
        q: str = Query(..., description="Search query text or keywords"),
        top: int = Query(5, ge=1, le=500, description="Top K results"),
        mode: str = Query("hybrid", description="hybrid, bm25, or vector"),
        rrf_k: int = Query(DEFAULT_RRF_K, ge=1, description="RRF k constant"),
        path: str | None = Query(None, description="Path prefix filter"),
        glob: str | None = Query(None, description="File glob filter"),
        since: str | None = Query(None, description="Modified since date"),
        after: str | None = Query(None, description="Alias for since"),
        until: str | None = Query(None, description="Modified until date"),
        before: str | None = Query(None, description="Alias for until"),
    ) -> SearchResponse:
        """Execute a search query against the SQLite database."""
        if not q.strip():
            raise HTTPException(
                status_code=400, detail="Search query 'q' must not be empty."
            )

        if mode not in ("hybrid", "bm25", "vector"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid mode '{mode}'. "
                    "Choose from hybrid, bm25, or vector."
                ),
            )

        if since and after and since != after:
            raise HTTPException(
                status_code=400,
                detail="Conflicting values provided for since and after.",
            )

        if until and before and until != before:
            raise HTTPException(
                status_code=400,
                detail="Conflicting values provided for until and before.",
            )

        effective_since = since or after
        effective_until = until or before

        try:
            since_dt = (
                parse_datetime_str(effective_since, "since")
                if effective_since
                else None
            )
            until_dt = (
                parse_datetime_str(effective_until, "until")
                if effective_until
                else None
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not resolved_db.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Database file '{resolved_db}' does not exist. "
                    "Run 'hysearch index' first."
                ),
            )

        search_mode: SearchMode = mode  # type: ignore[assignment]
        active_embedder = (
            get_cached_embedder() if search_mode != "bm25" else None
        )

        with get_connection(resolved_db) as conn:
            init_db(conn)
            try:
                return search(
                    conn=conn,
                    query=q,
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
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
