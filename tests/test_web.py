"""Tests for the FastAPI web application, port resolution, and server CLI."""

import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hysearch.cli.main import app as cli_app
from hysearch.core.database import get_connection, init_db
from hysearch.core.embedder import DummyEmbedder
from hysearch.core.indexer import index_directory
from hysearch.web.app import create_app
from hysearch.web.runner import (
    build_browser_url,
    is_port_available,
    resolve_server_port,
    run_server,
)

runner = CliRunner()


@pytest.fixture()
def indexed_db(tmp_path: Path, dummy_embedder: DummyEmbedder) -> Path:
    """Create and populate a temporary SQLite database for web tests."""
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
    db_file = tmp_path / "web_test.db"
    with get_connection(db_file) as conn:
        init_db(conn)
        index_directory(conn, docs_dir, embedder=dummy_embedder)
    return db_file


def test_web_index_and_status(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify GET / serves HTML UI and GET /api/status returns DB stats."""
    web_app = create_app(db_path=indexed_db, embedder=dummy_embedder)
    client = TestClient(web_app)

    # 1. HTML index page
    res_html = client.get("/")
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers["content-type"]
    assert "hysearch" in res_html.text

    # 2. Status endpoint
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["Files Indexed"] == 2
    assert data["Total Chunks"] >= 2
    assert data["Tokenizer"] == "trigram"


def test_web_status_and_query_missing_db(
    tmp_path: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify 404 is returned when the database file does not exist."""
    missing_db = tmp_path / "nonexistent.db"
    web_app = create_app(db_path=missing_db, embedder=dummy_embedder)
    client = TestClient(web_app)

    res_status = client.get("/api/status")
    assert res_status.status_code == 404
    assert "does not exist" in res_status.json()["detail"]

    res_query = client.get("/api/query", params={"q": "Python"})
    assert res_query.status_code == 404
    assert "does not exist" in res_query.json()["detail"]


def test_web_query_modes_and_filters(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify /api/query works with hybrid, bm25, vector, and filters."""
    web_app = create_app(db_path=indexed_db, embedder=dummy_embedder)
    client = TestClient(web_app)

    # Hybrid search
    res_hyb = client.get(
        "/api/query",
        params={"q": "Python", "mode": "hybrid", "top": 3, "rrf_k": 60},
    )
    assert res_hyb.status_code == 200
    body = res_hyb.json()
    assert body["query"] == "Python"
    assert body["total_hits"] >= 1

    # BM25 mode with glob and date filters
    res_bm25 = client.get(
        "/api/query",
        params={
            "q": "SQLite",
            "mode": "bm25",
            "glob": "*.txt",
            "since": "2020-01-01",
            "until": "2099-12-31",
        },
    )
    assert res_bm25.status_code == 200
    assert res_bm25.json()["total_hits"] == 1


def test_web_query_lazy_embedder_caching(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify FastEmbedder is lazily instantiated and cached when None."""
    with patch(
        "hysearch.web.app.FastEmbedder", return_value=dummy_embedder
    ) as mock_cls:
        web_app = create_app(db_path=indexed_db, embedder=None)
        client = TestClient(web_app)

        client.get("/api/query", params={"q": "Python", "mode": "vector"})
        client.get("/api/query", params={"q": "SQLite", "mode": "hybrid"})
        mock_cls.assert_called_once()


def test_web_query_validation_errors(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify 400 error responses for invalid parameters in /api/query."""
    web_app = create_app(db_path=indexed_db, embedder=dummy_embedder)
    client = TestClient(web_app)

    # Empty query string
    res_empty = client.get("/api/query", params={"q": "   "})
    assert res_empty.status_code == 400
    assert "must not be empty" in res_empty.json()["detail"]

    # Invalid mode
    res_mode = client.get("/api/query", params={"q": "test", "mode": "bad"})
    assert res_mode.status_code == 400
    assert "Invalid mode" in res_mode.json()["detail"]

    # Conflicting since and after
    res_since = client.get(
        "/api/query",
        params={"q": "test", "since": "2026-01-01", "after": "2026-02-01"},
    )
    assert res_since.status_code == 400
    assert "Conflicting values" in res_since.json()["detail"]

    # Conflicting until and before
    res_until = client.get(
        "/api/query",
        params={"q": "test", "until": "2026-01-01", "before": "2026-02-01"},
    )
    assert res_until.status_code == 400
    assert "Conflicting values" in res_until.json()["detail"]

    # Invalid date string
    res_date = client.get(
        "/api/query",
        params={"q": "test", "since": "not-a-date"},
    )
    assert res_date.status_code == 400
    assert "Invalid date format" in res_date.json()["detail"]

    # Search runtime error (e.g. missing tokenizer)
    with patch(
        "hysearch.web.app.search",
        side_effect=RuntimeError("Tokenizer unavailable"),
    ):
        res_err = client.get("/api/query", params={"q": "test"})
        assert res_err.status_code == 400
        assert "Tokenizer unavailable" in res_err.json()["detail"]


def test_port_availability_and_resolution() -> None:
    """Verify is_port_available and resolve_server_port logic."""
    # Out-of-range port check
    assert is_port_available("127.0.0.1", 0) is False
    assert is_port_available("127.0.0.1", 70000) is False

    with pytest.raises(ValueError, match="Invalid port number"):
        resolve_server_port("127.0.0.1", port=0)

    # Occupy an ephemeral port on 127.0.0.1 and test conflict & auto-increment
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        occupied_port = s.getsockname()[1]

        assert is_port_available("127.0.0.1", occupied_port) is False

        # Explicit port that is occupied -> raises RuntimeError
        with pytest.raises(RuntimeError, match="already in use"):
            resolve_server_port("127.0.0.1", port=occupied_port)

    # Bind failure branch when connect_ex != 0 but bind raises OSError
    with (
        patch("socket.socket.connect_ex", return_value=1),
        patch("socket.socket.bind", side_effect=OSError("bind failed")),
    ):
        assert is_port_available("127.0.0.1", 8888) is False

    # Available port branch when connect_ex != 0 and bind succeeds
    with (
        patch("socket.socket.connect_ex", return_value=1),
        patch("socket.socket.bind", return_value=None),
    ):
        assert is_port_available("127.0.0.1", 8888) is True

    # Explicit port that is available -> returns that port
    with patch("hysearch.web.runner.is_port_available", return_value=True):
        assert resolve_server_port("127.0.0.1", port=9000) == 9000

    # Auto-search: 8888 occupied, 8889 free -> returns 8889
    with patch(
        "hysearch.web.runner.is_port_available",
        side_effect=lambda h, p: p == 8889,
    ):
        assert resolve_server_port("127.0.0.1", port=None) == 8889

    # Auto-search: all ports occupied -> raises RuntimeError
    with (
        patch("hysearch.web.runner.is_port_available", return_value=False),
        pytest.raises(RuntimeError, match="Could not find an available port"),
    ):
        resolve_server_port("127.0.0.1", port=None, max_tries=3)


def test_build_browser_url_and_run_server(indexed_db: Path) -> None:
    """Verify build_browser_url and run_server execution with mocks."""
    assert build_browser_url("127.0.0.1", 8888) == "http://localhost:8888"
    assert build_browser_url("0.0.0.0", 8889) == "http://localhost:8889"
    assert (
        build_browser_url("192.168.1.10", 9000) == "http://192.168.1.10:9000"
    )

    with (
        patch("hysearch.web.runner.resolve_server_port", return_value=8888),
        patch("hysearch.web.runner.threading.Timer") as mock_timer,
        patch("hysearch.web.runner.uvicorn.run") as mock_uvicorn,
    ):
        mock_timer_inst = MagicMock()
        mock_timer.return_value = mock_timer_inst

        run_server(
            db_path=indexed_db,
            host="127.0.0.1",
            port=None,
            open_browser=True,
        )
        mock_timer_inst.start.assert_called_once()
        mock_uvicorn.assert_called_once()

        # Test KeyboardInterrupt handling with open_browser=False
        mock_uvicorn.side_effect = KeyboardInterrupt()
        run_server(
            db_path=indexed_db,
            host="127.0.0.1",
            port=8888,
            open_browser=False,
        )


def test_cli_server_command(indexed_db: Path, tmp_path: Path) -> None:
    """Verify CLI 'hysearch server' invokes run_server and handles errors."""
    # 1. Missing DB file -> exit code 1
    missing_db = tmp_path / "missing.db"
    res_missing = runner.invoke(cli_app, ["server", "--db", str(missing_db)])
    assert res_missing.exit_code == 1
    assert "does not exist" in " ".join(res_missing.stdout.split())

    # 2. Normal invocation with --no-open
    with patch("hysearch.cli.main.run_server") as mock_run:
        res_ok = runner.invoke(
            cli_app,
            [
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                "8888",
                "--no-open",
                "--db",
                str(indexed_db),
            ],
        )
        assert res_ok.exit_code == 0
        mock_run.assert_called_once_with(
            db_path=indexed_db,
            host="127.0.0.1",
            port=8888,
            open_browser=False,
        )

    # 3. Port conflict or invalid port raises RuntimeError/ValueError -> exit 1
    with patch(
        "hysearch.cli.main.run_server",
        side_effect=RuntimeError("Port 8888 on host '127.0.0.1' is in use."),
    ):
        res_err = runner.invoke(
            cli_app,
            ["server", "-p", "8888", "--db", str(indexed_db)],
        )
        assert res_err.exit_code == 1
        assert "in use" in " ".join(res_err.stdout.split())


def test_web_read_only_db(
    indexed_db: Path, dummy_embedder: DummyEmbedder
) -> None:
    """Verify /api/status and /api/query succeed on chmod 444 database."""
    indexed_db.chmod(0o444)
    try:
        web_app = create_app(db_path=indexed_db, embedder=dummy_embedder)
        client = TestClient(web_app)

        res_status = client.get("/api/status")
        assert res_status.status_code == 200
        assert res_status.json()["Files Indexed"] == 2

        res_query = client.get("/api/query", params={"q": "Python"})
        assert res_query.status_code == 200
        assert res_query.json()["total_hits"] >= 1
    finally:
        indexed_db.chmod(0o666)
