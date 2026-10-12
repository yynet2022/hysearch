"""Tests for CLI commands using typer.testing.CliRunner."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hysearch.cli.main import app
from hysearch.core.embedder import DummyEmbedder

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_fastembedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock FastEmbedder across core modules for lightning-fast CLI tests."""
    monkeypatch.setattr("hysearch.core.indexer.FastEmbedder", DummyEmbedder)
    monkeypatch.setattr("hysearch.core.searcher.FastEmbedder", DummyEmbedder)


def test_cli_index_and_query(tmp_path: Path) -> None:
    """Verify index and query commands with standard and JSON output."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text(
        "# Welcome\nThis is a sample document for CLI testing.",
        encoding="utf-8",
    )
    db_file = tmp_path / "test_cli.db"

    # 1. index コマンド
    res_index = runner.invoke(
        app,
        ["index", str(docs_dir), "--db", str(db_file)],
    )
    assert res_index.exit_code == 0
    assert "Indexing Complete" in res_index.stdout

    # 2. query コマンド（通常表示）
    res_query = runner.invoke(
        app,
        ["query", "sample", "--db", str(db_file)],
    )
    assert res_query.exit_code == 0
    assert "sample" in res_query.stdout

    # 3. query コマンド（JSON 出力）
    res_json = runner.invoke(
        app,
        ["query", "sample", "--json", "--db", str(db_file)],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["query"] == "sample"
    assert data["total_hits"] >= 1
    assert len(data["hits"]) >= 1


def test_cli_status(tmp_path: Path) -> None:
    """Verify status command reports correct properties and JSON output."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "status_test.txt").write_text("Hello status", encoding="utf-8")
    db_file = tmp_path / "status.db"

    runner.invoke(app, ["index", str(docs_dir), "--db", str(db_file)])

    res_status = runner.invoke(app, ["status", "--db", str(db_file)])
    assert res_status.exit_code == 0
    assert "Files Indexed" in res_status.stdout

    res_json = runner.invoke(app, ["status", "--json", "--db", str(db_file)])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["Files Indexed"] == 1


def test_cli_reset(tmp_path: Path) -> None:
    """Verify reset command deletes database file."""
    db_file = tmp_path / "to_delete.db"
    db_file.write_text("dummy", encoding="utf-8")

    res_reset = runner.invoke(app, ["reset", "--yes", "--db", str(db_file)])
    assert res_reset.exit_code == 0
    assert not db_file.exists()


def test_cli_missing_db_error(tmp_path: Path) -> None:
    """Verify querying nonexistent DB returns exit code 1."""
    missing_db = tmp_path / "nonexistent.db"
    res = runner.invoke(app, ["query", "hello", "--db", str(missing_db)])
    assert res.exit_code == 1


def test_cli_invalid_mode(tmp_path: Path) -> None:
    """Verify invalid search mode returns exit code 1."""
    db_file = tmp_path / "dummy.db"
    db_file.write_text("data", encoding="utf-8")
    res = runner.invoke(
        app,
        ["query", "hello", "--mode", "invalid_mode", "--db", str(db_file)],
    )
    assert res.exit_code == 1


def test_cli_status_nonexistent_db(tmp_path: Path) -> None:
    """Verify status command on missing database returns gracefully."""
    missing_db = tmp_path / "missing.db"
    res = runner.invoke(app, ["status", "--db", str(missing_db)])
    assert res.exit_code == 0
    assert "does not exist" in res.stdout

    res_json = runner.invoke(
        app, ["status", "--json", "--db", str(missing_db)]
    )
    assert res_json.exit_code == 0
    assert "not_found" in res_json.stdout


def test_cli_query_zero_hits(tmp_path: Path) -> None:
    """Verify query command handles zero results gracefully."""
    docs_dir = tmp_path / "empty_docs"
    docs_dir.mkdir()
    db_file = tmp_path / "empty.db"
    runner.invoke(app, ["index", str(docs_dir), "--db", str(db_file)])

    res = runner.invoke(
        app, ["query", "nonexistentquery", "--db", str(db_file)]
    )
    assert res.exit_code == 0
    assert "No results found" in res.stdout
