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


def test_cli_reset_nonexistent_db(tmp_path: Path) -> None:
    """Verify reset command on nonexistent database reports warning."""
    missing_db = tmp_path / "missing.db"
    res = runner.invoke(app, ["reset", "--db", str(missing_db)])
    assert res.exit_code == 0
    assert "does not exist" in res.stdout


def test_cli_reset_prompt_abort(tmp_path: Path) -> None:
    """Verify aborting reset confirmation preserves database."""
    db_file = tmp_path / "keep.db"
    db_file.write_text("data", encoding="utf-8")

    res = runner.invoke(app, ["reset", "--db", str(db_file)], input="n\n")
    assert res.exit_code != 0
    assert "Aborted reset" in res.stdout
    assert db_file.exists()


def test_cli_reset_prompt_confirm_with_wal(tmp_path: Path) -> None:
    """Verify confirming reset deletes database and extra journal files."""
    db_file = tmp_path / "clean.db"
    db_file.write_text("db_content", encoding="utf-8")

    wal_file = tmp_path / "clean.db-wal"
    wal_file.write_text("wal_content", encoding="utf-8")
    shm_file = tmp_path / "clean.db-shm"
    shm_file.write_text("shm_content", encoding="utf-8")

    res = runner.invoke(app, ["reset", "--db", str(db_file)], input="y\n")
    assert res.exit_code == 0
    assert "Successfully deleted database" in res.stdout
    assert not db_file.exists()
    assert not wal_file.exists()
    assert not shm_file.exists()


def test_cli_query_with_filters(tmp_path: Path) -> None:
    """Verify query command with path, glob, since, after, until, before."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file_md = docs_dir / "guide.md"
    file_md.write_text("Markdown user guide.", encoding="utf-8")
    file_txt = docs_dir / "log.txt"
    file_txt.write_text("Text log information.", encoding="utf-8")

    db_file = tmp_path / "filter_cli.db"
    runner.invoke(app, ["index", str(docs_dir), "--db", str(db_file)])

    # --path / -p
    res_path = runner.invoke(
        app,
        ["query", "guide", "-p", "guide.md", "--db", str(db_file)],
    )
    assert res_path.exit_code == 0
    assert "1 hits" in res_path.stdout

    # --glob / -g
    res_glob = runner.invoke(
        app,
        ["query", "user", "-g", "*.md", "--db", str(db_file)],
    )
    assert res_glob.exit_code == 0
    assert "1 hits" in res_glob.stdout

    # --since / --after (両ファイルとも対象)
    res_since = runner.invoke(
        app,
        ["query", "guide", "--since", "2000-01-01", "--db", str(db_file)],
    )
    assert res_since.exit_code == 0
    assert "2 hits" in res_since.stdout

    res_after = runner.invoke(
        app,
        ["query", "guide", "--after", "2000/01/01", "--db", str(db_file)],
    )
    assert res_after.exit_code == 0
    assert "2 hits" in res_after.stdout

    # --until / --before (両ファイルとも対象)
    res_until = runner.invoke(
        app,
        ["query", "guide", "--until", "2099-12-31", "--db", str(db_file)],
    )
    assert res_until.exit_code == 0
    assert "2 hits" in res_until.stdout

    res_before = runner.invoke(
        app,
        ["query", "guide", "--before", "2099/12", "--db", str(db_file)],
    )
    assert res_before.exit_code == 0
    assert "2 hits" in res_before.stdout

    # 日時範囲外フィルタで 0 件になることを確認
    res_zero = runner.invoke(
        app,
        ["query", "guide", "--since", "2099-01-01", "--db", str(db_file)],
    )
    assert res_zero.exit_code == 0
    assert "No results found" in res_zero.stdout

    # --json でファイルパスが完全にマッチすることを確認
    res_json_filter = runner.invoke(
        app,
        ["query", "guide", "-g", "*.md", "--json", "--db", str(db_file)],
    )
    assert res_json_filter.exit_code == 0
    data = json.loads(res_json_filter.stdout)
    assert data["total_hits"] == 1
    assert "guide.md" in data["hits"][0]["file_path"]


def test_cli_query_filter_conflicts_and_errors(tmp_path: Path) -> None:
    """Verify conflicting and invalid date filters produce exit code 1."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc.txt").write_text("Sample", encoding="utf-8")
    db_file = tmp_path / "error_cli.db"
    runner.invoke(app, ["index", str(docs_dir), "--db", str(db_file)])

    # 競合: since と after
    res_since_conflict = runner.invoke(
        app,
        [
            "query",
            "Sample",
            "--since",
            "2026-01-01",
            "--after",
            "2026-02-01",
            "--db",
            str(db_file),
        ],
    )
    assert res_since_conflict.exit_code == 1
    assert "Conflicting values" in res_since_conflict.stdout

    # 競合: until と before
    res_until_conflict = runner.invoke(
        app,
        [
            "query",
            "Sample",
            "--until",
            "2026-01-01",
            "--before",
            "2026-02-01",
            "--db",
            str(db_file),
        ],
    )
    assert res_until_conflict.exit_code == 1
    assert "Conflicting values" in res_until_conflict.stdout

    # 不正な日付形式
    res_invalid_date = runner.invoke(
        app,
        [
            "query",
            "Sample",
            "--since",
            "invalid-date-string",
            "--db",
            str(db_file),
        ],
    )
    assert res_invalid_date.exit_code == 1
    assert "Invalid date format" in res_invalid_date.stdout
