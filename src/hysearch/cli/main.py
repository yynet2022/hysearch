"""CLI entry points for hysearch commands using Typer.

This module provides the command-line interface for indexing documents,
running hybrid queries, viewing index status, and resetting the database.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from hysearch.cli.display import (
    console,
    print_indexing_stats,
    print_search_results,
    print_status,
)
from hysearch.core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_RRF_K,
)
from hysearch.core.database import get_connection, init_db
from hysearch.core.indexer import index_directory
from hysearch.core.searcher import SearchMode, search

app = typer.Typer(
    name="hysearch",
    help="Fast, zero-daemon local hybrid search engine powered by SQLite.",
    no_args_is_help=True,
)


@app.command(name="index")
def index_cmd(
    dir_path: Annotated[
        Path,
        typer.Argument(
            help="Directory to crawl and index",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = Path("."),
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Target character count per chunk"),
    ] = DEFAULT_CHUNK_SIZE,
    overlap: Annotated[
        int,
        typer.Option("--overlap", help="Character overlap between chunks"),
    ] = DEFAULT_CHUNK_OVERLAP,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force re-indexing all documents"),
    ] = False,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to the SQLite database file"),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Index or update documents in the specified directory."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Scanning files...", total=None)

        def on_progress(current: int, total: int, file_path: Path) -> None:
            progress.update(
                task_id,
                total=total,
                completed=current,
                description=f"[{current}/{total}] {file_path.name}",
            )

        with get_connection(db) as conn:
            init_db(conn)
            stats = index_directory(
                conn=conn,
                target_dir=dir_path,
                chunk_size=chunk_size,
                overlap=overlap,
                force=force,
                on_progress=on_progress,
            )
    print_indexing_stats(stats)


@app.command(name="query")
def query_cmd(
    query_str: Annotated[
        str,
        typer.Argument(help="Search query text or keywords"),
    ],
    top: Annotated[
        int,
        typer.Option(
            "--top",
            "-k",
            help="Number of top search hits to return",
        ),
    ] = 5,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Search mode: hybrid, bm25, or vector",
        ),
    ] = "hybrid",
    rrf_k: Annotated[
        int,
        typer.Option("--rrf-k", help="RRF smoothing constant"),
    ] = DEFAULT_RRF_K,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output results in JSON format"),
    ] = False,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to SQLite database file"),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Search the indexed database using hybrid, BM25, or vector mode."""
    if mode not in ("hybrid", "bm25", "vector"):
        console.print(
            f"[red]Error:[/] Invalid mode '{mode}'. "
            "Choose from hybrid, bm25, or vector."
        )
        raise typer.Exit(code=1)

    if not db.exists():
        console.print(
            f"[red]Error:[/] Database file '{db}' does not exist. "
            "Run 'hysearch index' first."
        )
        raise typer.Exit(code=1)

    search_mode: SearchMode = mode  # type: ignore[assignment]
    with get_connection(db) as conn:
        init_db(conn)
        response = search(
            conn=conn,
            query=query_str,
            mode=search_mode,
            top_k=top,
            rrf_k=rrf_k,
        )

    if output_json:
        typer.echo(response.model_dump_json(indent=2))
    else:
        print_search_results(response)


@app.command(name="status")
def status_cmd(
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output status as JSON"),
    ] = False,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to SQLite database file"),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Display index statistics and database file details."""
    if not db.exists():
        info = {
            "database": str(db),
            "status": "not_found",
            "file_count": 0,
            "chunk_count": 0,
        }
        if output_json:
            typer.echo(json.dumps(info, indent=2))
        else:
            console.print(f"[yellow]Database '{db}' does not exist.[/]")
        return

    stat = db.stat()
    size_kb = stat.st_size / 1024.0
    mod_time = datetime.fromtimestamp(stat.st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with get_connection(db) as conn:
        init_db(conn)
        file_count = conn.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[
            0
        ]

    status_data = {
        "Database Path": str(db.resolve()),
        "Files Indexed": file_count,
        "Total Chunks": chunk_count,
        "Database Size": f"{size_kb:.2f} KB",
        "Last Modified": mod_time,
    }

    if output_json:
        typer.echo(json.dumps(status_data, indent=2))
    else:
        print_status(status_data)


@app.command(name="reset")
def reset_cmd(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Confirm reset without prompt"),
    ] = False,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to SQLite database file"),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Delete the database and clear all stored indexes."""
    if not db.exists():
        console.print(f"[yellow]Database '{db}' does not exist.[/]")
        return

    if not yes:
        confirmed = typer.confirm(
            f"Are you sure you want to delete index database '{db}'?"
        )
        if not confirmed:
            console.print("[dim]Aborted reset.[/]")
            raise typer.Abort()

    db.unlink(missing_ok=True)
    # WAL / SHM ファイル等もクリーンアップ
    for suffix in ("-wal", "-shm", "-journal"):
        extra = db.with_name(db.name + suffix)
        if extra.exists():
            extra.unlink(missing_ok=True)

    console.print(f"[green]Successfully deleted database:[/] {db}")
