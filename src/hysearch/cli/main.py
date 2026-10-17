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
    print_prune_summary,
    print_pruned_file,
    print_search_results,
    print_status,
)
from hysearch.core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_RRF_K,
)
from hysearch.core.database import (
    get_connection,
    get_db_tokenizer,
    get_stored_tokenizer,
    init_db,
)
from hysearch.core.indexer import index_directory, prune_missing_files
from hysearch.core.searcher import SearchMode, search
from hysearch.core.tokenizer import resolve_tokenizer

app = typer.Typer(
    name="hysearch",
    help="Fast, zero-daemon local hybrid search engine powered by SQLite.",
    no_args_is_help=True,
)


def parse_extension_filters(
    ext_inputs: list[str] | None,
) -> list[str] | None:
    """Normalize file extensions into lowercase dotted format.

    Args:
        ext_inputs: Raw list of extension options provided via CLI.

    Returns:
        list[str] | None: Normalized list of extensions (e.g. ['.py']),
            or None if no filter provided.
    """
    if not ext_inputs:
        return None

    normalized: list[str] = []
    for item in ext_inputs:
        for part in item.split(","):
            clean = part.strip()
            if not clean:
                continue
            if clean.startswith("*"):
                clean = clean.lstrip("*")
            if not clean.startswith("."):
                clean = f".{clean}"
            clean = clean.lower()
            if clean not in normalized:
                normalized.append(clean)

    return normalized if normalized else None


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
    ext: Annotated[
        list[str] | None,
        typer.Option(
            "--ext",
            "-e",
            help="File extension(s) to index (e.g. .py; overrides defaults)",
        ),
    ] = None,
    tokenizer: Annotated[
        str | None,
        typer.Option(
            "--tokenizer",
            "-t",
            help=(
                "FTS tokenizer: sudachi-full, sudachi-core, "
                "sudachi-small, sudachi, or trigram"
            ),
        ),
    ] = None,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to the SQLite database file"),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Index or update documents in the specified directory."""
    if tokenizer is not None:
        try:
            resolved_req = resolve_tokenizer(tokenizer)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        resolved_req = None

    parsed_exts = parse_extension_filters(ext)
    with get_connection(db) as conn:
        stored_tok = get_stored_tokenizer(conn)
        try:
            target_tok = (
                resolved_req
                if resolved_req is not None
                else resolve_tokenizer(stored_tok)
            )
            init_db(
                conn,
                tokenizer=target_tok,
                force_reinit_fts=force,
            )
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(code=1) from exc

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

            stats = index_directory(
                conn=conn,
                target_dir=dir_path,
                chunk_size=chunk_size,
                overlap=overlap,
                force=force,
                extensions=parsed_exts,
                on_progress=on_progress,
            )
    print_indexing_stats(stats)


def parse_datetime_filter(val: str, param_name: str) -> datetime:
    """Parse a date or datetime string into a datetime object.

    Args:
        val: Input date string.
        param_name: Parameter name for error reporting.

    Returns:
        datetime: Parsed datetime instance.
    """
    clean_val = val.strip()
    try:
        return datetime.fromisoformat(clean_val)
    except ValueError:
        pass

    for fmt in (
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m",
        "%Y-%m",
        "%Y",
    ):
        try:
            return datetime.strptime(clean_val, fmt)
        except ValueError:
            pass

    console.print(
        f"[red]Error:[/] Invalid date format for {param_name}: '{val}'. "
        "Use YYYY-MM-DD or ISO 8601 format."
    )
    raise typer.Exit(code=1)


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
    path: Annotated[
        str | None,
        typer.Option(
            "--path",
            "-p",
            help="Filter by directory path or prefix",
        ),
    ] = None,
    glob: Annotated[
        str | None,
        typer.Option(
            "--glob",
            "-g",
            help="Filter by file glob pattern",
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Filter files modified since date/time",
        ),
    ] = None,
    after: Annotated[
        str | None,
        typer.Option(
            "--after",
            help="Alias for --since",
        ),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(
            "--until",
            help="Filter files modified until date/time",
        ),
    ] = None,
    before: Annotated[
        str | None,
        typer.Option(
            "--before",
            help="Alias for --until",
        ),
    ] = None,
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

    if since and after and since != after:
        console.print(
            "[red]Error:[/] Conflicting values provided for "
            "--since and --after."
        )
        raise typer.Exit(code=1)

    if until and before and until != before:
        console.print(
            "[red]Error:[/] Conflicting values provided for "
            "--until and --before."
        )
        raise typer.Exit(code=1)

    effective_since = since or after
    effective_until = until or before

    since_dt = (
        parse_datetime_filter(effective_since, "--since")
        if effective_since
        else None
    )
    until_dt = (
        parse_datetime_filter(effective_until, "--until")
        if effective_until
        else None
    )

    if not db.exists():
        console.print(
            f"[red]Error:[/] Database file '{db}' does not exist. "
            "Run 'hysearch index' first."
        )
        raise typer.Exit(code=1)

    search_mode: SearchMode = mode  # type: ignore[assignment]
    with get_connection(db) as conn:
        init_db(conn)
        try:
            response = search(
                conn=conn,
                query=query_str,
                mode=search_mode,
                top_k=top,
                rrf_k=rrf_k,
                path_filter=path,
                glob_filter=glob,
                since=since_dt,
                until=until_dt,
            )
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(code=1) from exc

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
        active_tokenizer = get_db_tokenizer(conn)
        file_count = conn.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[
            0
        ]

    status_data = {
        "Database Path": str(db.resolve()),
        "Tokenizer": active_tokenizer,
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


@app.command(name="prune")
def prune_cmd(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Show files to remove without deleting them",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output results in JSON format"),
    ] = False,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to SQLite database file"),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Remove database entries for files that no longer exist on disk."""
    if not db.exists():
        console.print(
            f"[red]Error:[/] Database file '{db}' does not exist. "
            "Run 'hysearch index' first."
        )
        raise typer.Exit(code=1)

    def on_missing(file_path: str, is_dry_run: bool) -> None:
        if not output_json:
            print_pruned_file(file_path, dry_run=is_dry_run)

    with get_connection(db) as conn:
        init_db(conn)
        total_checked, pruned_files = prune_missing_files(
            conn=conn,
            dry_run=dry_run,
            on_missing=on_missing,
        )

    if output_json:
        result = {
            "dry_run": dry_run,
            "total_checked": total_checked,
            "pruned_count": len(pruned_files),
            "pruned_files": pruned_files,
        }
        typer.echo(json.dumps(result, indent=2))
    else:
        print_prune_summary(
            total_checked=total_checked,
            pruned_count=len(pruned_files),
            dry_run=dry_run,
        )
