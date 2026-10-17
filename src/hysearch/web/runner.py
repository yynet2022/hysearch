"""Server port resolution, browser auto-launch, and Uvicorn runner.

This module manages port availability checks, automatic port incrementing
for the default port, browser opening, and Uvicorn server execution.
"""

import socket
import threading
import webbrowser
from pathlib import Path

import uvicorn

from hysearch.cli.display import console
from hysearch.core.config import (
    DEFAULT_DB_PATH,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    MAX_PORT_SEARCH_TRIES,
)
from hysearch.core.database import PathLike
from hysearch.web.app import create_app


def is_port_available(host: str, port: int) -> bool:
    """Check whether a TCP port is available for binding on the given host.

    Args:
        host: Hostname or IP address to check.
        port: TCP port number (1-65535).

    Returns:
        bool: True if the port can be bound, False if already in use.
    """
    if port < 1 or port > 65535:
        return False

    check_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        # connect_ex と bind の両方を確認して空き状態を判定する
        if sock.connect_ex((check_host, port)) == 0:
            return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bind_sock:
        try:
            bind_sock.bind((host, port))
            return True
        except OSError:
            return False


def resolve_server_port(
    host: str = DEFAULT_SERVER_HOST,
    port: int | None = None,
    max_tries: int = MAX_PORT_SEARCH_TRIES,
) -> int:
    """Resolve the port to bind the web server to.

    If `port` is explicitly provided, verifies that it is available and raises
    a RuntimeError if it is in use. If `port` is None, starts at
    DEFAULT_SERVER_PORT (8888) and increments until a free port is found.

    Args:
        host: Hostname or IP address to bind.
        port: Optional explicit port number specified by the user.
        max_tries: Maximum number of ports to probe when auto-searching.

    Returns:
        int: Resolved available port number.

    Raises:
        ValueError: If the requested port number is out of valid range.
        RuntimeError: If the explicit port is in use or no free port is found.
    """
    if port is not None:
        if port < 1 or port > 65535:
            raise ValueError(
                f"Invalid port number {port}. Must be between 1 and 65535."
            )
        if not is_port_available(host, port):
            raise RuntimeError(
                f"Port {port} on host '{host}' is already in use."
            )
        return port

    start_port = DEFAULT_SERVER_PORT
    for candidate in range(start_port, min(65536, start_port + max_tries)):
        if is_port_available(host, candidate):
            return candidate

    raise RuntimeError(
        f"Could not find an available port in range "
        f"{start_port}-{start_port + max_tries - 1} on host '{host}'."
    )


def build_browser_url(host: str, port: int) -> str:
    """Construct the URL to open in the user's browser.

    Args:
        host: Bound server host.
        port: Bound server port.

    Returns:
        str: HTTP URL pointing to the local server.
    """
    browser_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    return f"http://{browser_host}:{port}"


def run_server(
    db_path: PathLike = DEFAULT_DB_PATH,
    host: str = DEFAULT_SERVER_HOST,
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Start the FastAPI web server using Uvicorn.

    Args:
        db_path: Path to the SQLite database file.
        host: Hostname or IP address to bind.
        port: Optional explicit port number. If None, auto-detects from 8888.
        open_browser: If True, opens the default web browser automatically.
    """
    resolved_port = resolve_server_port(host=host, port=port)
    url = build_browser_url(host, resolved_port)

    console.print(
        f"[bold green]Starting hysearch server at[/] [bold cyan]{url}[/] "
        f"(bound to {host}:{resolved_port}, db: {Path(db_path)})"
    )
    console.print("[dim]Press Ctrl+C to stop the server.[/]")

    if open_browser:
        timer = threading.Timer(0.6, webbrowser.open, args=[url])
        timer.daemon = True
        timer.start()

    app = create_app(db_path=db_path)
    try:
        uvicorn.run(app, host=host, port=resolved_port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        console.print("[dim]Server stopped.[/]")
