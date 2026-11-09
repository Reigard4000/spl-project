"""Мини-сервер для веб-клиента SPL.

Клиент должен открываться с origin http://127.0.0.1:3000 или
http://localhost:3000, потому что эти адреса разрешены в CORS API.
"""

from __future__ import annotations

import argparse
import http.server
import os
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

CLIENT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3000


class ClientRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Отдаёт статические файлы клиента и не показывает список каталога."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def list_directory(self, path: str | os.PathLike[str]) -> None:
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[client] {self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Локальный веб-клиент SPL для проверки POST /check.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Адрес прослушивания (по умолчанию {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Порт (по умолчанию {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать браузер автоматически",
    )
    return parser.parse_args()


def main() -> int:
    """Запустить HTTP-сервер и открыть браузер."""
    args = parse_args()
    host = str(args.host)
    port = int(args.port)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}/"

    os.chdir(CLIENT_DIR)
    try:
        server = http.server.ThreadingHTTPServer(
            (host, port),
            ClientRequestHandler,
        )
    except OSError as exc:
        print(f"Не удалось запустить клиент на {host}:{port}: {exc}", file=sys.stderr)
        print(
            "Проверь, не занят ли порт 3000, или закрой старый экземпляр клиента.",
            file=sys.stderr,
        )
        return 1

    print(f"Веб-клиент SPL: {url}")
    print("API должен быть запущен отдельно, например: ./run.sh api")
    print("Остановить клиент: Ctrl+C")

    if not args.no_browser:
        threading.Timer(0.7, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nОстанавливаю веб-клиент...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
