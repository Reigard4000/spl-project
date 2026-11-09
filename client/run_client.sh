#!/usr/bin/env bash
# Запуск веб-клиента SPL на Linux/macOS.

set -euo pipefail

CLIENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$CLIENT_DIR")"

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
    PYTHON="$ROOT_DIR/venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "Python 3 не найден. Установи Python 3.10+." >&2
    exit 1
fi

exec "$PYTHON" "$CLIENT_DIR/server.py" "$@"
