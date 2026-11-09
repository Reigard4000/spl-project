#!/usr/bin/env bash
# Запуск SPL: подготовка окружения, данных, детекторов, API и Streamlit.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8501}"

MODE="all"
SKIP_INSTALL=0
SKIP_SETUP=0
API_PID=""

usage() {
    cat <<EOF
Использование: ./run.sh [all|api|ui] [--no-install] [--no-setup]

  all          запустить API и Streamlit (по умолчанию)
  api          запустить только API
  ui           запустить только Streamlit
  --no-install не устанавливать зависимости
  --no-setup   не генерировать данные и не обучать детекторы
  -h, --help   показать эту справку

Переменные окружения: VENV_DIR, PYTHON_BIN, API_HOST, API_PORT, UI_HOST, UI_PORT.
EOF
}

for argument in "$@"; do
    case "$argument" in
        all|api|ui)
            MODE="$argument"
            ;;
        --no-install)
            SKIP_INSTALL=1
            ;;
        --no-setup)
            SKIP_SETUP=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Неизвестный аргумент: $argument" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cleanup() {
    if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
        echo "Останавливаю API..."
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Создаю виртуальное окружение $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    install_marker="$VENV_DIR/.spl_requirements"
    if [[ ! -f "$install_marker" || requirements.txt -nt "$install_marker" ]]; then
        echo "Устанавливаю зависимости..."
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -r requirements.txt
        touch "$install_marker"
    fi
fi

if [[ "$SKIP_SETUP" -eq 0 ]]; then
    if [[ ! -f data/detector_train.csv ]]; then
        echo "Генерирую данные..."
        "$PYTHON" data/generate_data.py
    fi
    if [[ ! -f models/adversarial_detector.pkl || ! -f models/anomaly_detector.pkl ]]; then
        echo "Обучаю детекторы..."
        "$PYTHON" scripts/train_detectors.py
    fi
fi

if ! "$PYTHON" -c "import geracl" >/dev/null 2>&1; then
    echo "Подсказка: для архитектуры GeRaCl может понадобиться официальный backend:"
    echo '  pip install "geracl @ git+https://github.com/deepvk/geracl" \'
    echo "    loguru pytorch-lightning torchmetrics"
fi

wait_for_api() {
    local health_host="$API_HOST"
    if [[ "$health_host" == "0.0.0.0" || "$health_host" == "::" ]]; then
        health_host="127.0.0.1"
    fi

    local attempt
    for attempt in {1..40}; do
        if "$PYTHON" -c \
            "import urllib.request; urllib.request.urlopen('http://${health_host}:${API_PORT}/health', timeout=1)" \
            >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$API_PID" 2>/dev/null; then
            return 1
        fi
        sleep 0.5
    done
    return 1
}

start_api() {
    echo "Запускаю API на http://${API_HOST}:${API_PORT}..."
    "$PYTHON" -m uvicorn src.presentation.api.app:create_app \
        --factory \
        --host "$API_HOST" \
        --port "$API_PORT" &
    API_PID=$!

    if ! wait_for_api; then
        echo "API не запустился. Проверь лог выше." >&2
        exit 1
    fi
    echo "API готов: http://${API_HOST}:${API_PORT}"
    echo "Swagger: http://${API_HOST}:${API_PORT}/docs"
}

start_ui() {
    echo "Запускаю Streamlit на http://${UI_HOST}:${UI_PORT}..."
    PYTHONPATH="$ROOT_DIR" "$PYTHON" -m streamlit run \
        src/presentation/ui/streamlit_app.py \
        --server.address "$UI_HOST" \
        --server.port "$UI_PORT" \
        --server.headless true
}

case "$MODE" in
    all)
        start_api
        start_ui
        ;;
    api)
        start_api
        echo "Нажми Ctrl+C для остановки."
        wait "$API_PID"
        ;;
    ui)
        start_ui
        ;;
esac
