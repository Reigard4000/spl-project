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
SKIP_GERACL=0
RETRAIN=0
API_PID=""

usage() {
    cat <<EOF
Использование: ./run.sh [all|api|ui] [--no-install] [--no-setup] [--no-geracl] [--retrain]

  all          запустить API и Streamlit (по умолчанию)
  api          запустить только API
  ui           запустить только Streamlit
  --no-install не устанавливать зависимости
  --no-setup   не генерировать данные и не обучать детекторы
  --no-geracl  не устанавливать официальный backend geracl
  --retrain    переобучить детекторы даже при наличии pkl
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
        --no-geracl)
            SKIP_GERACL=1
            ;;
        --retrain)
            RETRAIN=1
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

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Нужен Python 3.10 или новее." >&2
    exit 1
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    install_marker="$VENV_DIR/.spl_requirements"
    if [[ ! -f "$install_marker" || requirements.txt -nt "$install_marker" ]]; then
        echo "Устанавливаю зависимости..."
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -r requirements.txt
        touch "$install_marker"
    fi
fi

model_is_loadable() {
    "$PYTHON" - <<'PY' >/dev/null 2>&1
import pickle

for path in ("models/adversarial_detector.pkl", "models/anomaly_detector.pkl"):
    with open(path, "rb") as file:
        artifacts = pickle.load(file)
    if "model" not in artifacts:
        raise SystemExit(1)
PY
}

install_geracl() {
    if "$PYTHON" -c "import geracl" >/dev/null 2>&1; then
        return 0
    fi

    if [[ "$SKIP_INSTALL" -eq 1 ]]; then
        echo "Подсказка: geracl не установлен, чистые запросы вернут ModelError."
        echo "Установи backend: make install-geracl"
        return 0
    fi

    if [[ "$SKIP_GERACL" -eq 1 ]]; then
        echo "Backend geracl пропущен (--no-geracl). Чистые запросы вернут ModelError."
        return 0
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "Для установки geracl нужен git. Чистые запросы вернут ModelError." >&2
        return 0
    fi

    echo "Устанавливаю geracl (это может занять время)..."
    if "$PYTHON" -m pip install -r requirements-geracl.txt; then
        echo "geracl установлен."
    else
        echo "Не удалось установить geracl. Чистые запросы вернут ModelError." >&2
    fi
}

install_geracl

MODELS_READY=0
if [[ -f models/adversarial_detector.pkl && -f models/anomaly_detector.pkl ]]; then
    if model_is_loadable; then
        MODELS_READY=1
    fi
fi

if [[ "$RETRAIN" -eq 1 ]]; then
    MODELS_READY=0
fi

if [[ "$SKIP_SETUP" -eq 1 ]]; then
    if [[ "$MODELS_READY" -eq 1 ]]; then
        echo "Использую существующие обученные детекторы."
    else
        echo "Внимание: --no-setup, но детекторы отсутствуют или не загружаются." >&2
    fi
elif [[ "$MODELS_READY" -eq 1 ]]; then
    echo "Обученные детекторы найдены, обучение пропущено."
else
    if [[ "$RETRAIN" -eq 1 ]]; then
        echo "Принудительное переобучение детекторов (--retrain)..."
    elif [[ -f models/adversarial_detector.pkl || -f models/anomaly_detector.pkl ]]; then
        echo "Сохранённые детекторы не загрузились, переобучаю..."
    fi
    if [[ ! -f data/detector_train.csv || ! -f data/anomaly_texts.txt ]]; then
        echo "Генерирую данные..."
        "$PYTHON" data/generate_data.py
    fi
    echo "Обучаю детекторы..."
    "$PYTHON" scripts/train_detectors.py
    if ! model_is_loadable; then
        echo "Ошибка: обученные детекторы не загружаются." >&2
        exit 1
    fi
fi

echo "Модель GeRaCl загрузится при первом чистом запросе (кэш около 600 МБ)."

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
