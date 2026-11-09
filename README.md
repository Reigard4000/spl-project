# SPL — Security & Protection Layer

SPL — слой защиты между пользователем и ML-моделью. Он проверяет каждый
текстовый запрос двумя детекторами, блокирует adversarial-атаки и аномалии,
а остальные запросы передаёт защищаемой zero-shot модели. Все решения
журналируются в SQLite и отображаются в панели Streamlit.

Защищаемая модель — `deepvk/GeRaCl-USER2-base`. Она не обучается и не
изменяется. Категории классификации: `документы`, `сроки`, `общежитие`,
`экзамены`, `оплата`.

## Что уже реализовано

- Adversarial-детектор: char TF-IDF (2–5-граммы) + 12 текстовых признаков
  через `FeatureUnion` + LightGBM; SHAP-объяснение топ-3 признаков.
- Anomaly-детектор: IsolationForest на 12 математических признаках текста.
- `CheckService`: блокировка при срабатывании любого детектора, вызов
  защищаемой модели только для чистых запросов.
- FastAPI API с dependency injection, pydantic-моделями и JSON-ошибками.
- SQLite-журнал всех решений, включая причину и объяснение блокировки.
- Streamlit-панель: метрики, фильтры и интерактивная таблица журнала.
- Скрипты генерации данных, обучения, оценки и расчёта метрик.
- Обученные детекторы и `models/metrics.json` входят в репозиторий: при
  обычном запуске обучение не требуется.
- Тесты сервисов и API на заглушках (без скачивания transformers).
- `ruff` и `mypy --strict` без ошибок.

## Архитектура

```
presentation (FastAPI, Streamlit)
        |
        v
services (CheckService, LogService)
        |
        v
domain (pydantic-модели, Protocol-интерфейсы)
        ^
        |
infrastructure (SQLite, sklearn, transformers)
        ^
        |
core (Settings, logging, exceptions)
```

Зависимости направлены внутрь: `presentation` знает о `services` и `domain`,
`services` — только о domain-интерфейсах, `infrastructure` реализует эти
интерфейсы. Сборка объектов выполняется в `src/presentation/api/deps.py`.

Поток обработки `POST /check`:

1. `AdversarialDetector` проверяет текст. При срабатывании запрос блокируется
   с причиной `adversarial` и SHAP-объяснением.
2. Иначе `AnomalyDetector` проверяет текст. При срабатывании запрос
   блокируется с причиной `anomaly`.
3. Если оба детектора пропустили текст, `GeraclModel` обрезает его до 512
   символов и возвращает категорию и scores.
4. Решение сохраняется в SQLite и возвращается клиенту.

Подробнее: [docs/architecture.md](docs/architecture.md).

## Требования

- Python 3.10–3.12 (рекомендуется). Проект рассчитан на Python 3.10+.
- Доступ в интернет и около 600 МБ на диске для кэша весов GeRaCl. Веса
  скачиваются при первом чистом запросе, а не при старте приложения.
- Для установки `geracl` нужен `git`.

## Установка

```bash
python -m venv venv
source venv/bin/activate
make install
cp .env.example .env
```

`make install` устанавливает основные зависимости из `requirements.txt`.

### Backend для GeRaCl

`GeraclModel` сначала пробует стандартный
`transformers.pipeline("zero-shot-classification")`. Архитектура GeRaCl не
поддерживается стандартным pipeline, поэтому при ошибке обёртка переключается
на официальный пакет `geracl`, если он установлен.

`run.sh` и `run.ps1` устанавливают `geracl` автоматически при первом запуске,
если его ещё нет. Отключить установку можно флагами `--no-geracl` (Linux/macOS)
и `-NoGeracl` (Windows). Вручную пакет ставится так:

```bash
make install-geracl
# или
pip install -r requirements-geracl.txt
```

Пакет опционален и распространяется под Apache-2.0. Без него API вернёт
`ModelError` на чистый запрос, но `/health`, `/logs` и блокировка
adversarial/anomaly-запросов продолжат работать.

## Быстрый старт

Один скрипт создаёт venv, устанавливает зависимости и `geracl`, использует
обученные детекторы из репозитория, при необходимости переобучает их и
запускает API вместе с панелью:

```bash
./run.sh
```

Режимы и флаги:

```bash
./run.sh api          # только API
./run.sh ui           # только Streamlit
./run.sh all          # API + Streamlit (по умолчанию)
./run.sh --no-install # не устанавливать зависимости
./run.sh --no-setup   # не генерировать данные и не обучать детекторы
./run.sh --no-geracl  # не устанавливать geracl
./run.sh --retrain    # принудительно переобучить детекторы
```

### Первый запуск

- Создаются `venv` и зависимости; `geracl` ставится автоматически, если его
  ещё нет.
- Детекторы уже лежат в `models/*.pkl`, поэтому обучение пропускается.
- Если сохранённые pkl не загружаются в текущем окружении, детекторы
  автоматически переобучаются.
- Данные генерируются только когда нужно обучение или `make evaluate`.
- Модель GeRaCl скачивается при первом чистом запросе в кэш Hugging Face
  (около 600 МБ). `/health`, `/logs` и блокировка атак работают без неё.

После запуска:

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Панель: http://127.0.0.1:8501

Порты и хосты можно переопределить через `API_HOST`, `API_PORT`, `UI_HOST`,
`UI_PORT`, каталог venv — через `VENV_DIR`.

### Makefile

```bash
make install-geracl # установить официальный backend geracl
make data      # сгенерировать чистые, аномальные и испорченные тексты
make train     # обучить детекторы, сохранить pkl и models/metrics.json
make retrain   # удалить pkl и обучить детекторы заново
make evaluate  # матрица ошибок и Precision/Recall/F1 на отложенной выборке
make metrics   # скорость и FPR на чистых текстах
make run-api   # API на http://127.0.0.1:8000
make run-ui    # панель на http://127.0.0.1:8501
make test      # pytest
make lint      # ruff + mypy
make clean     # удалить кеши, данные и отчёт, не трогая обученные детекторы
make clean-models # удалить pkl и models/metrics.json
```

`make train` сам вызывает `make data`. Скрипты и Makefile рассчитаны на запуск
из корня проекта: относительные пути из настроек трактуются от корня.

### Ручной запуск

```bash
pip install -r requirements.txt
pip install -r requirements-geracl.txt
uvicorn src.presentation.api.app:create_app --factory --host 127.0.0.1 --port 8000
PYTHONPATH=. streamlit run src/presentation/ui/streamlit_app.py
```

Детекторы уже лежат в `models/`, поэтому генерация данных и обучение не нужны.
Для переобучения или оценки:

```bash
python data/generate_data.py
python scripts/train_detectors.py
python scripts/evaluate.py
python scripts/compute_metrics.py
```

Также API можно запустить командой `python -m src`.

## Запуск на Windows 10/11

Нативно `run.sh` и `Makefile` не работают: это Unix-скрипты. Используй один
из вариантов ниже.

### WSL2 (рекомендуется)

Ближе всего к Linux-окружению:

```powershell
wsl --install -d Ubuntu
```

В Ubuntu:

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone <repo-url>
cd spl-project
./run.sh
```

Проект лучше клонировать в домашнюю папку WSL, а не в `/mnt/c`, иначе запуск
будет медленнее.

### Нативный Windows

Нужен Python 3.10–3.12 и PowerShell 5.1+. Запуск:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\run.ps1
```

Если менять Execution Policy не хочется, используй обёртку:

```cmd
.\run.cmd
```

Флаги:

```powershell
.\run.ps1 api
.\run.ps1 ui
.\run.ps1 -NoInstall
.\run.ps1 -NoSetup
.\run.ps1 -NoGeracl
.\run.ps1 -Retrain
.\run.ps1 -PythonBin C:\Python312\python.exe
```

`run.ps1` создаёт `venv`, ставит зависимости и `geracl`, использует готовые
детекторы из `models/`, при необходимости переобучает их и запускает API и
панель. Если PowerShell блокирует скрипт:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Если `geracl` не устанавливается нативно, используй WSL2. Путь к проекту
лучше выбирать без кириллицы, например `C:\projects\spl-project`.

### Ручной запуск на Windows

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-geracl.txt
.\venv\Scripts\python.exe data/generate_data.py
.\venv\Scripts\python.exe scripts/train_detectors.py
```

API в одном терминале:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.presentation.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Панель во втором терминале:

```powershell
.\venv\Scripts\python.exe -m streamlit run src/presentation/ui/streamlit_app.py
```

## API

| Метод | Путь | Описание |
|---|---|---|
| POST | `/check` | Проверить текст и получить решение SPL |
| GET | `/logs?limit=50` | Последние записи журнала (1–500) |
| GET | `/health` | Проверка доступности `{"status": "ok"}` |

Пример ответа на разрешённый запрос:

```json
{
  "allowed": true,
  "reason": null,
  "prediction": {
    "category": "документы",
    "scores": {
      "документы": 0.957,
      "сроки": 0.0001,
      "общежитие": 0.0003,
      "экзамены": 0.009,
      "оплата": 0.033
    }
  },
  "explanation": null
}
```

Заблокированный запрос:

```json
{
  "allowed": false,
  "reason": "adversarial",
  "prediction": null,
  "explanation": "Наиболее значимые признаки: Смешение алфавитов, Средняя длина слова, Фрагмент «сп»"
}
```

Для причины `anomaly` поле `explanation` равно `null`.

Ошибки возвращаются в едином формате:

```json
{"error": "DetectorError", "detail": "..."}
```

`SPLException` → HTTP 500, ошибки валидации → HTTP 422.

### Примеры curl

Чистый запрос:

```bash
curl -s -X POST http://127.0.0.1:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text":"Здравствуйте! Какие документы нужны для поступления на бюджет?"}'
```

Adversarial-запрос:

```bash
curl -s -X POST http://127.0.0.1:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text":"К@к1е д0кумеHты Hy)KHы для п0стyп/\еHия?"}'
```

Аномальный запрос:

```bash
curl -s -X POST http://127.0.0.1:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text":"ПОБЕДИТЕЛЬ №5! Вы выиграли 8000 рублей! Срочно перейдите по ссылке http://example.com/prize/5 и укажите данные банковской карты."}'
```

Журнал:

```bash
curl -s "http://127.0.0.1:8000/logs?limit=10"
```

## Данные и модели

`data/generate_data.py` создаёт:

- 1000 базовых чистых обращений в вуз;
- 250 hard negatives: нормальные обращения с Python, SQL, IT, URL, email
  и цитатами, чтобы детектор не считал латиницу атакой;
- 1000 искажённых текстов, по 250 на каждый тип атаки: гомоглифы, смена
  раскладки клавиатуры, вставка zero-width/soft hyphen символов и перестановка
  соседних букв;
- 200 аномалий: код, спам, английские тексты, сообщения с 50+ эмодзи и
  повторяющиеся слова.

CSV содержит `base_id` (группу исходного текста) и `category`. Train/val/test
делятся через `GroupShuffleSplit` 60/20/20 по группам, поэтому чистый текст и
его атака не попадают в разные части выборки.

Anomaly-детектор использует 12 признаков: длина, количество слов, средняя
длина слова, доля не-алфавитных символов, доля заглавных букв, число уникальных
символов, энтропия Шеннона, количество знаков препинания, доли кириллицы и
латиницы, число невидимых символов и доля слов со смешанными алфавитами.

Adversarial-детектор объединяет TF-IDF по char n-граммам 2–5 (до 20 000
признаков) с теми же 12 признаками через `FeatureUnion`, после чего использует
LightGBM. Порог блокировки подбирается только на validation при FPR ≤ 1%.

Обученные детекторы уже входят в репозиторий:

- `models/adversarial_detector.pkl`;
- `models/anomaly_detector.pkl`;
- `models/metrics.json`.

`models/pr_curve.png` создаётся командой `make evaluate` и не нужен для
запуска.

При обычном запуске обучение не выполняется. Переобучить детекторы можно
командами `make train`, `make retrain`, `./run.sh --retrain` или
`.\run.ps1 -Retrain`. Если сохранённые pkl несовместимы с версиями
scikit-learn/LightGBM в окружении, `run.sh` и `run.ps1` автоматически
переобучают детекторы.

## Метрики

Результаты на сгенерированном датасете (grouped test, 20% данных):

| Метрика | Adversarial | Anomaly |
|---|---|---|
| Precision | 1.0000 | 0.9661 |
| Recall (атаки) | 0.9755 | 0.2794 |
| F1-Score | 0.9876 | 0.4335 |
| ROC-AUC | 0.9972 | 0.7765 |
| False Positive Rate на чистых текстах | 0.00% | 0.80% |
| False Positives / False Negatives | 0 / 5 | 2 / 147 |

Adversarial Recall по типам атак: гомоглифы — 1.00, смена раскладки — 1.00,
невидимые символы — 1.00, опечатки — 0.8958. Anomaly-детектор не заменяет
adversarial-детектор: при FPR 0.8% он обнаруживает 96 из 200 реальных аномалий
(48.0%). В SPL детекторы работают вместе.

`make evaluate` дополнительно сохраняет PR-кривые в `models/pr_curve.png`.
Average Precision: adversarial — 0.9975, anomaly — 0.7889.

`make metrics` проверяет целевые показатели:

| Метрика | Значение | Цель |
|---|---|---|
| ROC-AUC adversarial-детектора на grouped test | 0.9972 | ≥ 0.90 |
| Среднее время adversarial + anomaly на текст | около 12 мс после прогрева | ≤ 50 мс |
| Доля ложных срабатываний на чистых тестах | 0.8% | ≤ 1% |

Точные значения зависят от сгенерированных данных, версии scikit-learn и
производительности машины.

## Конфигурация

Настройки читаются из `.env` и переменных окружения (pydantic-settings).
Значения по умолчанию заданы в `src/core/config.py`.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MODEL_NAME` | `deepvk/GeRaCl-USER2-base` | Hugging Face модель |
| `CATEGORIES` | список из 5 категорий | можно передать JSON-массивом |
| `DB_PATH` | `spl_logs.db` | путь к SQLite |
| `ADVERSARIAL_MODEL_PATH` | `models/adversarial_detector.pkl` | артефакт adversarial |
| `ANOMALY_MODEL_PATH` | `models/anomaly_detector.pkl` | артефакт anomaly |
| `TRAIN_CSV` | `data/detector_train.csv` | обучающий CSV |
| `ANOMALY_CONTAMINATION` | `0.01` | базовый кандидат contamination; итог выбирается CV |
| `LOG_LEVEL` | `INFO` | уровень logging |
| `API_HOST` | `127.0.0.1` | хост uvicorn |
| `API_PORT` | `8000` | порт uvicorn |

CORS разрешён только для локальных адресов панели и разработки:
`localhost`/`127.0.0.1` на портах 8501 и 3000.

Схема SQLite создаётся идемпотентно (`CREATE TABLE IF NOT EXISTS`), поэтому
журнал сохраняется при перезапуске API и обновлении панели.

## Тесты и качество кода

```bash
make test
make lint
```

На Windows:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m mypy src scripts data tests
```

- `pytest` — тесты `CheckService` (чистый текст, adversarial, anomaly, ошибка
  детектора, сохранение журнала) и API через `httpx.AsyncClient` (`/check`,
  `/health`, `/logs`, валидация).
- `ruff check .` — линтер, длина строки 100.
- `mypy src scripts data tests` — strict-режим.

Тесты не скачивают модель и не требуют pkl-файлов: детекторы и victim-модель
подменяются заглушками. SQLite в тестах — in-memory с shared cache.

## Структура проекта

```
spl-project/
├── README.md
├── requirements.txt
├── requirements-geracl.txt
├── pyproject.toml
├── .env.example
├── .gitattributes
├── Makefile
├── run.sh
├── run.ps1
├── run.cmd
├── data/
│   ├── generate_data.py
│   ├── clean_texts.txt          # генерируется
│   ├── anomaly_texts.txt        # генерируется
│   └── detector_train.csv       # генерируется
├── models/
│   ├── .gitkeep
│   ├── adversarial_detector.pkl # обученный детектор
│   ├── anomaly_detector.pkl     # обученный детектор
│   ├── metrics.json             # метрики обучения
│   └── pr_curve.png             # создаётся make evaluate
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   ├── domain/
│   │   ├── models.py
│   │   └── interfaces.py
│   ├── infrastructure/
│   │   ├── db/
│   │   │   ├── sqlite_repo.py
│   │   │   └── schema.sql
│   │   ├── detectors/
│   │   │   ├── adversarial.py
│   │   │   └── anomaly.py
│   │   └── victim/
│   │       └── geracl.py
│   ├── services/
│   │   ├── check_service.py
│   │   └── log_service.py
│   └── presentation/
│       ├── api/
│       │   ├── app.py
│       │   ├── deps.py
│       │   └── routes.py
│       └── ui/
│           └── streamlit_app.py
├── scripts/
│   ├── train_detectors.py
│   ├── evaluate.py
│   └── compute_metrics.py
├── docs/
│   ├── architecture.md
│   └── technical_specification.md
└── tests/
    ├── conftest.py
    ├── test_check_service.py
    └── test_api.py
```

## Документация

- [docs/architecture.md](docs/architecture.md) — слои, детекторы, хранилище,
  ошибки и панель.
- [docs/technical_specification.md](docs/technical_specification.md) —
  техническое задание на развитие MVP до прототипа и продукта.

## Ограничения

- Детекторы обучены на синтетических искажениях и не гарантируют защиту от
  реальных состязательных атак; требуется валидация на реальном трафике.
- Anomaly-детектор настроен на строгий FPR и не заменяет adversarial-детектор.
  SHAP-объяснения строятся только для adversarial-блокировок.
- Нет аутентификации, rate limiting и разграничения ролей API.
- SQLite подходит для учебного проекта и небольшой нагрузки; журнал растёт без
  автоматической очистки.
- CORS ограничен localhost: панель и API должны запускаться на одной машине.
- Защищаемая модель используется как есть; её доменная точность не оценивается.
- Тексты и обучение ориентированы на русский язык.
- Нет Docker, CI/CD и фоновой обработки.

## Лицензии

Код проекта распространяется под MIT. Используются компоненты с
разрешительными лицензиями: FastAPI (MIT), Uvicorn (BSD-3-Clause), Streamlit
(Apache-2.0), scikit-learn (BSD-3-Clause), LightGBM (MIT), matplotlib
(PSF/BSD-совместимая), transformers (Apache-2.0), PyTorch (BSD-3-Clause),
pandas (BSD-3-Clause), pydantic (MIT), shap (MIT), pytest (MIT), httpx
(BSD-3-Clause), ruff (MIT), mypy (MIT). Модель `deepvk/GeRaCl-USER2-base` и
опциональный пакет `geracl` распространяются под Apache-2.0.
