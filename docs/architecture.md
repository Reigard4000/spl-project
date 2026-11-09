# Архитектура SPL

SPL (Security & Protection Layer) — промежуточный слой между пользователем и
zero-shot моделью `deepvk/GeRaCl-USER2-base`. Каждый текстовый запрос проходит
две независимые проверки, после чего либо блокируется, либо передаётся модели.
Все решения сохраняются в SQLite и отображаются в Streamlit.

## Слои

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

Зависимости направлены внутрь: presentation знает о services и domain,
services — только о domain-интерфейсах, infrastructure реализует эти интерфейсы.
Сборка объектов выполняется в `src/presentation/api/deps.py` через FastAPI
`Depends` и `lru_cache`.

## Поток обработки запроса

1. `POST /check` принимает `CheckRequest` (1–10000 символов).
2. FastAPI через `Depends` передаёт запрос в `CheckService`.
3. `AdversarialDetector.check()` считает вероятность LightGBM.
4. Если текст заблокирован, `TreeExplainer` из SHAP строит топ-3 признака,
   повлиявших на решение.
5. Если текст не adversarial, `AnomalyDetector.check()` проверяет 12 признаков
   через IsolationForest.
6. При срабатывании любого детектора запрос блокируется, модель не вызывается.
7. Иначе `GeraclModel.predict` обрезает текст до 512 символов, запускает
   zero-shot pipeline и возвращает категорию и scores.
8. `CheckService` создаёт `LogEntry` с полем `explanation` и сохраняет её.
9. Ответ `CheckResponse` с полем `explanation` возвращается клиенту.

## Детекторы

### AdversarialDetector

- Признаки через `FeatureUnion`:
  - `TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=20000)`;
  - 12 текстовых статистик, общих с anomaly-детектором.
- Классификатор: `LGBMClassifier` (400 деревьев, learning_rate=0.05).
- Обучение: чистые тексты (label=0) и испорченные (label=1), по 250 примеров
  на каждый из четырёх типов атак.
- GroupShuffleSplit по `base_id` делит данные на train/val/test 60/20/20.
- Порог вероятности блокировки выбирается только на validation при FPR ≤ 1%;
  в текущем обучении он равен 0.0935. На grouped test получены Precision
  1.0000, Recall 0.9755, F1 0.9876, ROC-AUC 0.9972, FPR 0.
- Артефакт: `models/adversarial_detector.pkl`
  (`{"model": ..., "threshold": 0.0935}`).
- Объяснимость: `TreeExplainer` считает SHAP-вклады для LightGBM, после
  чего возвращаются три положительных признака с максимальным вкладом.
  Технические char-ngram'ы переводятся в понятные подписи, а статистические
  признаки имеют фиксированные русские названия.

### AnomalyDetector

- Признаки (12): длина, количество слов, средняя длина слова, доля
  не-алфавитных символов, доля заглавных букв, число уникальных символов,
  энтропия Шеннона, количество знаков препинания, доли кириллицы и латиницы,
  число невидимых символов, доля слов со смешанными алфавитами.
- Модель: `IsolationForest(contamination=..., n_estimators=150)`.
- `contamination` выбирается KFold-кросс-валидацией на чистых текстах из
  кандидатов 0.005, 0.01, 0.02, 0.03, 0.05 при целевом FPR ≤ 1%; в текущем
  запуске выбран 0.005 и получен FPR 0.8% на тесте.
- Обучение только на чистых текстах.
- Артефакт: `models/anomaly_detector.pkl`
  (`{"model": ..., "contamination": ..., "feature_names": [...]}`).
- Профильная оценка: обнаружено 96 из 200 аномалий (48.0%) при FPR 0.8%.
  Детектор не заменяет adversarial-модель и в основном отвечает за код, спам,
  английские тексты, эмодзи и повторяющиеся слова.

### Данные и оценка

`data/generate_data.py` создаёт 1000 базовых чистых текстов, 250 hard
negatives (латиница, URL, email, цитаты), 1000 искажённых текстов
(гомоглифы, смена раскладки, невидимые символы, опечатки) и 200 аномалий.
CSV содержит `base_id` и `category`, а split выполняется по группам
`GroupShuffleSplit` 60/20/20. `scripts/train_detectors.py` сохраняет Precision,
Recall, F1, ROC-AUC, матрицы ошибок, FPR и разбивку по типам/категориям в
`models/metrics.json`. `scripts/evaluate.py` независимо воспроизводит grouped
test split, печатает матрицы ошибок и метрики по типам атак и категориям, а
также сохраняет PR-кривые в `models/pr_curve.png`.

Обученные `models/adversarial_detector.pkl`, `models/anomaly_detector.pkl` и
`models/metrics.json` входят в репозиторий, поэтому при обычном запуске
обучение не выполняется. `run.sh` и `run.ps1` переобучают детекторы, если pkl
отсутствуют или не загружаются в текущем окружении.

## Защищаемая модель

`GeraclModel` лениво создаёт

```python
pipeline("zero-shot-classification", model="deepvk/GeRaCl-USER2-base")
```

Категории: `документы`, `сроки`, `общежитие`, `экзамены`, `оплата`.
Модель не обучается и не изменяется. Первый вызов может быть долгим из-за
скачивания весов с Hugging Face. Ошибки сети и загрузки преобразуются в
`ModelError`.

Архитектура GeRaCl (`model_type: GeRaCl`) не поддерживается стандартным
transformers pipeline. Поэтому после ошибки загрузки обёртка пытается
использовать официальный пакет `geracl` (Apache-2.0) с классом
`ZeroShotClassificationPipeline`. Пакет опционален: если его нет,
возвращается `ModelError` с командой установки. Ответ в обоих случаях
приводится к одному формату `{"category": ..., "scores": ...}`; для geracl
сырые similarity-логиты преобразуются через softmax.

## Хранилище

SQLite-таблица `logs`:

| Колонка | Тип | Назначение |
|---|---|---|
| id | INTEGER PK | Идентификатор записи |
| timestamp | TEXT | Время решения в ISO 8601 |
| text | TEXT | Исходный текст запроса |
| status | TEXT | `allowed` или `blocked` |
| reason | TEXT | `adversarial`, `anomaly` или NULL |
| prediction | TEXT | JSON предсказания модели или NULL |
| explanation | TEXT | Текст SHAP-объяснения или NULL |

`init_db()` идемпотентно создаёт таблицу и индекс через `IF NOT EXISTS`.
Существующий журнал при старте API и при обновлении панели не удаляется.
Для изменения схемы в будущем потребуются миграции.

Соединение открывается на каждую операцию и закрывается после неё. Это
упрощает потокобезопасность и не требует общего глобального соединения.
Для in-memory БД в тестах используется shared-cache URI и удерживающее
соединение (`keeper`), иначе база исчезала бы между запросами.

## Ошибки

- `SPLException` — базовое исключение.
- `DetectorError` — нет pkl-файла, ошибка загрузки или предсказания детектора.
- `ModelError` — ошибка загрузки или работы transformers-модели.
- `RepositoryError` — ошибка SQLite.

FastAPI-обработчики возвращают JSON:

```json
{"error": "DetectorError", "detail": "..."}
```

`SPLException` → HTTP 500, `ValidationError` / `RequestValidationError` → HTTP 422.

## Ленивые синглтоны

`get_settings`, `get_log_repo`, `get_adversarial_detector`, `get_anomaly_detector`
и `get_victim_model` обёрнуты в `lru_cache`. Тяжёлые артефакты внутри
детекторов и модели загружаются при первом вызове, а не при старте API.
Это позволяет запускать `/health` и `/logs` без скачивания transformers-модели.

## Панель Streamlit

Панель не содержит графиков. Сверху показываются три метрики, в сайдбаре —
фильтры по статусу, причине и поиску. Журнал выводится интерактивной таблицей
с настроенной шириной колонок; ID находится в конце, технический столбец
`explanation` скрыт. При выборе строки (`on_select="rerun"`,
`selection_mode="single-row"`) под таблицей появляется блок с полным текстом,
статусом, причиной, категорией и SHAP-объяснением. Без выбора выводится
подсказка «Выберите строку в таблице, чтобы увидеть детали».

## Веб-клиент

В папке `client/` находится отдельный статический HTML/JS-клиент. Он
отправляет текст в `POST /check`, показывает решение и загружает журнал через
`GET /logs`. Клиент запускается стандартным `http.server` из
`client/server.py` на `http://127.0.0.1:3000`; этот origin разрешён в CORS API.
Сторонние JS-библиотеки и зависимости не используются.
