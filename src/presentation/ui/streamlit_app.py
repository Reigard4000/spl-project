"""Веб-панель SPL на Streamlit."""

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from src.presentation.api.deps import get_log_repo  # noqa: E402
from src.services.log_service import LogService  # noqa: E402

logger = logging.getLogger(__name__)

STATUS_LABELS = {
    "allowed": "Пропущено",
    "blocked": "Заблокировано",
}
STATUS_FILTER_LABELS = {
    "all": "Все запросы",
    "allowed": "Только пропущенные",
    "blocked": "Только заблокированные",
}
REASON_LABELS = {
    "adversarial": "Adversarial-атака",
    "anomaly": "Аномалия",
    "__none__": "Без причины (запрос пропущен)",
}
REASON_ORDER = ("adversarial", "anomaly", "__none__")

STATUS_FILTER_KEY = "spl_status_filter"
REASON_FILTER_KEY = "spl_reason_filter"
SEARCH_KEY = "spl_search"

LOG_COLUMNS = [
    "id",
    "timestamp",
    "text",
    "status",
    "status_label",
    "reason_code",
    "reason_label",
    "category",
    "confidence",
    "explanation",
]

setup_logging(get_settings().log_level)


def parse_prediction(value: object) -> tuple[str, float | None]:
    """Извлечь категорию и уверенность из JSON-предсказания модели."""
    if not isinstance(value, dict):
        return "—", None

    category = value.get("category")
    category_text = str(category) if category else "—"

    scores = value.get("scores")
    if not isinstance(scores, dict):
        return category_text, None

    numeric_scores = [
        float(score)
        for score in scores.values()
        if isinstance(score, (int, float)) and not isinstance(score, bool)
    ]
    confidence = max(numeric_scores) if numeric_scores else None
    return category_text, confidence


def format_timestamp(value: object) -> str:
    """Аккуратно отформатировать время; некорректное значение заменить прочерком."""
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "—"
    return str(timestamp.strftime("%d.%m.%Y %H:%M:%S"))


def load_logs() -> pd.DataFrame:
    """Прочитать последние 50 записей журнала и подготовить их для панели."""
    repository = get_log_repo()
    repository.init_db()
    entries = LogService(repository).get_recent(limit=50)
    if not entries:
        return pd.DataFrame(columns=LOG_COLUMNS)

    rows: list[dict[str, Any]] = []
    for entry in entries:
        category, confidence = parse_prediction(entry.prediction)
        reason_code = entry.reason if entry.reason else "__none__"
        rows.append(
            {
                "id": entry.id,
                "timestamp": entry.timestamp,
                "text": entry.text,
                "status": entry.status,
                "status_label": STATUS_LABELS.get(entry.status, entry.status),
                "reason_code": reason_code,
                "reason_label": REASON_LABELS.get(reason_code, reason_code),
                "category": category,
                "confidence": confidence,
                "explanation": entry.explanation,
            }
        )
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def reset_filters() -> None:
    """Вернуть фильтры к значениям по умолчанию."""
    st.session_state[STATUS_FILTER_KEY] = "all"
    st.session_state[REASON_FILTER_KEY] = []
    st.session_state[SEARCH_KEY] = ""


def render_metrics(data_frame: pd.DataFrame) -> None:
    """Показать три ключевые метрики по отфильтрованным данным."""
    total = len(data_frame)
    blocked = int((data_frame["status"] == "blocked").sum())
    allowed = int((data_frame["status"] == "allowed").sum())

    columns = st.columns(3)
    columns[0].metric("Всего запросов", total, help="С учётом выбранных фильтров")
    columns[1].metric("Заблокировано", blocked, help="Adversarial-атака или аномалия")
    columns[2].metric("Пропущено", allowed, help="Запрос передан защищаемой модели")


def build_table_key(
    status_filter: str,
    selected_reasons: list[str],
    search_query: str,
    table: pd.DataFrame,
) -> str:
    """Стабильный ключ таблицы: сбрасывает выбор при смене фильтров или данных."""
    first_id = int(table["ID"].iloc[0]) if len(table) else 0
    signature = (
        f"{status_filter}|{','.join(sorted(selected_reasons))}|"
        f"{search_query}|{len(table)}|{first_id}"
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:12]
    return f"spl_logs_table_{digest}"


def render_details(row: pd.Series) -> None:
    """Показать подробности выбранной строки."""
    blocked = row["Статус"] == STATUS_LABELS["blocked"]
    request_id = int(row["ID"])
    title = f"Запрос #{request_id} — {'заблокирован' if blocked else 'пропущен'}"

    if blocked:
        st.error(title)
    else:
        st.info(title)

    details = st.columns(3)
    details[0].markdown(f"**Статус:** {row['Статус']}")
    details[1].markdown(f"**Причина:** {row['Причина']}")
    details[2].markdown(f"**Категория модели:** {row['Категория']}")

    st.markdown("**Полный текст запроса**")
    st.write(str(row["Текст запроса"]))

    st.markdown("**Объяснение детектора**")
    explanation = row.get("Объяснение")
    if isinstance(explanation, str) and explanation.strip() and explanation != "—":
        st.write(explanation)
    elif blocked:
        st.caption("Для этого типа блокировки объяснение не предусмотрено.")
    else:
        st.caption("Запрос пропущен, объяснение не требуется.")


def render_table(
    data_frame: pd.DataFrame,
    status_filter: str,
    selected_reasons: list[str],
    search_query: str,
) -> None:
    """Показать интерактивную таблицу журнала и детали выбранной строки."""
    displayed = data_frame.sort_values("id", ascending=False).head(50).copy()
    displayed["Время (UTC)"] = displayed["timestamp"].apply(format_timestamp)
    displayed["Статус"] = displayed["status_label"]
    displayed["Причина"] = displayed["reason_label"]
    displayed["Категория"] = displayed["category"].fillna("—")
    displayed["Уверенность"] = displayed["confidence"].apply(
        lambda value: float(value) * 100
        if isinstance(value, (int, float)) and not pd.isna(value)
        else None
    )
    displayed["Текст запроса"] = displayed["text"]
    displayed["ID"] = displayed["id"]
    displayed["Объяснение"] = displayed["explanation"].fillna("—")

    table = displayed[
        [
            "Время (UTC)",
            "Статус",
            "Причина",
            "Категория",
            "Уверенность",
            "Текст запроса",
            "ID",
            "Объяснение",
        ]
    ].reset_index(drop=True)

    st.subheader("Журнал запросов")
    event = st.dataframe(
        table,
        hide_index=True,
        height=max(160, min(620, 50 + len(table) * 36)),
        on_select="rerun",
        selection_mode="single-row",
        column_order=[
            "Время (UTC)",
            "Статус",
            "Причина",
            "Категория",
            "Уверенность",
            "Текст запроса",
            "ID",
        ],
        key=build_table_key(
            status_filter,
            selected_reasons,
            search_query,
            table,
        ),
        column_config={
            "Время (UTC)": st.column_config.TextColumn(
                "Время (UTC)",
                width="medium",
            ),
            "Статус": st.column_config.TextColumn("Статус", width="small"),
            "Причина": st.column_config.TextColumn("Причина", width="medium"),
            "Категория": st.column_config.TextColumn("Категория", width="medium"),
            "Уверенность": st.column_config.ProgressColumn(
                "Уверенность",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="small",
            ),
            "Текст запроса": st.column_config.TextColumn(
                "Текст запроса",
                width="large",
            ),
            "ID": st.column_config.NumberColumn("ID", format="%d", width="small"),
        },
    )

    selection = getattr(event, "selection", None)
    selected_rows = list(selection.rows) if selection is not None else []
    if selected_rows and 0 <= selected_rows[0] < len(table):
        st.divider()
        render_details(table.iloc[selected_rows[0]])
    else:
        st.info("Выберите строку в таблице, чтобы увидеть детали.")


def render_empty_journal() -> None:
    """Показать подсказку для пустого журнала."""
    st.info(
        "Журнал пуст. Отправь запрос в POST /check, после этого нажми "
        "«Обновить данные» в боковой панели."
    )
    st.code(
        'curl -X POST http://127.0.0.1:8000/check \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"text": "Какие документы нужны для поступления?"}\'',
        language="bash",
    )


def main() -> None:
    """Отрисовать панель мониторинга решений SPL."""
    st.set_page_config(
        page_title="SPL — панель мониторинга",
        layout="wide",
    )
    st.title("SPL — мониторинг запросов")
    st.caption(
        "Последние 50 решений Security & Protection Layer. "
        "Выберите строку в таблице, чтобы увидеть полный текст и объяснение."
    )

    try:
        data_frame = load_logs()
    except Exception:
        logger.exception("Не удалось загрузить журнал SPL")
        st.error(
            "Не удалось загрузить журнал. Запусти API, чтобы создать схему БД, "
            "и проверь путь DB_PATH."
        )
        st.stop()
        return

    with st.sidebar:
        st.header("Фильтры")

        status_filter = str(st.selectbox(
            "Статус",
            options=list(STATUS_FILTER_LABELS),
            format_func=lambda code: STATUS_FILTER_LABELS[code],
            key=STATUS_FILTER_KEY,
        ))

        present_reasons = set(data_frame["reason_code"].dropna().unique())
        available_reasons = [
            code for code in REASON_ORDER if code in present_reasons
        ]
        available_reasons += sorted(
            str(code)
            for code in present_reasons
            if str(code) not in REASON_ORDER
        )
        selected_reasons = list(st.multiselect(
            "Причина",
            options=available_reasons,
            format_func=lambda code: REASON_LABELS.get(code, code),
            key=REASON_FILTER_KEY,
            disabled=not available_reasons,
            help="Можно выбрать несколько причин одновременно",
        ))

        search_query = str(
            st.text_input(
                "Поиск по тексту",
                key=SEARCH_KEY,
                placeholder="Например: общежитие",
            )
        ).strip()

        st.divider()
        st.button(
            "Обновить данные",
            type="primary",
            help="Перечитать последние записи из SQLite",
        )
        st.button(
            "Сбросить фильтры",
            on_click=reset_filters,
            disabled=not (
                status_filter != "all" or selected_reasons or search_query
            ),
        )
        st.caption("Загружаются последние 50 записей.")

    filtered = data_frame
    if status_filter != "all":
        filtered = filtered[filtered["status"] == status_filter]
    if selected_reasons:
        filtered = filtered[filtered["reason_code"].isin(selected_reasons)]
    if search_query:
        filtered = filtered[
            filtered["text"].str.contains(
                search_query,
                case=False,
                na=False,
                regex=False,
            )
        ]

    filters_active = bool(
        status_filter != "all" or selected_reasons or search_query
    )
    if filters_active:
        st.caption(f"Показано {len(filtered)} из {len(data_frame)} последних записей.")

    render_metrics(filtered)

    if data_frame.empty:
        render_empty_journal()
        return

    if filtered.empty:
        st.warning(
            "По выбранным фильтрам записей нет. Измени или сбрось фильтры "
            "в боковой панели."
        )
        return

    st.divider()
    render_table(filtered, status_filter, selected_reasons, search_query)


if __name__ == "__main__":
    main()
