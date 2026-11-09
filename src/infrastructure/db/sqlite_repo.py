"""SQLite-реализация репозитория журнала решений."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from src.core.exceptions import RepositoryError
from src.domain.models import LogEntry

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SQLiteLogRepository:
    """Потокобезопасный репозиторий: одно соединение на одну операцию."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._keeper: sqlite3.Connection | None = None

        if self._is_memory:
            # Общая in-memory база живёт, пока открыто хотя бы одно соединение.
            db_name = f"spl_memory_{uuid.uuid4().hex}"
            self._uri = f"file:{db_name}?mode=memory&cache=shared"
            self._keeper = sqlite3.connect(
                self._uri,
                uri=True,
                check_same_thread=False,
            )
        else:
            self._uri = db_path
            try:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RepositoryError(
                    f"Не удалось создать каталог для базы {self._db_path}: {exc}"
                ) from exc

    def _connect(self) -> sqlite3.Connection:
        """Открыть новое соединение с базой."""
        try:
            connection = sqlite3.connect(
                self._uri,
                uri=self._is_memory,
                timeout=5.0,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error as exc:
            raise RepositoryError(f"Не удалось открыть базу {self._db_path}: {exc}") from exc

    def init_db(self) -> None:
        """Идемпотентно создать схему журнала, сохраняя существующие записи."""
        try:
            schema = SCHEMA_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise RepositoryError(f"Не удалось прочитать схему {SCHEMA_PATH}: {exc}") from exc

        connection = self._connect()
        try:
            connection.executescript(schema)
            connection.commit()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Не удалось инициализировать базу: {exc}") from exc
        finally:
            connection.close()
        logger.info("База данных инициализирована: %s", self._db_path)

    def save(self, entry: LogEntry) -> LogEntry:
        """Сохранить запись журнала и вернуть её с присвоенным id."""
        prediction_json = (
            json.dumps(entry.prediction, ensure_ascii=False)
            if entry.prediction is not None
            else None
        )

        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO logs (timestamp, text, status, reason, prediction, explanation)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.timestamp.isoformat(),
                    entry.text,
                    entry.status,
                    entry.reason,
                    prediction_json,
                    entry.explanation,
                ),
            )
            connection.commit()
            entry_id = cursor.lastrowid
        except sqlite3.Error as exc:
            raise RepositoryError(f"Не удалось сохранить запись журнала: {exc}") from exc
        finally:
            connection.close()

        if entry_id is None:
            raise RepositoryError("SQLite не вернул id сохранённой записи")

        saved_entry = entry.model_copy(update={"id": entry_id})
        logger.info(
            "Запись сохранена: id=%s, status=%s, reason=%s",
            saved_entry.id,
            saved_entry.status,
            saved_entry.reason,
        )
        return saved_entry

    def get_recent(self, limit: int = 50) -> list[LogEntry]:
        """Вернуть последние записи журнала (от новых к старым)."""
        safe_limit = max(0, limit)
        if safe_limit == 0:
            return []

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, timestamp, text, status, reason, prediction, explanation
                FROM logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Не удалось прочитать журнал: {exc}") from exc
        finally:
            connection.close()

        entries: list[LogEntry] = []
        for row in rows:
            try:
                entries.append(self._row_to_entry(row))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Пропущена некорректная запись журнала id=%s: %s",
                    row["id"],
                    exc,
                )

        logger.debug("Прочитано записей журнала: %s", len(entries))
        return entries

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LogEntry:
        """Преобразовать строку SQLite в доменную модель."""
        prediction_json = row["prediction"]
        prediction = (
            json.loads(prediction_json) if prediction_json is not None else None
        )
        return LogEntry(
            id=int(row["id"]),
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
            text=str(row["text"]),
            status=str(row["status"]),
            reason=str(row["reason"]) if row["reason"] is not None else None,
            prediction=prediction,
            explanation=(
                str(row["explanation"]) if row["explanation"] is not None else None
            ),
        )
