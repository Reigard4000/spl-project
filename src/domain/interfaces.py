"""Protocol-интерфейсы для dependency injection и подмены в тестах."""

from typing import Any, Protocol

from src.domain.models import LogEntry


class LogRepositoryProtocol(Protocol):
    """Хранилище журнала решений."""

    def save(self, entry: LogEntry) -> LogEntry:
        """Сохранить запись и вернуть её с присвоенным id."""
        ...

    def get_recent(self, limit: int = 50) -> list[LogEntry]:
        """Вернуть последние записи в порядке от новых к старым."""
        ...


class DetectorProtocol(Protocol):
    """Детектор подозрительного текста."""

    def is_suspicious(self, text: str) -> bool:
        """Вернуть True, если текст является подозрительным."""
        ...

    def check(self, text: str) -> tuple[bool, dict[str, float]]:
        """Проверить текст и вернуть решение вместе с топ-признаками."""
        ...


class VictimModelProtocol(Protocol):
    """Защищаемая ML-модель."""

    def predict(self, text: str) -> dict[str, Any]:
        """Вернуть предсказание модели для текста."""
        ...
