"""Сервис чтения журнала решений."""

import logging

from src.domain.interfaces import LogRepositoryProtocol
from src.domain.models import LogEntry

logger = logging.getLogger(__name__)


class LogService:
    """Предоставляет последние записи журнала для API и панели."""

    def __init__(self, repository: LogRepositoryProtocol) -> None:
        self._repository = repository

    def get_recent(self, limit: int = 50) -> list[LogEntry]:
        """Вернуть последние записи журнала."""
        entries = self._repository.get_recent(limit=limit)
        logger.debug("LogService вернул %s записей", len(entries))
        return entries
