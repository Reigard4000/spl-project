"""HTTP-маршруты SPL."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.domain.models import CheckRequest, CheckResponse, LogEntry
from src.presentation.api.deps import get_check_service, get_log_service
from src.services.check_service import CheckService
from src.services.log_service import LogService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/check", response_model=CheckResponse, summary="Проверить текст")
def check_text(
    request: CheckRequest,
    service: Annotated[CheckService, Depends(get_check_service)],
) -> CheckResponse:
    """Пропустить текст к модели или заблокировать его."""
    logger.info("Получен запрос на проверку длиной %s символов", len(request.text))
    return service.check(request.text)


@router.get("/logs", response_model=list[LogEntry], summary="Последние решения")
def get_logs(
    service: Annotated[LogService, Depends(get_log_service)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[LogEntry]:
    """Вернуть последние записи журнала."""
    return service.get_recent(limit=limit)


@router.get("/health", summary="Проверка доступности")
def health() -> dict[str, str]:
    """Проверить, что API запущен."""
    return {"status": "ok"}
