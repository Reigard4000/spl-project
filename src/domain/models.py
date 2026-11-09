"""Pydantic-модели запросов, ответов и записей журнала."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    """Текстовый запрос пользователя, который нужно проверить."""

    text: str = Field(min_length=1, max_length=10000)


class CheckResponse(BaseModel):
    """Решение SPL: пропустить запрос к модели или заблокировать."""

    allowed: bool
    reason: str | None = None
    prediction: dict[str, Any] | None = None
    explanation: str | None = None


class LogEntry(BaseModel):
    """Запись журнала решений SPL."""

    id: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str
    status: Literal["allowed", "blocked"]
    reason: str | None = None
    prediction: dict[str, Any] | None = None
    explanation: str | None = None
