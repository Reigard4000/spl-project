"""Общие фикстуры и заглушки для тестов."""

from typing import Any

import pytest

from src.infrastructure.db.sqlite_repo import SQLiteLogRepository
from src.services.check_service import CheckService


class StubDetector:
    """Управляемая заглушка детектора."""

    def __init__(
        self,
        suspicious: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.suspicious = suspicious
        self.error = error

    def is_suspicious(self, text: str) -> bool:
        """Вернуть заданный результат или выбросить заданную ошибку."""
        if self.error is not None:
            raise self.error
        return self.suspicious

    def check(self, text: str) -> tuple[bool, dict[str, float]]:
        """Вернуть решение и тестовое объяснение для заблокированного текста."""
        suspicious = self.is_suspicious(text)
        if not suspicious:
            return False, {}
        return True, {"Наличие невидимых символов": 0.5}


class StubVictimModel:
    """Заглушка защищаемой модели с фиксированным ответом."""

    def __init__(self) -> None:
        self.calls = 0
        self.prediction: dict[str, Any] = {
            "category": "документы",
            "scores": {"документы": 0.9, "сроки": 0.1},
        }

    def predict(self, text: str) -> dict[str, Any]:
        """Посчитать вызов и вернуть заранее заданное предсказание."""
        self.calls += 1
        return self.prediction


@pytest.fixture
def log_repo() -> SQLiteLogRepository:
    """In-memory SQLite с созданной схемой."""
    repository = SQLiteLogRepository(":memory:")
    repository.init_db()
    return repository


@pytest.fixture
def adversarial_detector() -> StubDetector:
    """По умолчанию adversarial-детектор считает текст чистым."""
    return StubDetector(suspicious=False)


@pytest.fixture
def anomaly_detector() -> StubDetector:
    """По умолчанию anomaly-детектор считает текст чистым."""
    return StubDetector(suspicious=False)


@pytest.fixture
def victim_model() -> StubVictimModel:
    """Заглушка защищаемой модели."""
    return StubVictimModel()


@pytest.fixture
def check_service(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> CheckService:
    """CheckService с подконтрольными зависимостями."""
    return CheckService(
        adversarial_detector=adversarial_detector,
        anomaly_detector=anomaly_detector,
        victim_model=victim_model,
        log_repo=log_repo,
    )


@pytest.fixture
def anyio_backend() -> str:
    """Запускать async-тесты через asyncio."""
    return "asyncio"
