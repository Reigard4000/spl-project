"""Тесты FastAPI-маршрутов SPL через httpx AsyncClient."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.infrastructure.db.sqlite_repo import SQLiteLogRepository
from src.presentation.api.app import create_app
from src.presentation.api.deps import (
    get_check_service,
    get_log_repo,
    get_log_service,
)
from src.services.check_service import CheckService
from src.services.log_service import LogService
from tests.conftest import StubDetector, StubVictimModel

CLEAN_TEXT = "Подскажите, какие документы нужны для поступления?"
ADVERSARIAL_TEXT = "K@k1e d0kyMeHTbl HY}KHo?"
ANOMALY_TEXT = "!!!!????####"


def build_app(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> FastAPI:
    """Создать приложение и подменить инфраструктурные зависимости."""

    def override_log_repo() -> SQLiteLogRepository:
        return log_repo

    def override_check_service() -> CheckService:
        return CheckService(
            adversarial_detector=adversarial_detector,
            anomaly_detector=anomaly_detector,
            victim_model=victim_model,
            log_repo=log_repo,
        )

    def override_log_service() -> LogService:
        return LogService(log_repo)

    app = create_app()
    app.dependency_overrides[get_log_repo] = override_log_repo
    app.dependency_overrides[get_check_service] = override_check_service
    app.dependency_overrides[get_log_service] = override_log_service
    return app


@asynccontextmanager
async def api_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Поднять ASGI-приложение в httpx без реального сетевого порта."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.anyio
async def test_check_clean_text(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """Чистый запрос проходит и возвращает предсказание модели."""
    app = build_app(log_repo, adversarial_detector, anomaly_detector, victim_model)

    async with api_client(app) as client:
        response = await client.post("/check", json={"text": CLEAN_TEXT})

    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is True
    assert payload["reason"] is None
    assert payload["prediction"]["category"] == "документы"


@pytest.mark.anyio
async def test_check_adversarial_text(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """Adversarial-запрос блокируется с причиной adversarial."""
    adversarial_detector.suspicious = True
    app = build_app(log_repo, adversarial_detector, anomaly_detector, victim_model)

    async with api_client(app) as client:
        response = await client.post("/check", json={"text": ADVERSARIAL_TEXT})

    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is False
    assert payload["reason"] == "adversarial"
    assert "Наличие невидимых символов" in payload["explanation"]


@pytest.mark.anyio
async def test_check_anomaly_text(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """Аномальный запрос блокируется с причиной anomaly."""
    anomaly_detector.suspicious = True
    app = build_app(log_repo, adversarial_detector, anomaly_detector, victim_model)

    async with api_client(app) as client:
        response = await client.post("/check", json={"text": ANOMALY_TEXT})

    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is False
    assert payload["reason"] == "anomaly"


@pytest.mark.anyio
async def test_health() -> None:
    """Health-check возвращает статус ok."""
    app = create_app()

    async with api_client(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_logs_returns_saved_entries(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """GET /logs отдаёт записи, созданные POST /check."""
    app = build_app(log_repo, adversarial_detector, anomaly_detector, victim_model)

    async with api_client(app) as client:
        await client.post("/check", json={"text": CLEAN_TEXT})
        response = await client.get("/logs", params={"limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["status"] == "allowed"
    assert payload[0]["text"] == CLEAN_TEXT


@pytest.mark.anyio
async def test_check_validation_error(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """Пустой текст отклоняется с кодом 422."""
    app = build_app(log_repo, adversarial_detector, anomaly_detector, victim_model)

    async with api_client(app) as client:
        response = await client.post("/check", json={"text": ""})

    assert response.status_code == 422
    assert response.json()["error"] == "ValidationError"
