"""Тесты бизнес-логики CheckService."""

import pytest

from src.core.exceptions import DetectorError
from src.infrastructure.db.sqlite_repo import SQLiteLogRepository
from src.services.check_service import CheckService
from tests.conftest import StubDetector, StubVictimModel


def test_clean_text_is_allowed(
    check_service: CheckService,
    log_repo: SQLiteLogRepository,
    victim_model: StubVictimModel,
) -> None:
    """Чистый текст проходит к модели и попадает в журнал как allowed."""
    response = check_service.check("Какие документы нужны для поступления?")

    assert response.allowed is True
    assert response.reason is None
    assert response.prediction == victim_model.prediction
    assert victim_model.calls == 1

    entries = log_repo.get_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].status == "allowed"
    assert entries[0].reason is None


def test_adversarial_text_is_blocked(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """Adversarial-текст блокируется без обращения к модели."""
    adversarial_detector.suspicious = True
    service = CheckService(
        adversarial_detector=adversarial_detector,
        anomaly_detector=anomaly_detector,
        victim_model=victim_model,
        log_repo=log_repo,
    )

    response = service.check("K@k1e d0kyMeHTbl HY}KHo?")

    assert response.allowed is False
    assert response.reason == "adversarial"
    assert response.prediction is None
    assert response.explanation is not None
    assert "Наличие невидимых символов" in response.explanation
    assert victim_model.calls == 0

    entries = log_repo.get_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].status == "blocked"
    assert entries[0].reason == "adversarial"
    assert entries[0].explanation == response.explanation


def test_anomaly_text_is_blocked(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """Аномальный текст блокируется без обращения к модели."""
    anomaly_detector.suspicious = True
    service = CheckService(
        adversarial_detector=adversarial_detector,
        anomaly_detector=anomaly_detector,
        victim_model=victim_model,
        log_repo=log_repo,
    )

    response = service.check("!!!!????####")

    assert response.allowed is False
    assert response.reason == "anomaly"
    assert victim_model.calls == 0

    entries = log_repo.get_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].reason == "anomaly"


def test_init_db_preserves_existing_logs(
    check_service: CheckService,
    log_repo: SQLiteLogRepository,
) -> None:
    """Повторная инициализация схемы не удаляет уже сохранённые записи."""
    check_service.check("Какие документы нужны для поступления?")
    log_repo.init_db()

    entries = log_repo.get_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].status == "allowed"


def test_detector_error_propagates(
    log_repo: SQLiteLogRepository,
    adversarial_detector: StubDetector,
    anomaly_detector: StubDetector,
    victim_model: StubVictimModel,
) -> None:
    """Ошибка детектора не маскируется и не создаёт запись журнала."""
    adversarial_detector.error = DetectorError("детектор сломан")
    service = CheckService(
        adversarial_detector=adversarial_detector,
        anomaly_detector=anomaly_detector,
        victim_model=victim_model,
        log_repo=log_repo,
    )

    with pytest.raises(DetectorError):
        service.check("Обычный текст запроса")

    assert log_repo.get_recent(limit=10) == []
