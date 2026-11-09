"""Провайдеры зависимостей для FastAPI и Streamlit."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.core.config import get_settings
from src.infrastructure.db.sqlite_repo import SQLiteLogRepository
from src.infrastructure.detectors.adversarial import AdversarialDetector
from src.infrastructure.detectors.anomaly import AnomalyDetector
from src.infrastructure.victim.geracl import GeraclModel
from src.services.check_service import CheckService
from src.services.log_service import LogService


@lru_cache
def get_log_repo() -> SQLiteLogRepository:
    """Вернуть singleton-репозиторий SQLite."""
    settings = get_settings()
    return SQLiteLogRepository(settings.db_path)


@lru_cache
def get_adversarial_detector() -> AdversarialDetector:
    """Вернуть singleton adversarial-детектора (артефакты загрузятся лениво)."""
    settings = get_settings()
    return AdversarialDetector(settings.adversarial_model_path)


@lru_cache
def get_anomaly_detector() -> AnomalyDetector:
    """Вернуть singleton anomaly-детектора (модель загрузится лениво)."""
    settings = get_settings()
    return AnomalyDetector(settings.anomaly_model_path)


@lru_cache
def get_victim_model() -> GeraclModel:
    """Вернуть singleton защищаемой модели (загрузится лениво)."""
    settings = get_settings()
    return GeraclModel(settings.model_name, settings.categories)


@lru_cache
def get_check_service(
    adversarial_detector: Annotated[
        AdversarialDetector, Depends(get_adversarial_detector)
    ],
    anomaly_detector: Annotated[AnomalyDetector, Depends(get_anomaly_detector)],
    victim_model: Annotated[GeraclModel, Depends(get_victim_model)],
    log_repo: Annotated[SQLiteLogRepository, Depends(get_log_repo)],
) -> CheckService:
    """Собрать CheckService из инфраструктурных зависимостей."""
    return CheckService(
        adversarial_detector=adversarial_detector,
        anomaly_detector=anomaly_detector,
        victim_model=victim_model,
        log_repo=log_repo,
    )


@lru_cache
def get_log_service(
    log_repo: Annotated[SQLiteLogRepository, Depends(get_log_repo)],
) -> LogService:
    """Собрать LogService."""
    return LogService(log_repo)
