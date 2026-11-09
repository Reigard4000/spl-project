"""Сервис проверки запроса двумя детекторами и защищаемой моделью."""

import logging
from typing import Any, Literal

from src.domain.interfaces import (
    DetectorProtocol,
    LogRepositoryProtocol,
    VictimModelProtocol,
)
from src.domain.models import CheckResponse, LogEntry

logger = logging.getLogger(__name__)


class CheckService:
    """Принимает решение: пропустить текст к модели или заблокировать."""

    def __init__(
        self,
        adversarial_detector: DetectorProtocol,
        anomaly_detector: DetectorProtocol,
        victim_model: VictimModelProtocol,
        log_repo: LogRepositoryProtocol,
    ) -> None:
        self._adversarial_detector = adversarial_detector
        self._anomaly_detector = anomaly_detector
        self._victim_model = victim_model
        self._log_repo = log_repo

    def check(self, text: str) -> CheckResponse:
        """Проверить текст и записать решение в журнал."""
        is_adversarial, top_features = self._adversarial_detector.check(text)
        if is_adversarial:
            explanation = self._format_explanation(top_features)
            logger.warning("Запрос заблокирован: adversarial-детектор")
            return self._block(text, "adversarial", explanation)

        is_anomaly, _ = self._anomaly_detector.check(text)
        if is_anomaly:
            logger.warning("Запрос заблокирован: anomaly-детектор")
            return self._block(text, "anomaly", explanation=None)

        prediction = self._victim_model.predict(text)
        logger.info("Запрос разрешён, модель вернула категорию")
        self._save_log(
            text=text,
            status="allowed",
            reason=None,
            prediction=prediction,
            explanation=None,
        )
        return CheckResponse(
            allowed=True,
            reason=None,
            prediction=prediction,
            explanation=None,
        )

    def _block(
        self,
        text: str,
        reason: str,
        explanation: str | None,
    ) -> CheckResponse:
        """Сформировать ответ о блокировке и сохранить его в журнал."""
        self._save_log(
            text=text,
            status="blocked",
            reason=reason,
            prediction=None,
            explanation=explanation,
        )
        return CheckResponse(
            allowed=False,
            reason=reason,
            prediction=None,
            explanation=explanation,
        )

    @staticmethod
    def _format_explanation(features: dict[str, float]) -> str | None:
        """Собрать текстовое объяснение из топ-признаков детектора."""
        if not features:
            return None
        return f"Наиболее значимые признаки: {', '.join(features)}"

    def _save_log(
        self,
        text: str,
        status: Literal["allowed", "blocked"],
        reason: str | None,
        prediction: dict[str, Any] | None,
        explanation: str | None,
    ) -> None:
        """Записать решение SPL в хранилище."""
        entry = LogEntry(
            text=text,
            status=status,
            reason=reason,
            prediction=prediction,
            explanation=explanation,
        )
        self._log_repo.save(entry)
