"""Обёртка над deepvk/GeRaCl-USER2-base (zero-shot классификация)."""

import logging
import threading
from typing import Any, Literal

from src.core.exceptions import ModelError

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 512

BackendName = Literal["hf", "geracl"]


class GeraclModel:
    """Лениво загружает pipeline и классифицирует русский текст.

    Основной путь — стандартный ``transformers.pipeline``. Если архитектура
    модели не поддерживается стандартным pipeline, используется официальный
    пакет ``geracl`` (если он установлен).
    """

    def __init__(self, model_name: str, categories: list[str]) -> None:
        self._model_name = model_name
        self._categories = list(categories)
        self._pipeline: Any = None
        self._backend: BackendName | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        """Загрузить pipeline один раз при первом обращении."""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            self._pipeline, self._backend = self._create_pipeline()

        return self._pipeline

    def _create_pipeline(self) -> tuple[Any, BackendName]:
        """Попробовать стандартный pipeline, затем официальный geracl."""
        try:
            from transformers import pipeline

            logger.info("Загружается модель %s (это может занять время)", self._model_name)
            standard_pipeline = pipeline(
                "zero-shot-classification",
                model=self._model_name,
            )
            return standard_pipeline, "hf"
        except Exception as standard_error:
            logger.info(
                "Стандартный transformers pipeline не загрузил модель %s: %s. "
                "Пробую официальный geracl backend.",
                self._model_name,
                standard_error,
            )
            return self._load_geracl_pipeline(), "geracl"

    def _load_geracl_pipeline(self) -> Any:
        """Загрузить модель через официальный пакет geracl (если он есть)."""
        try:
            import torch
            from geracl import (
                GeraclConfigHF,
                GeraclHF,
                ZeroShotClassificationPipeline,
            )
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ModelError(
                f"Не удалось загрузить модель {self._model_name} стандартным pipeline, "
                "а официальный пакет geracl не установлен. Установи его и зависимости: "
                'pip install "geracl @ git+https://github.com/deepvk/geracl" '
                "loguru pytorch-lightning torchmetrics."
            ) from exc

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = GeraclConfigHF.from_pretrained(self._model_name)
            config.device = device
            model = GeraclHF.from_pretrained(self._model_name, config=config)
            model = model.to(device).eval()

            core = getattr(model, "_classification_core", None)
            if core is not None:
                core._device = device

            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                self._model_name
            )
            geracl_pipeline = ZeroShotClassificationPipeline(
                model,
                tokenizer,
                device=device,
                progress_bar=False,
            )
        except Exception as exc:
            raise ModelError(
                f"Не удалось загрузить модель {self._model_name} через geracl: {exc}"
            ) from exc

        logger.info("Модель %s загружена через официальный geracl backend", self._model_name)
        return geracl_pipeline

    def predict(self, text: str) -> dict[str, Any]:
        """Вернуть наиболее вероятную категорию и scores по всем категориям."""
        model = self._load()
        truncated_text = text[:MAX_TEXT_LENGTH]

        try:
            if self._backend == "geracl":
                prediction = self._predict_geracl(model, truncated_text)
            else:
                prediction = self._predict_hf(model, truncated_text)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Ошибка защищаемой модели: {exc}") from exc

        logger.info("Модель вернула категорию: %s", prediction["category"])
        return prediction

    def _predict_hf(self, model: Any, text: str) -> dict[str, Any]:
        """Предсказание через стандартный transformers pipeline."""
        result = model(
            text,
            candidate_labels=self._categories,
            multi_label=False,
        )
        if not isinstance(result, dict):
            raise ModelError("Модель вернула неожиданный формат ответа")

        labels = result["labels"]
        scores = result["scores"]
        return {
            "category": str(labels[0]),
            "scores": {
                str(label): float(score)
                for label, score in zip(labels, scores, strict=True)
            },
        }

    def _predict_geracl(self, model: Any, text: str) -> dict[str, Any]:
        """Предсказание через официальный geracl pipeline."""
        import torch

        similarities = model.get_similarities(
            [text],
            self._categories,
            batch_size=1,
        )[0]
        probabilities = torch.softmax(similarities, dim=0).tolist()
        scores = {
            label: float(score)
            for label, score in zip(self._categories, probabilities, strict=True)
        }
        top_label = max(scores, key=lambda label: scores[label])
        return {"category": top_label, "scores": scores}
