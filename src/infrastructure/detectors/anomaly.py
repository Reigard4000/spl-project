"""Детектор аномалий на IsolationForest и 12 текстовых признаках."""

import logging
import math
import pickle
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from src.core.exceptions import DetectorError

logger = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
INVISIBLE_CHARACTERS = {"\u200b", "\u00ad"}

FEATURE_NAMES = (
    "length",
    "word_count",
    "average_word_length",
    "non_alphabetic_share",
    "uppercase_share",
    "unique_characters",
    "shannon_entropy",
    "punctuation_count",
    "cyrillic_share",
    "latin_share",
    "invisible_characters",
    "mixed_script_word_share",
)


def _shannon_entropy(text: str) -> float:
    """Посчитать энтропию Шеннона по частотам символов."""
    if not text:
        return 0.0
    length = len(text)
    frequencies = Counter(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in frequencies.values()
    )


def extract_features(text: str) -> list[float]:
    """Построить 12 числовых признаков текста."""
    length = float(len(text))
    words = WORD_PATTERN.findall(text)
    word_count = float(len(words))
    average_word_length = (
        sum(len(word) for word in words) / len(words) if words else 0.0
    )
    non_alphabetic_share = (
        sum(1 for character in text if not character.isalpha()) / len(text)
        if text
        else 1.0
    )
    letters = sum(1 for character in text if character.isalpha())
    uppercase_share = (
        sum(1 for character in text if character.isalpha() and character.isupper()) / letters
        if letters
        else 0.0
    )
    unique_characters = float(len(set(text)))
    punctuation_count = float(
        sum(
            1
            for character in text
            if unicodedata.category(character).startswith("P")
        )
    )
    cyrillic_letters = len(CYRILLIC_PATTERN.findall(text))
    latin_letters = len(LATIN_PATTERN.findall(text))
    cyrillic_share = cyrillic_letters / letters if letters else 0.0
    latin_share = latin_letters / letters if letters else 0.0
    invisible_characters = float(
        sum(1 for character in text if character in INVISIBLE_CHARACTERS)
    )
    mixed_script_words = sum(
        1
        for word in words
        if CYRILLIC_PATTERN.search(word) and LATIN_PATTERN.search(word)
    )
    mixed_script_word_share = (
        mixed_script_words / len(words) if words else 0.0
    )

    return [
        length,
        word_count,
        average_word_length,
        non_alphabetic_share,
        uppercase_share,
        unique_characters,
        _shannon_entropy(text),
        punctuation_count,
        cyrillic_share,
        latin_share,
        invisible_characters,
        mixed_script_word_share,
    ]


class AnomalyDetector:
    """Лениво загружает IsolationForest из pkl-файла."""

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._model: Any = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        """Загрузить модель один раз при первом обращении."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            if not self._model_path.exists():
                raise DetectorError(
                    f"Файл anomaly-детектора не найден: {self._model_path}. "
                    "Запусти scripts/train_detectors.py или make train."
                )
            try:
                with self._model_path.open("rb") as file:
                    artifacts = pickle.load(file)
                self._model = artifacts["model"]
            except Exception as exc:
                raise DetectorError(
                    f"Не удалось загрузить anomaly-детектор из {self._model_path}: {exc}"
                ) from exc

            self._loaded = True
            logger.info("Anomaly-детектор загружен из %s", self._model_path)

    def is_anomaly(self, text: str) -> bool:
        """Вернуть True, если текст является аномалией."""
        self._load()
        if self._model is None:
            raise DetectorError("Anomaly-детектор не инициализирован")

        try:
            prediction = int(self._model.predict([extract_features(text)])[0])
        except Exception as exc:
            raise DetectorError(f"Ошибка anomaly-детектора: {exc}") from exc

        return prediction == -1

    def check(self, text: str) -> tuple[bool, dict[str, float]]:
        """Проверить текст; anomaly-детектор не строит SHAP-объяснение."""
        return self.is_anomaly(text), {}

    def is_suspicious(self, text: str) -> bool:
        """Единый интерфейс детектора для CheckService."""
        return self.is_anomaly(text)
