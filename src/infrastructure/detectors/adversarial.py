"""Детектор adversarial-текстов на TF-IDF, признаках и LightGBM с SHAP."""

import logging
import pickle
import threading
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.core.exceptions import DetectorError
from src.infrastructure.detectors.anomaly import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 20000
ADVERSARIAL_THRESHOLD = 0.5
TOP_FEATURES_COUNT = 3

FEATURE_LABELS = {
    "length": "Длина текста",
    "word_count": "Количество слов",
    "average_word_length": "Средняя длина слова",
    "non_alphabetic_share": "Доля не-алфавитных символов",
    "uppercase_share": "Доля заглавных букв",
    "unique_characters": "Количество уникальных символов",
    "shannon_entropy": "Энтропия текста",
    "punctuation_count": "Количество знаков препинания",
    "cyrillic_share": "Доля кириллицы",
    "latin_share": "Доля латиницы",
    "invisible_characters": "Наличие невидимых символов",
    "mixed_script_word_share": "Смешение алфавитов",
}


def text_statistics_matrix(texts: Sequence[str]) -> list[list[float]]:
    """Построить матрицу текстовых признаков для FeatureUnion."""
    return [extract_features(str(text)) for text in texts]


def build_adversarial_model() -> Any:
    """Создать обучаемый pipeline: TF-IDF + статистики текста + LightGBM."""
    features = FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    max_features=MAX_TFIDF_FEATURES,
                ),
            ),
            (
                "text_stats",
                FunctionTransformer(
                    text_statistics_matrix,
                    validate=False,
                ),
            ),
        ]
    )
    classifier = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def _humanize_feature(raw_name: str) -> str:
    """Превратить техническое имя признака в понятную подпись."""
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]

    if "\u200b" in raw_name or "\u00ad" in raw_name:
        return "Наличие невидимых символов"

    has_cyrillic = any(
        "а" <= character.lower() <= "я" or character.lower() == "ё"
        for character in raw_name
    )
    has_latin = any("a" <= character.lower() <= "z" for character in raw_name)
    if has_cyrillic and has_latin:
        return "Смешение алфавитов"
    if any(character.isdigit() for character in raw_name):
        return "Цифры и служебные символы"
    if any(character in "@#$%^&*_+=<>/\\|{}[]" for character in raw_name):
        return "Спецсимволы в тексте"

    display_name = raw_name.replace("\u200b", "").replace("\u00ad", "").strip()
    if not display_name:
        return "Невидимые символы или пробелы"
    return f"Фрагмент «{display_name}»"


class AdversarialDetector:
    """Лениво загружает pipeline и объясняет решения через SHAP."""

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._model: Any = None
        self._feature_union: Any = None
        self._classifier: Any = None
        self._explainer: Any = None
        self._feature_names: list[str] = []
        self._threshold = 0.5
        self._loaded = False
        self._load_lock = threading.Lock()
        self._explain_lock = threading.Lock()

    def _load(self) -> None:
        """Загрузить артефакты один раз при первом обращении."""
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return
            if not self._model_path.exists():
                raise DetectorError(
                    f"Файл adversarial-детектора не найден: {self._model_path}. "
                    "Запусти scripts/train_detectors.py или make train."
                )
            try:
                with self._model_path.open("rb") as file:
                    artifacts = pickle.load(file)
                self._model = artifacts["model"]
                self._threshold = float(
                    artifacts.get("threshold", ADVERSARIAL_THRESHOLD)
                )
                self._feature_union = self._model.named_steps["features"]
                self._classifier = self._model.named_steps["classifier"]
                tfidf = dict(self._feature_union.transformer_list)["tfidf"]
                self._feature_names = [
                    str(name) for name in tfidf.get_feature_names_out()
                ] + list(FEATURE_NAMES)
            except Exception as exc:
                raise DetectorError(
                    f"Не удалось загрузить adversarial-детектор из {self._model_path}: {exc}"
                ) from exc

            self._loaded = True
            logger.info("Adversarial-детектор загружен из %s", self._model_path)

    def _transform(self, text: str) -> Any:
        """Построить матрицу признаков для одного текста."""
        if self._feature_union is None:
            raise DetectorError("Adversarial-детектор не инициализирован")
        try:
            return self._feature_union.transform([text])
        except Exception as exc:
            raise DetectorError(f"Ошибка подготовки признаков: {exc}") from exc

    def _probability(self, features: Any) -> float:
        """Вернуть вероятность класса adversarial."""
        if self._classifier is None:
            raise DetectorError("Adversarial-детектор не инициализирован")
        try:
            return float(self._classifier.predict_proba(features)[0][1])
        except Exception as exc:
            raise DetectorError(f"Ошибка adversarial-детектора: {exc}") from exc

    def _get_explainer(self) -> Any:
        """Лениво создать TreeExplainer для LightGBM."""
        if self._explainer is not None:
            return self._explainer

        with self._explain_lock:
            if self._explainer is not None:
                return self._explainer
            try:
                import shap

                self._explainer = shap.TreeExplainer(self._classifier)
            except Exception as exc:
                raise DetectorError(f"Не удалось создать SHAP-объяснение: {exc}") from exc
        return self._explainer

    @staticmethod
    def _class_one_values(shap_values: Any) -> np.ndarray:
        """Привести разные форматы вывода SHAP к вектору вкладов класса 1."""
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1])
        else:
            values = np.asarray(shap_values)

        if values.ndim == 3:
            if values.shape[0] == 1 and values.shape[2] >= 2:
                values = values[0, :, 1]
            elif values.shape[0] >= 2 and values.shape[1] == 1:
                values = values[1, 0, :]
            else:
                values = values.reshape(values.shape[0], values.shape[1], -1)[:, :, -1]
        if values.ndim == 2:
            values = values[0] if values.shape[0] == 1 else values[:, 0]
        return np.asarray(values).reshape(-1)

    def _explain_features(self, features: Any) -> dict[str, float]:
        """Вернуть топ-3 признака по положительному вкладу SHAP."""
        dense_features = (
            features.toarray() if hasattr(features, "toarray") else np.asarray(features)
        )
        explainer = self._get_explainer()
        with self._explain_lock:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="LightGBM binary classifier with TreeExplainer",
                )
                shap_values = explainer.shap_values(dense_features)
        values = self._class_one_values(shap_values)

        if len(values) != len(self._feature_names):
            raise DetectorError(
                "Размерность SHAP-значений не совпадает с числом признаков"
            )

        positive_indices = [
            index for index, value in enumerate(values) if float(value) > 0
        ]
        indices = positive_indices or list(range(len(values)))
        indices.sort(key=lambda index: abs(float(values[index])), reverse=True)

        top_features: dict[str, float] = {}
        for index in indices:
            label = _humanize_feature(self._feature_names[index])
            if label in top_features:
                continue
            top_features[label] = float(values[index])
            if len(top_features) == TOP_FEATURES_COUNT:
                break
        return top_features

    def check(self, text: str) -> tuple[bool, dict[str, float]]:
        """Проверить текст и вернуть решение вместе с топ-3 SHAP-признаками."""
        self._load()
        features = self._transform(text)
        suspicious = self._probability(features) >= self._threshold
        if not suspicious:
            return False, {}

        try:
            explanation = self._explain_features(features)
        except Exception:
            logger.exception("Не удалось построить SHAP-объяснение")
            explanation = {}
        return True, explanation

    def is_adversarial(self, text: str) -> bool:
        """Вернуть True, если текст похож на adversarial-атаку."""
        self._load()
        features = self._transform(text)
        return self._probability(features) >= self._threshold

    def is_suspicious(self, text: str) -> bool:
        """Единый интерфейс детектора для CheckService."""
        return self.is_adversarial(text)
