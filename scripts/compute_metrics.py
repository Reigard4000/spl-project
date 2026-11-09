"""Расчёт метрик качества и скорости детекторов на grouped test split."""

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from src.infrastructure.detectors.adversarial import AdversarialDetector  # noqa: E402
from src.infrastructure.detectors.anomaly import AnomalyDetector  # noqa: E402

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMING_SAMPLE_SIZE = 100


def resolve_path(value: str) -> Path:
    """Превратить путь из настроек в абсолютный относительно корня проекта."""
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_artifacts(path: Path) -> dict[str, Any]:
    """Загрузить pkl-артефакты детектора."""
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}. Сначала запусти scripts/train_detectors.py "
            "или make train."
        )
    with path.open("rb") as file:
        return cast(dict[str, Any], pickle.load(file))


def load_data(csv_path: Path) -> pd.DataFrame:
    """Прочитать CSV с base_id и категориями."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Не найден файл {csv_path}. "
            "Сначала запусти python data/generate_data.py или make data."
        )
    data = pd.read_csv(csv_path)
    required = {"text", "label", "base_id", "category"}
    if not required.issubset(set(data.columns)):
        raise ValueError(
            f"В {csv_path} должны быть колонки {sorted(required)}. Перегенерируй данные."
        )
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(int)
    data["base_id"] = data["base_id"].astype(str)
    data["category"] = data["category"].fillna("unknown").astype(str)
    return data


def load_test_data(csv_path: Path) -> pd.DataFrame:
    """Воспроизвести grouped test split из train_detectors.py."""
    data = load_data(csv_path)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    _, test_positions = next(splitter.split(data, groups=data["base_id"]))
    return data.iloc[test_positions].copy()


def compute_adversarial_auc(
    test_data: pd.DataFrame,
    artifacts_path: Path,
) -> float:
    """Посчитать AUC-ROC на grouped test split."""
    artifacts = load_artifacts(artifacts_path)
    model = artifacts["model"]
    probabilities = model.predict_proba(test_data["text"].tolist())[:, 1]
    return float(
        roc_auc_score(test_data["label"].tolist(), probabilities)
    )


def measure_detectors(
    adversarial_detector: AdversarialDetector,
    anomaly_detector: AnomalyDetector,
    clean_texts: list[str],
) -> tuple[float, float]:
    """Измерить скорость и FPR на чистых текстах grouped test split."""
    timing_texts = clean_texts[:TIMING_SAMPLE_SIZE]

    adversarial_detector.is_suspicious("прогрев детектора")
    anomaly_detector.is_suspicious("прогрев детектора")

    started_at = time.perf_counter()
    for text in timing_texts:
        adversarial_detector.is_suspicious(text)
        anomaly_detector.is_suspicious(text)
    elapsed = time.perf_counter() - started_at
    average_ms = elapsed / len(timing_texts) * 1000 if timing_texts else 0.0

    if clean_texts:
        false_positives = sum(
            1
            for text in clean_texts
            if adversarial_detector.is_suspicious(text)
            or anomaly_detector.is_suspicious(text)
        )
        false_positive_rate = false_positives / len(clean_texts)
    else:
        false_positive_rate = 0.0

    return average_ms, false_positive_rate


def main() -> None:
    """Посчитать метрики и вывести таблицу «метрика | значение | цель»."""
    settings = get_settings()
    setup_logging(settings.log_level)

    csv_path = resolve_path(settings.train_csv)
    adversarial_path = resolve_path(settings.adversarial_model_path)
    anomaly_path = resolve_path(settings.anomaly_model_path)

    try:
        test_data = load_test_data(csv_path)
        adversarial_auc = compute_adversarial_auc(test_data, adversarial_path)

        clean_texts = test_data[test_data["label"] == 0]["text"].tolist()
        adversarial_detector = AdversarialDetector(str(adversarial_path))
        anomaly_detector = AnomalyDetector(str(anomaly_path))
        average_ms, false_positive_rate = measure_detectors(
            adversarial_detector,
            anomaly_detector,
            clean_texts,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info("%-52s | %-14s | %s", "Метрика", "Значение", "Цель")
    logger.info("%s", "-" * 88)
    logger.info(
        "%-52s | %-14s | %s",
        "AUC-ROC adversarial-детектора на grouped test",
        f"{adversarial_auc:.4f}",
        ">= 0.90",
    )
    logger.info(
        "%-52s | %-14s | %s",
        "Среднее время adversarial + anomaly на текст",
        f"{average_ms:.2f} мс",
        "<= 50 мс",
    )
    logger.info(
        "%-52s | %-14s | %s",
        "Доля ложных срабатываний на тестовых чистых текстах",
        f"{false_positive_rate:.1%}",
        "<= 1%",
    )


if __name__ == "__main__":
    main()
