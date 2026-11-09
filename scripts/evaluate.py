"""Честная оценка детекторов на grouped test split, PR-кривые и отчёты."""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from src.infrastructure.detectors.anomaly import extract_features  # noqa: E402

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
PR_CURVE_PATH = ROOT / "models" / "pr_curve.png"
CategoryMetrics = dict[str, dict[str, float | int]]


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
            f"Не найден файл {csv_path}. Сначала запусти make data."
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
    if "attack_type" not in data.columns:
        data["attack_type"] = ""
    else:
        data["attack_type"] = data["attack_type"].fillna("").astype(str)
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


def binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float],
) -> dict[str, Any]:
    """Посчитать метрики бинарной классификации."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def per_attack_type_metrics(
    data: pd.DataFrame,
    predictions: pd.Series,
) -> dict[str, dict[str, float | int]]:
    """Посчитать поддержку, найденные атаки и Recall по типам атак."""
    result: dict[str, dict[str, float | int]] = {}
    attacks = data[data["label"] == 1]
    for attack_type, group in attacks.groupby("attack_type"):
        if not attack_type:
            continue
        support = len(group)
        detected = int((predictions.loc[group.index] == 1).sum())
        result[str(attack_type)] = {
            "support": support,
            "detected": detected,
            "missed": support - detected,
            "recall": detected / support if support else 0.0,
        }
    return result


def per_category_metrics(
    data: pd.DataFrame,
    predictions: pd.Series,
) -> dict[str, dict[str, float | int]]:
    """Посчитать FPR на clean и Recall на атаках по категориям."""
    result: dict[str, dict[str, float | int]] = {}
    for category, group in data.groupby("category"):
        clean = group[group["label"] == 0]
        attacks = group[group["label"] == 1]
        false_positives = int((predictions.loc[clean.index] == 1).sum())
        detected = int((predictions.loc[attacks.index] == 1).sum())
        result[str(category)] = {
            "clean_support": len(clean),
            "false_positives": false_positives,
            "false_positive_rate": (
                false_positives / len(clean) if len(clean) else 0.0
            ),
            "attack_support": len(attacks),
            "detected": detected,
            "recall": detected / len(attacks) if len(attacks) else 0.0,
        }
    return result


def print_report(
    name: str,
    metrics: dict[str, Any],
    attack_metrics: dict[str, dict[str, float | int]],
    category_metrics: dict[str, dict[str, float | int]],
    threshold: float | None = None,
) -> None:
    """Вывести матрицу ошибок, общие метрики и разбивку по типам/категориям."""
    logger.info("=" * 88)
    logger.info("Детектор: %s", name)
    if threshold is not None:
        logger.info("Порог (выбран на validation): %.4f", threshold)
    logger.info("Матрица ошибок:")
    logger.info("                      предсказано 0   предсказано 1")
    logger.info(
        "  фактически 0 (clean)      %6d          %6d",
        metrics["true_negatives"],
        metrics["false_positives"],
    )
    logger.info(
        "  фактически 1 (attack)     %6d          %6d",
        metrics["false_negatives"],
        metrics["true_positives"],
    )
    logger.info(
        "Чистых текстов ошибочно заблокировано (FP): %s",
        metrics["false_positives"],
    )
    logger.info("Атак пропущено (FN): %s", metrics["false_negatives"])
    logger.info(
        "FPR на чистых: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
        metrics["false_positive_rate"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )

    logger.info("По типам атак:")
    logger.info("  %-20s %8s %8s %8s %8s", "тип", "support", "detected", "missed", "recall")
    for attack_type, values in sorted(attack_metrics.items()):
        logger.info(
            "  %-20s %8d %8d %8d %8.4f",
            attack_type,
            int(values["support"]),
            int(values["detected"]),
            int(values["missed"]),
            float(values["recall"]),
        )

    logger.info("По категориям текстов:")
    logger.info(
        "  %-20s %7s %7s %7s %7s %8s",
        "категория",
        "clean",
        "FP",
        "FPR",
        "attacks",
        "recall",
    )
    for category, values in sorted(category_metrics.items()):
        logger.info(
            "  %-20s %7d %7d %7.4f %7d %8.4f",
            category,
            int(values["clean_support"]),
            int(values["false_positives"]),
            float(values["false_positive_rate"]),
            int(values["attack_support"]),
            float(values["recall"]),
        )


def evaluate_adversarial(
    test_data: pd.DataFrame,
    model_path: Path,
) -> tuple[
    dict[str, Any],
    CategoryMetrics,
    CategoryMetrics,
    float,
    list[int],
    list[float],
]:
    """Оценить adversarial-детектор и вернуть scores для PR-кривой."""
    artifacts = load_artifacts(model_path)
    model = artifacts["model"]
    threshold = float(artifacts.get("threshold", 0.5))
    probabilities = model.predict_proba(test_data["text"].tolist())[:, 1]
    predictions = pd.Series(
        (probabilities >= threshold).astype(int),
        index=test_data.index,
    )
    y_true = [int(value) for value in test_data["label"].tolist()]
    metrics = binary_metrics(y_true, predictions.tolist(), probabilities.tolist())
    metrics["threshold"] = threshold
    attack_metrics = per_attack_type_metrics(test_data, predictions)
    category_metrics = per_category_metrics(test_data, predictions)
    return (
        metrics,
        attack_metrics,
        category_metrics,
        threshold,
        y_true,
        probabilities.tolist(),
    )


def evaluate_anomaly(
    test_data: pd.DataFrame,
    model_path: Path,
    anomaly_texts_path: Path,
) -> tuple[
    dict[str, Any],
    CategoryMetrics,
    CategoryMetrics,
    list[int],
    list[float],
    float,
    int,
    int,
]:
    """Оценить anomaly-детектор и вернуть scores для PR-кривой."""
    artifacts = load_artifacts(model_path)
    model = artifacts["model"]

    test_features = [extract_features(text) for text in test_data["text"].tolist()]
    raw_predictions = model.predict(test_features)
    predictions = pd.Series(
        (raw_predictions == -1).astype(int),
        index=test_data.index,
    )
    scores = (-model.decision_function(test_features)).tolist()
    y_true = [int(value) for value in test_data["label"].tolist()]
    metrics = binary_metrics(y_true, predictions.tolist(), scores)
    metrics["contamination"] = artifacts.get("contamination")
    attack_metrics = per_attack_type_metrics(test_data, predictions)
    category_metrics = per_category_metrics(test_data, predictions)

    clean_test = test_data[test_data["label"] == 0]
    clean_fpr = (
        float(predictions.loc[clean_test.index].mean()) if len(clean_test) else 0.0
    )

    if not anomaly_texts_path.exists():
        raise FileNotFoundError(
            f"Не найден файл {anomaly_texts_path}. Сначала запусти make data."
        )
    anomaly_texts = anomaly_texts_path.read_text(encoding="utf-8").splitlines()
    anomaly_features = [extract_features(text) for text in anomaly_texts]
    anomaly_predictions = model.predict(anomaly_features)
    detected = int((anomaly_predictions == -1).sum())
    anomaly_total = len(anomaly_texts)

    return (
        metrics,
        attack_metrics,
        category_metrics,
        y_true,
        scores,
        clean_fpr,
        detected,
        anomaly_total,
    )


def plot_pr_curves(
    adversarial_y: list[int],
    adversarial_scores: list[float],
    adversarial_threshold: float,
    anomaly_y: list[int],
    anomaly_scores: list[float],
) -> tuple[float, float]:
    """Построить и сохранить PR-кривые обоих детекторов."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    adv_precision, adv_recall, _ = precision_recall_curve(
        adversarial_y,
        adversarial_scores,
    )
    adv_ap = float(average_precision_score(adversarial_y, adversarial_scores))
    axes[0].plot(adv_recall, adv_precision, label=f"AP = {adv_ap:.4f}")
    adv_pred = np.asarray(adversarial_scores) >= adversarial_threshold
    point_precision, point_recall, _, _ = precision_recall_fscore_support(
        adversarial_y,
        adv_pred.astype(int),
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    axes[0].scatter(
        [point_recall],
        [point_precision],
        color="red",
        label=f"порог {adversarial_threshold:.3f}",
    )
    axes[0].set_title("Adversarial: Precision-Recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    anomaly_precision, anomaly_recall, _ = precision_recall_curve(
        anomaly_y,
        anomaly_scores,
    )
    anomaly_ap = float(average_precision_score(anomaly_y, anomaly_scores))
    axes[1].plot(anomaly_recall, anomaly_precision, label=f"AP = {anomaly_ap:.4f}")
    axes[1].set_title("Anomaly: Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    PR_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PR_CURVE_PATH, dpi=150)
    plt.close(figure)
    return adv_ap, anomaly_ap


def main() -> None:
    """Загрузить детекторы, проверить их на grouped test и вывести отчёт."""
    settings = get_settings()
    setup_logging(settings.log_level)

    csv_path = resolve_path(settings.train_csv)
    adversarial_path = resolve_path(settings.adversarial_model_path)
    anomaly_path = resolve_path(settings.anomaly_model_path)
    anomaly_texts_path = ROOT / "data" / "anomaly_texts.txt"

    try:
        test_data = load_test_data(csv_path)
        logger.info(
            "Тестовая выборка: %s текстов (clean=%s, attacks=%s), групп=%s",
            len(test_data),
            int((test_data["label"] == 0).sum()),
            int((test_data["label"] == 1).sum()),
            test_data["base_id"].nunique(),
        )

        (
            adversarial_metrics,
            adversarial_attack_metrics,
            adversarial_category_metrics,
            adversarial_threshold,
            adversarial_y,
            adversarial_scores,
        ) = evaluate_adversarial(test_data, adversarial_path)
        print_report(
            "adversarial",
            adversarial_metrics,
            adversarial_attack_metrics,
            adversarial_category_metrics,
            threshold=adversarial_threshold,
        )

        (
            anomaly_metrics,
            anomaly_attack_metrics,
            anomaly_category_metrics,
            anomaly_y,
            anomaly_scores,
            clean_fpr,
            detected_anomalies,
            anomaly_total,
        ) = evaluate_anomaly(
            test_data,
            anomaly_path,
            anomaly_texts_path,
        )
        print_report(
            "anomaly",
            anomaly_metrics,
            anomaly_attack_metrics,
            anomaly_category_metrics,
        )
        logger.info("Anomaly FPR на чистых тестовых текстах: %.4f", clean_fpr)
        logger.info(
            "Anomaly обнаружил %s из %s реальных аномалий (%.1f%%)",
            detected_anomalies,
            anomaly_total,
            (detected_anomalies / anomaly_total * 100) if anomaly_total else 0.0,
        )

        adv_ap, anomaly_ap = plot_pr_curves(
            adversarial_y,
            adversarial_scores,
            adversarial_threshold,
            anomaly_y,
            anomaly_scores,
        )
        logger.info(
            "PR-кривые сохранены: %s (adversarial AP=%.4f, anomaly AP=%.4f)",
            PR_CURVE_PATH,
            adv_ap,
            anomaly_ap,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info("=" * 88)
    logger.info("Оценка завершена")


if __name__ == "__main__":
    main()
