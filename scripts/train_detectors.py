"""Обучение детекторов SPL с групповым split, hard negatives и LightGBM."""

import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from src.infrastructure.detectors.adversarial import (  # noqa: E402
    build_adversarial_model,
)
from src.infrastructure.detectors.anomaly import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SHARE_OF_TRAIN_VAL = 0.25
TARGET_FPR = 0.01
TARGET_ANOMALY_FPR = 0.01
TARGET_AUC = 0.90
CONTAMINATION_CANDIDATES = (0.005, 0.01, 0.02, 0.03, 0.05)
CV_FOLDS = 5
ANOMALY_TREES = 150


def resolve_path(value: str) -> Path:
    """Превратить путь из настроек в абсолютный относительно корня проекта."""
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_training_data(csv_path: Path) -> pd.DataFrame:
    """Прочитать CSV с base_id, category и типом атаки."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Не найден файл {csv_path}. "
            "Сначала запусти python data/generate_data.py или make data."
        )

    data = pd.read_csv(csv_path)
    required_columns = {"text", "label", "base_id", "category"}
    if not required_columns.issubset(set(data.columns)):
        raise ValueError(
            f"В {csv_path} должны быть колонки {sorted(required_columns)}. "
            "Перегенерируй данные: make data."
        )
    if data.empty:
        raise ValueError(f"Файл {csv_path} не содержит данных")

    if data["text"].isna().any():
        raise ValueError("В колонке text есть пустые значения")
    if data["base_id"].isna().any():
        raise ValueError("В колонке base_id есть пустые значения")

    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(int)
    data["base_id"] = data["base_id"].astype(str)
    data["category"] = data["category"].fillna("unknown").astype(str)
    if "attack_type" not in data.columns:
        data["attack_type"] = ""
    else:
        data["attack_type"] = data["attack_type"].fillna("").astype(str)

    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("В обучающем наборе должны быть классы 0 и 1")
    return data


def split_dataset(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить данные на train/val/test, не разрывая группы base_id."""
    groups = data["base_id"]
    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_val_positions, test_positions = next(
        test_splitter.split(data, groups=groups)
    )
    train_val_data = data.iloc[train_val_positions].copy()
    test_data = data.iloc[test_positions].copy()

    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SHARE_OF_TRAIN_VAL,
        random_state=RANDOM_STATE,
    )
    train_positions, val_positions = next(
        val_splitter.split(train_val_data, groups=train_val_data["base_id"])
    )
    train_data = train_val_data.iloc[train_positions].copy()
    val_data = train_val_data.iloc[val_positions].copy()

    train_groups = set(train_data["base_id"])
    val_groups = set(val_data["base_id"])
    test_groups = set(test_data["base_id"])
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("Обнаружена утечка base_id между train/val/test")

    for name, part in (
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ):
        if set(part["label"].unique()) != {0, 1}:
            raise ValueError(f"В {name} должны присутствовать оба класса")
    return train_data, val_data, test_data


def binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float],
) -> dict[str, Any]:
    """Посчитать Precision, Recall, F1, ROC-AUC и матрицу ошибок."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def select_threshold(
    y_true: list[int],
    scores: list[float],
    target_fpr: float = TARGET_FPR,
) -> dict[str, float]:
    """Выбрать порог на validation при ограничении FPR."""
    y = np.asarray(y_true)
    score_array = np.asarray(scores)
    negative = y == 0
    positive = y == 1
    negative_count = int(negative.sum())
    positive_count = int(positive.sum())

    candidates = sorted(
        {0.0, 1.01, *(float(value) for value in np.unique(score_array))}
    )
    best_threshold = 1.01
    best_recall = -1.0
    best_precision = -1.0
    best_fpr = 1.0
    best_fp = 0
    best_tp = 0

    for threshold in candidates:
        predicted = score_array >= threshold
        false_positives = int((predicted & negative).sum())
        true_positives = int((predicted & positive).sum())
        fpr = false_positives / negative_count if negative_count else 0.0
        if fpr > target_fpr:
            continue
        recall = true_positives / positive_count if positive_count else 0.0
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives)
            else 0.0
        )
        if (recall, precision, threshold) > (
            best_recall,
            best_precision,
            best_threshold,
        ):
            best_threshold = threshold
            best_recall = recall
            best_precision = precision
            best_fpr = fpr
            best_fp = false_positives
            best_tp = true_positives

    return {
        "threshold": float(best_threshold),
        "false_positive_rate": float(best_fpr),
        "recall": float(best_recall if best_recall >= 0 else 0.0),
        "precision": float(best_precision if best_precision >= 0 else 0.0),
        "false_positives": float(best_fp),
        "true_positives": float(best_tp),
    }


def per_attack_type_metrics(
    data: pd.DataFrame,
    predictions: pd.Series,
) -> dict[str, dict[str, float | int]]:
    """Посчитать полноту и поддержку по каждому типу искажения."""
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
    """Посчитать FPR на чистых текстах и Recall на атаках по категориям."""
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


def log_confusion_matrix(name: str, metrics: dict[str, Any]) -> None:
    """Напечатать матрицу ошибок в читаемом виде."""
    logger.info("Матрица ошибок (%s):", name)
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


def train_adversarial_detector(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_path: Path,
) -> dict[str, Any]:
    """Обучить LightGBM и выбрать порог только на validation."""
    x_train = train_data["text"].tolist()
    y_train = train_data["label"].tolist()
    y_val = val_data["label"].tolist()
    y_test = test_data["label"].tolist()

    logger.info("Обучение adversarial-детектора на %s текстах...", len(x_train))
    model = build_adversarial_model()
    model.fit(x_train, y_train)
    model.named_steps["classifier"].set_params(n_jobs=1)

    val_scores = model.predict_proba(val_data["text"].tolist())[:, 1].tolist()
    validation = select_threshold(y_val, val_scores, TARGET_FPR)
    threshold = float(validation["threshold"])

    test_scores = model.predict_proba(test_data["text"].tolist())[:, 1].tolist()
    predictions = pd.Series(
        (np.asarray(test_scores) >= threshold).astype(int),
        index=test_data.index,
    )
    metrics = binary_metrics(y_test, predictions.tolist(), test_scores)
    metrics.update(
        {
            "threshold": threshold,
            "validation": validation,
            "per_attack_type": per_attack_type_metrics(test_data, predictions),
            "per_category": per_category_metrics(test_data, predictions),
        }
    )

    logger.info(
        "Порог adversarial: %.4f (val FPR=%.4f, val Recall=%.4f)",
        threshold,
        validation["false_positive_rate"],
        validation["recall"],
    )
    logger.info(
        "Adversarial test: Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    log_confusion_matrix("adversarial", metrics)
    logger.info("Adversarial по типам атак: %s", metrics["per_attack_type"])
    logger.info("Adversarial по категориям: %s", metrics["per_category"])
    if metrics["false_positive_rate"] > TARGET_FPR:
        logger.warning(
            "FPR на тесте выше цели %.3f: %.4f",
            TARGET_FPR,
            metrics["false_positive_rate"],
        )
    if metrics["roc_auc"] < TARGET_AUC:
        logger.warning(
            "ROC-AUC ниже цели: %.4f < %.2f",
            metrics["roc_auc"],
            TARGET_AUC,
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "threshold": threshold,
            },
            file,
        )
    logger.info("Adversarial-детектор сохранён: %s", model_path)
    return metrics


def tune_anomaly_contamination(
    clean_texts: list[str],
    candidates: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    """Подобрать contamination через KFold на чистых текстах."""
    features = [extract_features(text) for text in clean_texts]
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(folds.split(features))
    scores: dict[str, float] = {}

    for contamination in candidates:
        fold_fprs: list[float] = []
        for train_index, validation_index in fold_indices:
            train_features = [features[index] for index in train_index]
            validation_features = [features[index] for index in validation_index]
            model = IsolationForest(
                contamination=contamination,
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(train_features)
            predictions = model.predict(validation_features)
            fold_fprs.append(float((predictions == -1).mean()))
        scores[f"{contamination:.3f}"] = sum(fold_fprs) / len(fold_fprs)
        logger.info(
            "contamination=%.3f: средний FPR на CV = %.4f",
            contamination,
            scores[f"{contamination:.3f}"],
        )

    acceptable = [
        float(value)
        for value in scores
        if scores[value] <= TARGET_ANOMALY_FPR
    ]
    if acceptable:
        chosen = max(acceptable)
        logger.info(
            "Выбран contamination=%.3f (максимальная чувствительность при FPR <= %.3f)",
            chosen,
            TARGET_ANOMALY_FPR,
        )
    else:
        chosen = min(candidates)
        logger.warning(
            "Ни один вариант не уложился в FPR <= %.3f, выбран минимальный contamination=%.3f",
            TARGET_ANOMALY_FPR,
            chosen,
        )
    return chosen, scores


def calibrate_anomaly_offset(
    model: IsolationForest,
    val_clean_texts: list[str],
) -> dict[str, float]:
    """При необходимости ужесточить offset модели по validation FPR."""
    if not val_clean_texts:
        return {"validation_fpr_before": 0.0, "validation_fpr_after": 0.0}

    val_features = [extract_features(text) for text in val_clean_texts]
    before_predictions = model.predict(val_features)
    before_fpr = float((before_predictions == -1).mean())
    if before_fpr <= TARGET_ANOMALY_FPR:
        return {
            "validation_fpr_before": before_fpr,
            "validation_fpr_after": before_fpr,
            "offset_adjusted": 0.0,
        }

    decisions = model.decision_function(val_features)
    quantile = float(np.quantile(decisions, TARGET_ANOMALY_FPR))
    model.offset_ = float(model.offset_) + quantile
    after_predictions = model.predict(val_features)
    after_fpr = float((after_predictions == -1).mean())
    logger.info(
        "Anomaly offset откалиброван: FPR %.4f -> %.4f",
        before_fpr,
        after_fpr,
    )
    return {
        "validation_fpr_before": before_fpr,
        "validation_fpr_after": after_fpr,
        "offset_adjusted": quantile,
    }


def train_anomaly_detector(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    anomaly_texts_path: Path,
    model_path: Path,
    candidates: tuple[float, ...],
) -> dict[str, Any]:
    """Обучить IsolationForest на чистых текстах и оценить его."""
    clean_train = train_data[train_data["label"] == 0]["text"].tolist()
    if not clean_train:
        raise ValueError("В обучающем наборе нет чистых текстов (label=0)")

    contamination, cv_fprs = tune_anomaly_contamination(clean_train, candidates)
    train_features = [extract_features(text) for text in clean_train]
    model = IsolationForest(
        contamination=contamination,
        n_estimators=ANOMALY_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_features)
    model.set_params(n_jobs=1)

    val_clean = val_data[val_data["label"] == 0]["text"].tolist()
    calibration = calibrate_anomaly_offset(model, val_clean)

    test_features = [extract_features(text) for text in test_data["text"].tolist()]
    raw_predictions = model.predict(test_features)
    predictions = pd.Series(
        (raw_predictions == -1).astype(int),
        index=test_data.index,
    )
    scores = (-model.decision_function(test_features)).tolist()
    metrics = binary_metrics(
        test_data["label"].tolist(),
        predictions.tolist(),
        scores,
    )
    metrics.update(
        {
            "per_attack_type": per_attack_type_metrics(test_data, predictions),
            "per_category": per_category_metrics(test_data, predictions),
        }
    )

    clean_test = test_data[test_data["label"] == 0]
    test_clean_fpr = (
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
    anomaly_recall = detected / len(anomaly_texts) if anomaly_texts else 0.0

    metrics.update(
        {
            "contamination": contamination,
            "cv_fpr": cv_fprs,
            "validation": calibration,
            "test_fpr_on_clean": test_clean_fpr,
            "anomaly_texts_total": len(anomaly_texts),
            "anomaly_texts_detected": detected,
            "anomaly_texts_recall": anomaly_recall,
            "feature_names": list(FEATURE_NAMES),
        }
    )

    logger.info(
        "Anomaly test: Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    log_confusion_matrix("anomaly", metrics)
    logger.info("Anomaly FPR на чистых тестовых текстах: %.4f", test_clean_fpr)
    logger.info(
        "Anomaly обнаружил %s из %s реальных аномалий (%.1f%%)",
        detected,
        len(anomaly_texts),
        anomaly_recall * 100,
    )
    if test_clean_fpr > TARGET_FPR:
        logger.warning(
            "FPR anomaly-детектора выше %.1f%%: %.4f",
            TARGET_FPR * 100,
            test_clean_fpr,
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "contamination": contamination,
                "feature_names": list(FEATURE_NAMES),
            },
            file,
        )
    logger.info("Anomaly-детектор сохранён: %s", model_path)
    return metrics


def write_metrics(
    metrics: dict[str, Any],
    metrics_path: Path,
) -> None:
    """Сохранить метрики в JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info("Метрики сохранены: %s", metrics_path)


def main() -> None:
    """Обучить оба детектора и сохранить честные метрики."""
    settings = get_settings()
    setup_logging(settings.log_level)

    csv_path = resolve_path(settings.train_csv)
    adversarial_path = resolve_path(settings.adversarial_model_path)
    anomaly_path = resolve_path(settings.anomaly_model_path)
    anomaly_texts_path = ROOT / "data" / "anomaly_texts.txt"
    metrics_path = ROOT / "models" / "metrics.json"

    try:
        data = load_training_data(csv_path)
        train_data, val_data, test_data = split_dataset(data)
        logger.info(
            "Данные: всего=%s, train=%s, val=%s, test=%s; base-групп=%s",
            len(data),
            len(train_data),
            len(val_data),
            len(test_data),
            data["base_id"].nunique(),
        )

        contamination_candidates = tuple(
            sorted(
                {
                    *CONTAMINATION_CANDIDATES,
                    settings.anomaly_contamination,
                }
            )
        )
        adversarial_metrics = train_adversarial_detector(
            train_data,
            val_data,
            test_data,
            adversarial_path,
        )
        anomaly_metrics = train_anomaly_detector(
            train_data,
            val_data,
            test_data,
            anomaly_texts_path,
            anomaly_path,
            contamination_candidates,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total": len(data),
            "clean": int((data["label"] == 0).sum()),
            "adversarial": int((data["label"] == 1).sum()),
            "hard_negatives": int((data["category"] == "hard_negative").sum()),
            "base_groups": int(data["base_id"].nunique()),
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
            "categories": {
                str(key): int(value)
                for key, value in data.groupby("category").size().items()
            },
            "attack_types": {
                str(key): int(value)
                for key, value in data[data["label"] == 1]
                .groupby("attack_type")
                .size()
                .items()
            },
        },
        "adversarial": adversarial_metrics,
        "anomaly": anomaly_metrics,
    }
    write_metrics(metrics, metrics_path)
    logger.info("Обучение и оценка детекторов завершены")


if __name__ == "__main__":
    main()
