from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def select_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_values)) if f1_values.size else 0
    return float(thresholds[best_idx])


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    score_kind: str = "prob",
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    preds = (scores >= float(threshold)).astype(np.int32)

    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    auc_roc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0
    auc_pr = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0

    payload: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "mcc": float(matthews_corrcoef(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0,
        "specificity": specificity,
        "threshold_score": float(threshold),
        "threshold_kind": score_kind,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if score_kind == "logit":
        payload["threshold_logit"] = float(threshold)
        payload["threshold_prob"] = float(sigmoid(np.asarray([threshold]))[0])
    else:
        payload["threshold_prob"] = float(threshold)
    return payload


def compute_metrics_with_selected_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    score_kind: str = "prob",
) -> Dict[str, Any]:
    threshold = select_threshold_by_f1(y_true, scores)
    metrics = compute_binary_classification_metrics(y_true, scores, threshold, score_kind=score_kind)
    metrics["threshold_source"] = "auto"
    return metrics


def apply_fixed_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    score_kind: str = "prob",
) -> Dict[str, Any]:
    metrics = compute_binary_classification_metrics(y_true, scores, threshold, score_kind=score_kind)
    metrics["threshold_source"] = "fixed"
    return metrics
