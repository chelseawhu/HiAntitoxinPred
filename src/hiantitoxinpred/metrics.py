"""Benchmark metrics from raw logits and independent-run summaries."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def binary_metrics(labels, raw_logits):
    y, z = np.asarray(labels), np.asarray(raw_logits, dtype=np.float64)
    if (
        y.ndim != 1
        or z.ndim != 1
        or len(y) != len(z)
        or not len(y)
        or not np.isfinite(z).all()
        or set(y.tolist()) != {0, 1}
    ):
        raise ValueError(
            "Expected finite one-dimensional scores and both binary classes"
        )
    prediction = z >= 0
    tn, fp, fn, tp = map(int, confusion_matrix(y, prediction, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return dict(
        precision=precision,
        recall=recall,
        F1=f1,
        MCC=float(matthews_corrcoef(y, prediction)),
        accuracy=(tp + tn) / len(y),
        AUROC=float(roc_auc_score(y, z)),
        AP=float(average_precision_score(y, z)),
        TP=tp,
        FP=fp,
        TN=tn,
        FN=fn,
    )


def summarize_runs(rows):
    """Mean and sample SD of already-computed per-run metrics.

    Caller must supply one homogeneous dataset/configuration/checkpoint rule.
    This function never aggregates probabilities across models.
    """
    if len(rows) < 2 or not rows[0]:
        raise ValueError("At least two complete runs are needed for sample SD")
    keys = tuple(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise ValueError("Runs have different metric fields")
    values = np.asarray([[row[k] for k in keys] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Run metrics must be finite")
    return dict(
        n=len(rows),
        mean=dict(zip(keys, values.mean(axis=0).tolist())),
        sd=dict(zip(keys, values.std(axis=0, ddof=1).tolist())),
    )
