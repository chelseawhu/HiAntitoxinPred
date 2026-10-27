"""Post-hoc region metrics; annotations never enter MAIA inference."""

from itertools import combinations
import numpy as np
from sklearn.metrics import average_precision_score


def top_weights(scores, budget):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected finite residue scores")
    if not np.isfinite(budget) or not 0 < budget <= len(scores):
        raise ValueError("Invalid Top budget")
    boundary = np.sort(scores)[::-1][int(np.ceil(budget)) - 1]
    weights = (scores > boundary).astype(float)
    tied = scores == boundary
    weights[tied] = (budget - weights.sum()) / tied.sum()
    return weights


def region_metrics(scores, mask):
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask)
    if (
        mask.shape != scores.shape
        or not np.isin(mask, [0, 1]).all()
        or not 0 < mask.sum() < len(mask)
    ):
        raise ValueError("Expected a nontrivial binary region mask")
    weights = top_weights(scores, float(mask.sum()))
    return dict(
        region_AP=float(average_precision_score(mask, scores)),
        top_D_overlap=float(weights[mask.astype(bool)].sum() / mask.sum()),
    )


def stability(profiles, fraction=0.2):
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2 or len(profiles) < 2:
        raise ValueError("At least two model profiles required")
    weights = [top_weights(row, fraction * profiles.shape[1]) for row in profiles]
    pairs = [
        dict(
            first=i,
            second=j,
            jaccard=float(
                np.minimum(weights[i], weights[j]).sum()
                / np.maximum(weights[i], weights[j]).sum()
            ),
        )
        for i, j in combinations(range(len(weights)), 2)
    ]
    values = [p["jaccard"] for p in pairs]
    return dict(
        mean=float(np.mean(values)),
        descriptive_pair_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        pairs=pairs,
    )
