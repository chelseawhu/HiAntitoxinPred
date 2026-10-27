"""Public prediction geometry: fixed cap200 padding, independent of neighbours."""

import numpy as np
import torch

from .data import collate_features


def predict_features(model, features, *, batch_size=16, padding_width=200):
    """Predict raw logits for ordered CPU FP32 residue feature tensors.

    Public prediction defaults to width200. An explicit width supports archived
    MAIA batch geometry. Main reference evaluation uses its separately recorded
    longest-in-batch geometry. This API does not encode sequences.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if not len(features):
        raise ValueError("Empty prediction input")
    if model.config.max_length != 200:
        raise ValueError("Public prediction requires cap200")
    if (
        isinstance(padding_width, bool)
        or not isinstance(padding_width, int)
        or not 2 <= padding_width <= 200
    ):
        raise ValueError("padding_width must be an integer from 2 to 200")
    device = next(model.parameters()).device
    was_training = model.training
    chunks = []
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                x, mask = collate_features(
                    features[start : start + batch_size], width=padding_width
                )
                z = model(x.to(device), mask.to(device))
                if not torch.isfinite(z).all():
                    raise ValueError("Non-finite prediction")
                chunks.append(z.cpu().numpy().copy())
    finally:
        model.train(was_training)
    return np.concatenate(chunks)
