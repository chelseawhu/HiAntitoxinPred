"""Loss-only checkpoint choice, independent of threshold or ranking metrics."""

import math


class LossSelection:
    def __init__(self, patience=30, min_delta=1e-5):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        self.patience, self.min_delta = patience, min_delta
        self.reference_loss = math.inf
        self.best_loss, self.best_epoch = math.inf, 0
        self.wait, self.last_epoch = 0, 0

    def update(self, epoch, loss):
        """Return whether the true-minimum-loss checkpoint must be saved.

        A decrease smaller than min_delta can save L while still incrementing
        the early-stop wait. Exactly equal loss retains the earlier checkpoint.
        """
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch != self.last_epoch + 1
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError("Invalid validation loss trajectory")
        if self.should_stop:
            raise ValueError("Training continued beyond the early-stop point")
        self.last_epoch = epoch
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss, self.wait = loss, 0
        else:
            self.wait += 1
        selected = loss < self.best_loss
        if selected:
            self.best_loss, self.best_epoch = loss, epoch
        return selected

    @property
    def should_stop(self):
        return self.wait >= self.patience
