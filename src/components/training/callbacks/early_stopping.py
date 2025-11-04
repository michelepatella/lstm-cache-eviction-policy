"""early_stopping.py

Utility module for early stopping during model training.

This module provides the `EarlyStopping` class, which monitors the average
loss during training and validation. If the loss does not improve beyond
a specified delta for a number of consecutive epochs equal to the patience,
training can be stopped early to prevent overfitting and save computation.

Classes:
    EarlyStopping(patience: int, delta: float):
        Implements an early stopping callback to track the best average loss
        and determine when to stop training early.
"""

import numpy as np

from components.const import (
    EARLY_STOPPING_TRIGGERED,
    EARLY_STOPPING_UNTRIGGERED,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


class EarlyStopping:
    """Early stopping callback for model training and validation.

    This class implements early stopping callback which monitors
    the average loss during training and validation. If no significant
    improvement is observed for a number of epochs equal to the patience
    value, training is stopped early to prevent overfitting or wasted
    computation.

    Attributes:
        patience (int): Number of epochs to wait without improvement before
                        triggering early stopping.
        delta (float): Minimum change in average loss to qualify as an
                       improvement.
        best_avg_loss (float): Best observed average loss so far.
        counter (int): Number of consecutive epochs without improvement.
        early_stop (bool): Flag indicating whether early stopping has been
                           triggered.
    """

    def __init__(self: "EarlyStopping", patience: int, delta: float) -> None:
        """Initialize EarlyStopping.

        This function initializes EarlyStopping, setting patience and delta,
        as well as initializing the best average loss, counter, and early
        stropping triggering flag.

        Args:
            self ("EarlyStopping"): Current class instance.
            patience (int): Number of epochs to wait for improvement before
                            triggering early stopping.
            delta (float): Minimum change in the monitored average loss to
                           qualify as an improvement.

        Returns:
            None
        """
        # Set patience (how many epochs to
        # wait before stopping)
        self.patience = patience

        # Set delta (minimum improvement
        # to reset counter)
        self.delta = delta

        # Initialize best average loss
        self.best_avg_loss = np.inf

        # Initialize counter for epochs
        # without improvement
        self.counter = 0

        # Flag indicating whether early stopping
        # should be triggered
        self.early_stop = EARLY_STOPPING_UNTRIGGERED

        debug(
            "EarlyStopping initialization executed",
            extra={
                "patience": self.patience,
                "delta": self.delta,
                "loss_avg_best": None
                if np.isinf(self.best_avg_loss)
                else float(self.best_avg_loss),
                "counter": self.counter,
                "early_stop_flag": self.early_stop,
                "context": "EarlyStopping",
            },
        )

    def __call__(self: "EarlyStopping", avg_loss: float) -> None:
        """Update early stopping status based on the current average loss.

        This function checks if the current average loss is an improvement
        over the best observed loss, considering a delta tolerance. It
        updates the internal counter and sets the early stopping trigger
        flag if patience is exceeded.

        Args:
            self ("EarlyStopping"): Current class instance.
            avg_loss (float): Current average loss for the epoch.

        Returns:
            None

        Raises:
            RuntimeError: If checking early stopping fails:
                * Comparison between average loss and best average loss fails
                  due to invalid types (TypeError).
                * Incrementing counter or setting early stopping flag fails
                  due to invalid types (TypeError).
        """
        try:
            # Check for improvement beyond delta tolerance
            if avg_loss < self.best_avg_loss - self.delta:
                # Improvement detected: update the best
                # average loss and reset counter
                self.best_avg_loss = avg_loss
                self.counter = 0
            else:
                # No significant improvement:
                # increment counter
                self.counter += 1

                # Trigger early stopping if patience exceeded
                if self.counter >= self.patience:
                    self.early_stop = EARLY_STOPPING_TRIGGERED

            debug(
                "EarlyStopping state updated",
                extra={
                    "loss_avg": None
                    if np.isinf(avg_loss) or np.isnan(avg_loss)
                    else float(avg_loss),
                    "loss_avg_best": self.best_avg_loss,
                    "delta": self.delta,
                    "counter": self.counter,
                    "patience": self.patience,
                    "early_stop_flag": self.early_stop,
                    "context": "EarlyStopping",
                },
            )
        except TypeError as e:
            msg = "Early stopping check failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "loss_avg": None
                    if np.isinf(avg_loss) or np.isnan(avg_loss)
                    else float(avg_loss),
                    "loss_avg_best": getattr(self, "best_avg_loss", None),
                    "delta": getattr(self, "delta", None),
                    "counter": getattr(self, "counter", None),
                    "patience": getattr(self, "patience", None),
                    "early_stop_flag": getattr(self, "early_stop", None),
                    "context": "EarlyStopping",
                },
            )
            raise RuntimeError(msg) from e
