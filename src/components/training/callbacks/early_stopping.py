import numpy as np

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import EARLY_STOPPING_DISABLED, EARLY_STOPPING_ENABLED


class EarlyStopping:
    """
    Early stopping callback for model training and validation.

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
        """
        Initialize EarlyStopping.

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
        self.early_stop = EARLY_STOPPING_DISABLED

        debug(
            "EarlyStopping initial settings:\n"
            + f"Patience: {self.patience}\n"
            + f"Delta: {self.delta}\n"
            + f"Best average loss: {self.best_avg_loss}\n"
            + f"Counter: {self.counter}\n"
            + f"Early stopping: {self.early_stop}"
        )
        info("EarlyStopping initialized")

    def __call__(self: "EarlyStopping", avg_loss: float) -> None:
        """
        Update early stopping status based on the current average loss.

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
            debug(
                f"Current best average loss for early stopping check: {self.best_avg_loss}"
            )
            debug(f"Average loss for current early stopping check: {avg_loss}")
            debug(f"Counter before early stopping check: {self.counter}")

            # Check for improvement beyond delta tolerance
            if avg_loss < self.best_avg_loss - self.delta:
                # Improvement detected: update the best
                # average loss and reset counter
                self.best_avg_loss = avg_loss
                self.counter = 0

                info(
                    f"Early stopping check completed (New best average "
                    f"loss: {self.best_avg_loss}, counter set to: {self.counter})"
                )
            else:
                # No significant improvement:
                # increment counter
                self.counter += 1

                # Trigger early stopping if patience exceeded
                if self.counter >= self.patience:
                    self.early_stop = EARLY_STOPPING_ENABLED

                    info(
                        f"Early stopping check completed (early stopping triggered, "
                        f"counter ({self.counter}) >= patience({self.patience}))"
                    )
                else:
                    info(
                        f"Early stopping check completed (early stopping not triggered,"
                        f"counter ({self.counter}) < patience({self.patience}))"
                    )
        except TypeError as e:
            msg = "Failed to check early stopping"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
