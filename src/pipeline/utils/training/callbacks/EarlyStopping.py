import numpy as np

from config.classes.Config import Config
from const import LOGS_VALIDATION_PHASE
from utils.logs.initializer import logs_phase
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


class EarlyStopping:
    """
    Early stopping handler for model training.

    Monitors the average loss and stops training if no improvement
    is observed for a specified number of epochs.
    """

    def __init__(self: "EarlyStopping", config: Config) -> None:
        """
        Initialize EarlyStopping with configuration.

        This function initializes EarlyStopping, setting
        patience and delta, as well as initializing the best
        average loss, counter, and early stropping triggering flag.

        Args:
            self ("EarlyStopping"): Current class instance.
            config (Config): Configuration object.

        Returns:
            None
        """
        # Read the current pipeline phase
        current_phase = logs_phase.get()

        debug(f"Current pipeline phase: {current_phase}")

        # Set patience (how many epochs to wait before stopping)
        self.patience = (
            config.validation.early_stopping.patience
            if current_phase == LOGS_VALIDATION_PHASE
            else config.training.early_stopping.patience
        )

        # Set delta (minimum improvement to reset counter)
        self.delta = (
            config.validation.early_stopping.delta
            if current_phase == LOGS_VALIDATION_PHASE
            else config.training.early_stopping.delta
        )

        # Initialize best average loss
        self.best_avg_loss = np.inf

        # Initialize counter for epochs
        # without improvement
        self.counter = 0

        # Flag indicating whether early stopping
        # should be triggered
        self.early_stop = False

        info("EarlyStopping initialized")

        debug(
            "EarlyStopping initial settings:\n"
            + f"Patience: {self.patience}\n"
            + f"Delta: {self.delta}\n"
            + f"Best average loss: {self.best_avg_loss}\n"
            + f"Counter: {self.counter}\n"
            + f"Early stopping: {self.early_stop}"
        )

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
        """
        debug(f"Current best average loss: {self.best_avg_loss}")
        debug(f"Average loss for current check: {avg_loss}")

        # Check for improvement beyond delta tolerance
        if avg_loss < self.best_avg_loss - self.delta:
            # Improvement detected: update the best
            # average loss and reset counter
            self.best_avg_loss = avg_loss
            self.counter = 0

            debug(f"New best average loss: {self.best_avg_loss}")
            debug(f"Early stopping counter reset to: {self.counter}")
        else:
            # No significant improvement:
            # increment counter
            self.counter += 1

            debug("No improvement in average loss")
            debug(f"Early stopping counter incremented to: {self.counter}")

            # Trigger early stopping if patience exceeded
            if self.counter >= self.patience:
                self.early_stop = True

                debug("Patience exceeded: early stopping triggered")

        info("Early stopping check completed")
