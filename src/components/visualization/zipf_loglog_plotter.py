from collections import Counter
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from components.const import (
    PLOT_LABEL_FONT_SIZE,
    PLOT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    PLOT_ZIPF_LOG_LOG_MARKER,
    PLOT_ZIPF_LOG_LOG_TITLE,
    PLOT_ZIPF_LOG_LOG_X_LABEL,
    PLOT_ZIPF_LOG_LOG_Y_LABEL,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def plot_zipf_loglog(requests: List[int], save_path: str) -> None:
    """
    Plot Zipfian distribution via log-log.

    This function plots Zipfian distribution of key accesses via log-log.
    The generated plot should highlight much more key accesses to the first
    keys than those to later ones.

    Args:
        requests (List[int]): List of key accesses.
        save_path (str): Path to save the figure.

    Returns:
        None


    Raises:
        RuntimeError: If plotting Zipf log-log fails:
            * Non-iterable or invalid requests input (TypeError).
            * Invalid values in requests or internal array creation (ValueError).
    """
    try:
        debug(f"Number of requests for Zipf log-log plot: {len(requests)}")

        # Count how many keys are
        # involved in generated requests
        key_counts = Counter(requests)
        debug(f"Number of keys for Zipf log-log plot: {key_counts}")

        # Get key access frequencies
        key_frequencies = np.array(sorted(key_counts.values(), reverse=True))
        debug(f"Key frequencies for Zipf log-log plot: {key_frequencies}")

        # Define ranks ranging from
        # the first place to the last one
        ranks = np.arange(1, len(key_frequencies) + 1)

        # Prepare, show, and save the plot
        plt.figure(figsize=(PLOT_SIZE, PLOT_SIZE))
        plt.loglog(
            ranks,
            key_frequencies,
            marker=PLOT_ZIPF_LOG_LOG_MARKER,
        )
        plt.title(
            PLOT_ZIPF_LOG_LOG_TITLE,
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        plt.xlabel(
            PLOT_ZIPF_LOG_LOG_X_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.ylabel(
            PLOT_ZIPF_LOG_LOG_Y_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

        info(f"Zipf log-log plotted and saved to: {save_path}")
    except (ValueError, TypeError) as e:
        msg = "Failed to plot Zipf log-log"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
