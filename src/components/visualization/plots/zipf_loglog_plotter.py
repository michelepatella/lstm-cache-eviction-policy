from collections import Counter
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from pipeline.const import (
    PLOT_LABEL_FONT_SIZE,
    PLOT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    ZIPF_LOG_LOG_PLOT_MARKER,
    ZIPF_LOG_LOG_PLOT_TITLE,
    ZIPF_LOG_LOG_PLOT_X_LABEL,
    ZIPF_LOG_LOG_PLOT_Y_LABEL,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def plot_zipf_loglog(requests: List[int], save_path: str) -> None:
    """
    Plot Zipfian distribution via log-log.

    This function plots Zipfian distribution
    of key accesses via log-log. The generated
    plot should highlight much more key accesses to
    the first keys than those to later ones.

    Args:
        requests (List[int]): List of key accesses.
        save_path (str): Path to save the figure.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while plotting Zipf log-log, e.g.:
            * If the list of requests is empty or
              contains one or more negative values.
            * If the list of requests is not
              iterable or contains non-iterable items.
    """
    try:
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

        # Plot the Zipfian distribution
        # log-log
        plt.figure(figsize=(PLOT_SIZE, PLOT_SIZE))
        plt.loglog(
            ranks,
            key_frequencies,
            marker=ZIPF_LOG_LOG_PLOT_MARKER,
        )
        plt.title(
            ZIPF_LOG_LOG_PLOT_TITLE,
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        plt.xlabel(
            ZIPF_LOG_LOG_PLOT_X_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.ylabel(
            ZIPF_LOG_LOG_PLOT_Y_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

        info(f"Zipf log-log plotted and saved to {save_path}")
    except (ValueError, TypeError) as e:
        msg = "Failed to plot Zipf log-log"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
