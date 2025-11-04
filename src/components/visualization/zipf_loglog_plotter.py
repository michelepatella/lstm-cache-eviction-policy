"""zipf_loglog_plotter.py

Utility module for visualizing the distribution of key accesses using
a Zipfian log-log plot.

This module provides the `plot_zipf_loglog` function, which calculates
the frequency of each unique key access, ranks them, and plots the
rank vs. frequency on a log-log scale to show the characteristic
Zipfian distribution pattern.

Functions:
    plot_zipf_loglog(
        requests: list[int],
        save_path: str
    ) -> None
        Plots the key access frequency distribution in a log-log scale
        and saves the resulting figure.
"""

from collections import Counter

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


def plot_zipf_loglog(requests: list[int], save_path: str) -> None:
    """Plot Zipfian distribution via log-log.

    This function plots Zipfian distribution of key accesses via log-log.
    The generated plot should highlight much more key accesses to the first
    keys than those to later ones.

    Args:
        requests (list[int]): List of key accesses.
        save_path (str): Path to save the figure.

    Returns:
        None


    Raises:
        RuntimeError: If plotting Zipf log-log fails:
            * Non-iterable or invalid requests input (TypeError).
            * Invalid values in requests or internal array creation (ValueError).
    """
    try:
        # Count how many keys are
        # involved in generated requests
        key_counts = Counter(requests)

        # Get key access frequencies
        key_frequencies = np.array(sorted(key_counts.values(), reverse=True))

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

        debug(
            "Zipf log-log plotted and saved",
            extra={
                "save_path": str(save_path),
                "requests_num": len(requests),
                "keys_num": len(key_counts),
                "ranks_range": (1, len(ranks)),
                "frequencies_range": (
                    int(key_frequencies.min()),
                    int(key_frequencies.max()),
                ),
                "context": "Zipf log-log",
            },
        )
    except (ValueError, TypeError) as e:
        msg = "Plotting Zipf log-log failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "save_path": str(save_path),
                "requests_num": (
                    len(requests) if isinstance(requests, list) else None
                ),
                "context": "Zipf log-log",
            },
        )
        raise RuntimeError(msg) from e
