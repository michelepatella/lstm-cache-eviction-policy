import numpy as np
from matplotlib import pyplot as plt

from const import (
    DAILY_PROFILE_PLOT_ALIGN,
    DAILY_PROFILE_PLOT_BIN_SIZE,
    DAILY_PROFILE_PLOT_EDGE_COLOR,
    DAILY_PROFILE_PLOT_STEP,
    DAILY_PROFILE_PLOT_TITLE,
    DAILY_PROFILE_PLOT_X_LABEL,
    DAILY_PROFILE_PLOT_Y_LABEL,
    FIGURE_LABEL_FONT_SIZE,
    FIGURE_SIZE,
    FIGURE_TITLE_FONT_SIZE,
    MAX_HOUR,
    MIN_HOUR,
)
from lstm_eviction_policy.utils.logs.levels.debug_logger import debug
from lstm_eviction_policy.utils.logs.levels.error_logger import error
from lstm_eviction_policy.utils.logs.levels.info_logger import info


def plot_daily_profile(
    timestamps_hours: np.ndarray,
) -> None:
    """
    Plot daily profile of requests.

    This function plots the daily
    profile of requests, given timestamps
    in hours when they occurred.

    Parameters:
        timestamps_hours (np.ndarray): Timestamp in hours of requests.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while plotting daily profile, e.g.:
            * If timestamps list is empty, contains negative values, or non-numeric entries.
            * If timestamps data structure is not a numpy array or list-like.
            * If bin size is set to zero.
    """
    try:
        # Define the number of bins to
        # be displayed
        num_bins = int((MAX_HOUR - MIN_HOUR) / DAILY_PROFILE_PLOT_BIN_SIZE) + 1

        debug(f"Number of bins for daily profile plot: {num_bins}")

        # Define the bins ranging from predefined
        # min hour to max hour
        bins = np.linspace(MIN_HOUR, MAX_HOUR, num_bins + 1)

        debug(f"Bins for daily profile histogram: {bins}")

        # Define the histogram and get the
        # count of bins, each one of them
        # has a height proportional to the
        # number of requests occurred in that
        # hour of day
        counts, _ = np.histogram(timestamps_hours, bins=bins)

        debug(f"Counts per daily profile histogram bin: {counts}")

        # Plot daily profile
        plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
        plt.bar(
            bins[:-1],
            counts,
            width=DAILY_PROFILE_PLOT_BIN_SIZE,
            align=DAILY_PROFILE_PLOT_ALIGN,
            edgecolor=DAILY_PROFILE_PLOT_EDGE_COLOR,
        )
        plt.title(
            DAILY_PROFILE_PLOT_TITLE,
            fontsize=FIGURE_TITLE_FONT_SIZE,
        )
        plt.xlabel(
            DAILY_PROFILE_PLOT_X_LABEL,
            fontsize=FIGURE_LABEL_FONT_SIZE,
        )
        plt.ylabel(
            DAILY_PROFILE_PLOT_Y_LABEL,
            fontsize=FIGURE_LABEL_FONT_SIZE,
        )
        plt.xticks(
            np.arange(
                MIN_HOUR,
                MAX_HOUR + 1,
                step=DAILY_PROFILE_PLOT_STEP,
            ),
            fontsize=FIGURE_LABEL_FONT_SIZE,
        )
        plt.tight_layout()
        plt.show()
        plt.close()

        info("Daily profile plotted")
    except (
        ZeroDivisionError,
        ValueError,
        TypeError,
    ) as e:
        msg = "Failed to plot daily profile"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
