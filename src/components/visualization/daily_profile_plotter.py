import numpy as np
from matplotlib import pyplot as plt

from components.const import (
    PLOT_DAILY_PROFILE_BIN_SIZE,
    PLOT_DAILY_PROFILE_STEP,
    PLOT_DAILY_PROFILE_TITLE,
    PLOT_DAILY_PROFILE_X_LABEL,
    PLOT_DAILY_PROFILE_Y_LABEL,
    PLOT_LABEL_FONT_SIZE,
    PLOT_SIZE,
    PLOT_TITLE_FONT_SIZE,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from src.const import (
    TIME_END_HOUR,
    TIME_START_HOUR,
)


def plot_daily_profile(timestamps_hours: np.ndarray, save_path: str) -> None:
    """Plot daily profile of requests.

    This function plots the daily profile of requests, given timestamps
    in hours when they occurred.

    Args:
        timestamps_hours (np.ndarray): Timestamp in hours of requests.
        save_path (str): Path to save the figure.

    Returns:
        None

    Raises:
        RuntimeError: If plotting the daily profile fails:
            * Division by zero occurs due to bin size equals zero (ZeroDivisionError).
            * Input arrays or constants are not numeric (TypeError).
            * Histogram or bar plot fails due to dimension mismatch (ValueError).
    """
    try:
        # Define the number of bins to
        # be displayed
        num_bins = (
            int(
                (TIME_END_HOUR - TIME_START_HOUR)
                / PLOT_DAILY_PROFILE_BIN_SIZE,
            )
            + 1
        )

        # Define the bins ranging from predefined
        # min hour to max hour
        bins = np.linspace(
            TIME_START_HOUR,
            TIME_END_HOUR,
            num_bins + 1,
        )

        # Define the histogram and get the
        # count of bins, each one of them
        # has a height proportional to the
        # number of requests occurred in that
        # hour of day
        bins_counts, _ = np.histogram(timestamps_hours, bins=bins)

        # Prepare, show, and save the plot
        plt.figure(figsize=(PLOT_SIZE, PLOT_SIZE))
        plt.bar(
            bins[:-1],
            bins_counts,
            width=PLOT_DAILY_PROFILE_BIN_SIZE,
        )
        plt.title(
            PLOT_DAILY_PROFILE_TITLE,
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        plt.xlabel(
            PLOT_DAILY_PROFILE_X_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.ylabel(
            PLOT_DAILY_PROFILE_Y_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.xticks(
            np.arange(
                TIME_START_HOUR,
                TIME_END_HOUR + 1,
                step=PLOT_DAILY_PROFILE_STEP,
            ),
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

        debug(
            "Daily profile plotted and saved",
            extra={
                "save_path": str(save_path),
                "timestamps_num": len(timestamps_hours),
                "bins_num": num_bins,
                "bins_size": PLOT_DAILY_PROFILE_BIN_SIZE,
                "x_range": (
                    TIME_START_HOUR,
                    TIME_END_HOUR,
                ),
                "context": "Daily profile plot",
            },
        )
    except (ZeroDivisionError, ValueError, TypeError) as e:
        msg = "Plotting daily profile failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "save_path": str(save_path),
                "timestamps_num": (
                    len(timestamps_hours)
                    if isinstance(timestamps_hours, np.ndarray)
                    else None
                ),
                "bins_size": PLOT_DAILY_PROFILE_BIN_SIZE,
                "x_range": (
                    TIME_START_HOUR,
                    TIME_END_HOUR,
                ),
                "context": "Daily profile plot",
            },
        )
        raise RuntimeError(msg) from e
