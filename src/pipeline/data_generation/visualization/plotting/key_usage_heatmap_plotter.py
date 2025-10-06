from typing import List

import numpy as np
from matplotlib import pyplot as plt

from const import (
    PLOT_LABEL_FONT_SIZE,
    PLOT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    KEY_USAGE_HEATMAP_ASPECT,
    KEY_USAGE_HEATMAP_CMAP,
    KEY_USAGE_HEATMAP_COLORBAR_LABEL,
    KEY_USAGE_HEATMAP_ROTATION,
    KEY_USAGE_HEATMAP_TITLE,
    KEY_USAGE_HEATMAP_X_LABEL,
    KEY_USAGE_HEATMAP_Y_LABEL,
    DATA_GENERATION_FINAL_HOUR,
    DATA_GENERATION_INITIAL_HOUR,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def plot_key_usage_heatmap(
    min_key: int,
    max_key: int,
    requests: List[int],
    timestamps_hours: np.ndarray,
    save_path: str,
) -> None:
    """
    Plot key usage heatmap over hours of the day.

    This function generates a heatmap showing the frequency
    of accesses for each key across the hours of the day.

    Parameters:
        min_key (int): The lowest key.
        max_key (int): The greatest key.
        requests (List[int]): List of key accesses.
        timestamps_hours (np.ndarray): Array of request timestamps in hours.
        save_path (str): Path to save the figure.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while
                      plotting key usage heatmap, e.g.:
            * If requests or timestamps data contain invalid
              values or mismatched lengths.
            * If requests data is not iterable or timestamps data is not a
              NumPy array of numeric values.
    """
    try:
        # Initialize heatmap
        num_hours = DATA_GENERATION_FINAL_HOUR - DATA_GENERATION_INITIAL_HOUR + 1
        num_keys = max_key - min_key + 1
        heatmap = np.zeros(
            (num_hours, num_keys),
            dtype=int,
        )

        debug(f"Key usage heatmap shape (hours x keys): {heatmap.shape}")

        # Fill heatmap
        for (
            current_request,
            current_timestamp_hour,
        ) in zip(requests, timestamps_hours):
            # Force current timestamp to
            # be an integer representing hour
            current_timestamp_hour_int = int(current_timestamp_hour)

            # Get the index of the requested key
            requested_key_idx = current_request - min_key

            debug(
                f"Current request: {current_request}, "
                f"hour: {current_timestamp_hour_int}, "
                f"index in heatmap: {requested_key_idx}"
            )

            # Check whether the current timestamp is
            # within the predefined range time and
            # corresponding requested key is also
            # within the predefined range
            if (
                    DATA_GENERATION_INITIAL_HOUR <= current_timestamp_hour_int < DATA_GENERATION_INITIAL_HOUR + num_hours
            ) and (0 <= requested_key_idx < num_keys):
                # Increment the current heatmap
                # position by one
                heatmap[
                    current_timestamp_hour_int - DATA_GENERATION_INITIAL_HOUR,
                    requested_key_idx,
                ] += 1

        # Plot key usage heatmap
        plt.figure(figsize=(PLOT_SIZE, PLOT_SIZE))
        plt.imshow(
            heatmap,
            aspect=KEY_USAGE_HEATMAP_ASPECT,
            cmap=KEY_USAGE_HEATMAP_CMAP,
        )
        plt.title(
            KEY_USAGE_HEATMAP_TITLE,
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        plt.colorbar(label=KEY_USAGE_HEATMAP_COLORBAR_LABEL)
        plt.xlabel(
            KEY_USAGE_HEATMAP_X_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.ylabel(
            KEY_USAGE_HEATMAP_Y_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.yticks(
            ticks=np.arange(num_hours),
            labels=[f"{h}:00" for h in range(DATA_GENERATION_INITIAL_HOUR, DATA_GENERATION_FINAL_HOUR + 1)],
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.xticks(
            ticks=np.arange(num_keys),
            labels=np.arange(min_key, max_key + 1),
            rotation=KEY_USAGE_HEATMAP_ROTATION,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

        info(f"Key usage heatmap plotted and saved to {save_path}")
    except (TypeError, ValueError) as e:
        msg = "Failed to plot key usage heatmap"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
