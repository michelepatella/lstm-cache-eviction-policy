import numpy as np
from matplotlib import pyplot as plt

from const import (
    FIGURE_LABEL_FONT_SIZE,
    FIGURE_SIZE,
    FIGURE_TITLE_FONT_SIZE,
    KEY_USAGE_HEATMAP_ASPECT,
    KEY_USAGE_HEATMAP_CMAP,
    KEY_USAGE_HEATMAP_COLORBAR_LABEL,
    KEY_USAGE_HEATMAP_ROTATION,
    KEY_USAGE_HEATMAP_TITLE,
    KEY_USAGE_HEATMAP_X_LABEL,
    KEY_USAGE_HEATMAP_Y_LABEL,
    MAX_HOUR,
    MIN_HOUR,
)
from lstm_eviction_policy.config.classes.Config import (
    Config,
)
from lstm_eviction_policy.utils.logs.levels.debug_logger import debug
from lstm_eviction_policy.utils.logs.levels.error_logger import error
from lstm_eviction_policy.utils.logs.levels.info_logger import info


def plot_key_usage_heatmap(
    requests: list[int],
    timestamps_hours: np.ndarray,
    config: Config,
) -> None:
    """
    Plot key usage heatmap over hours of the day.

    This function generates a heatmap showing the frequency
    of accesses for each key across the hours of the day.

    Parameters:
        requests (list[int]): List of key accesses.
        timestamps_hours (np.ndarray): Array of request timestamps in hours.
        config (Config): Configuration object.

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs while plotting key usage heatmap, e.g.:
            * If requests or timestamps data contain invalid
              values or mismatched lengths.
            * If requests data is not iterable or timestamps data is not a
              NumPy array of numeric values.
    """
    try:
        # Retrieve both min and max keys
        min_key = config.data.general.keys.min
        max_key = config.data.general.keys.max

        # Initialize heatmap
        num_hours = MAX_HOUR - MIN_HOUR + 1
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
                f"Current request: {current_request}, hour: {current_timestamp_hour_int}, index in heatmap: {requested_key_idx}"
            )

            # Check whether the current timestamp is
            # within the predefined range time and
            # corresponding requested key is also
            # within the predefined range
            if (
                MIN_HOUR <= current_timestamp_hour_int < MIN_HOUR + num_hours
            ) and (0 <= requested_key_idx < num_keys):
                # Increment the current heatmap
                # position by one
                heatmap[
                    current_timestamp_hour_int - MIN_HOUR,
                    requested_key_idx,
                ] += 1

        # Plot key usage heatmap
        plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
        plt.imshow(
            heatmap,
            aspect=KEY_USAGE_HEATMAP_ASPECT,
            cmap=KEY_USAGE_HEATMAP_CMAP,
        )
        plt.title(
            KEY_USAGE_HEATMAP_TITLE,
            fontsize=FIGURE_TITLE_FONT_SIZE,
        )
        plt.colorbar(label=KEY_USAGE_HEATMAP_COLORBAR_LABEL)
        plt.xlabel(
            KEY_USAGE_HEATMAP_X_LABEL,
            fontsize=FIGURE_LABEL_FONT_SIZE,
        )
        plt.ylabel(
            KEY_USAGE_HEATMAP_Y_LABEL,
            fontsize=FIGURE_LABEL_FONT_SIZE,
        )
        plt.yticks(
            ticks=np.arange(num_hours),
            labels=[f"{h}:00" for h in range(MIN_HOUR, MAX_HOUR + 1)],
            fontsize=FIGURE_LABEL_FONT_SIZE,
        )
        plt.xticks(
            ticks=np.arange(num_keys),
            labels=np.arange(min_key, max_key + 1),
            rotation=KEY_USAGE_HEATMAP_ROTATION,
            fontsize=FIGURE_LABEL_FONT_SIZE,
        )
        plt.tight_layout()
        plt.show()
        plt.close()

        info("Key usage heatmap plotted")
    except (TypeError, ValueError) as e:
        msg = "Failed to plot key usage heatmap"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
