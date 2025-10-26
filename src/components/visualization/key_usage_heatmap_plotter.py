import numpy as np
from matplotlib import pyplot as plt

from components.const import (
    PLOT_KEY_USAGE_HEATMAP_ASPECT,
    PLOT_KEY_USAGE_HEATMAP_CMAP,
    PLOT_KEY_USAGE_HEATMAP_COLORBAR_LABEL,
    PLOT_KEY_USAGE_HEATMAP_ROTATION,
    PLOT_KEY_USAGE_HEATMAP_STEP,
    PLOT_KEY_USAGE_HEATMAP_TITLE,
    PLOT_KEY_USAGE_HEATMAP_X_LABEL,
    PLOT_KEY_USAGE_HEATMAP_Y_LABEL,
    PLOT_LABEL_FONT_SIZE,
    PLOT_SIZE,
    PLOT_TITLE_FONT_SIZE,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from src.const import (
    DATA_GENERATION_FINAL_HOUR,
    DATA_GENERATION_INITIAL_HOUR,
)


def plot_key_usage_heatmap(
    min_key: int,
    max_key: int,
    requests: list[int],
    timestamps_hours: np.ndarray,
    save_path: str,
) -> None:
    """Plot key usage heatmap over hours of the day.

    This function generates a heatmap showing the frequency of
    accesses for each key across the hours of the day.

    Args:
        min_key (int): The lowest key.
        max_key (int): The greatest key.
        requests (list[int]): List of key accesses.
        timestamps_hours (np.ndarray): Array of request timestamps in hours.
        save_path (str): Path to save the figure.

    Returns:
        None

    Raises:
        RuntimeError: If plotting the key usage heatmap fails:
            * Input sequences are not valid or cannot be iterated (TypeError).
            * Values in requests or timestamps cannot be converted to int or
              used in array indexing (TypeError or ValueError).
            * NumPy operations fail due to invalid parameters (ValueError).
    """
    try:
        # Calculate number of hours to
        # show in the heatmap
        num_hours = (
            DATA_GENERATION_FINAL_HOUR - DATA_GENERATION_INITIAL_HOUR + 1
        )

        # Calculate number of keys involved
        # in the heatmap
        num_keys = max_key - min_key + 1

        # Initialize heatmap
        heatmap = np.zeros(
            (num_hours, num_keys),
            dtype=int,
        )

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

            # Check whether the current timestamp is
            # within the predefined range time and
            # corresponding requested key is also
            # within the predefined range
            if (
                DATA_GENERATION_INITIAL_HOUR
                <= current_timestamp_hour_int
                < DATA_GENERATION_INITIAL_HOUR + num_hours
            ) and (0 <= requested_key_idx < num_keys):
                # Increment the current heatmap
                # position by one
                heatmap[
                    current_timestamp_hour_int - DATA_GENERATION_INITIAL_HOUR,
                    requested_key_idx,
                ] += 1

        # Prepare, show, and save plot
        plt.figure(figsize=(PLOT_SIZE, PLOT_SIZE))
        plt.imshow(
            heatmap,
            aspect=PLOT_KEY_USAGE_HEATMAP_ASPECT,
            cmap=PLOT_KEY_USAGE_HEATMAP_CMAP,
        )
        plt.title(
            PLOT_KEY_USAGE_HEATMAP_TITLE,
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        plt.colorbar(label=PLOT_KEY_USAGE_HEATMAP_COLORBAR_LABEL)
        plt.xlabel(
            PLOT_KEY_USAGE_HEATMAP_X_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.ylabel(
            PLOT_KEY_USAGE_HEATMAP_Y_LABEL,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.yticks(
            ticks=np.arange(num_hours),
            labels=[
                f"{h}"
                for h in range(
                    DATA_GENERATION_INITIAL_HOUR,
                    DATA_GENERATION_FINAL_HOUR + 1,
                )
            ],
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.xticks(
            ticks=np.arange(0, num_keys, PLOT_KEY_USAGE_HEATMAP_STEP),
            labels=np.arange(
                min_key,
                max_key + 1,
                PLOT_KEY_USAGE_HEATMAP_STEP,
            ),
            rotation=PLOT_KEY_USAGE_HEATMAP_ROTATION,
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

        debug(
            "Key usage heatmap plotted and saved",
            extra={
                "save_path": save_path,
                "hours_num": num_hours,
                "keys_num": num_keys,
                "requests_num": len(requests),
                "timestamps_num": len(timestamps_hours),
                "heatmap_shape": heatmap.shape,
                "context": "Key usage heatmap",
            },
        )
    except (TypeError, ValueError) as e:
        msg = "Plotting key usage heatmap failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "save_path": save_path,
                "key_min": min_key,
                "key_max": max_key,
                "requests_num": (
                    len(requests) if isinstance(requests, list) else None
                ),
                "timestamps_num": (
                    len(timestamps_hours)
                    if isinstance(timestamps_hours, np.ndarray)
                    else None
                ),
                "context": "Key usage heatmap",
            },
        )
        raise RuntimeError(msg) from e
