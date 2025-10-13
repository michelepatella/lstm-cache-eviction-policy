import math
from typing import Tuple

from torch import Tensor

from const import HOURS_IN_DAY, SECONDS_IN_HOUR
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def extract_time_key_from_row(
    row: Tuple[Tensor, Tensor, Tensor],
) -> Tuple[float, int]:
    """
    Extract current time in seconds and requested
    key from a dataset row.

    This function processes a single row from the dataset, and
    calculates and returns the current time in seconds — based on the
    sinusoidal encoding — and the requested key.

    Args:
        row (Tuple[Tensor, Tensor, Tensor]): A single dataset row.

    Returns:
        Tuple[float, int]: A tuple containing the current time in
                           seconds and the requested key.

    Raises:
        RuntimeError: If an error occurs while extracting time and key
                      from dataset row e.g.:
                          * Invalid input format.
                          * Invalid input type.
    """
    try:
        # Unpack the provided row
        x_features, _, y_key = row

        # Extract sin and cos time encoding
        # from the provided row
        sin_time, cos_time = x_features[0]

        debug(
            f"Sin time: {sin_time}, cos time: {cos_time},"
            f" extracted from row"
        )

        # Compute angle in radians
        angle = math.atan2(sin_time, cos_time)

        debug(f"Raw angle extracted from row: {angle} (radians)")

        # Normalize angle to be always
        # positive
        if angle < 0:
            angle += 2 * math.pi
            debug(f"Normalized angle after row extraction: {angle} (radians)")

        # Convert to time seconds
        current_time = angle / (2 * math.pi) * HOURS_IN_DAY * SECONDS_IN_HOUR

        # Extract the requested key
        key = y_key.item()

        debug(
            f"Current time: {current_time:.2f}, and key: {key},"
            f" extracted from row"
        )

        info("Time and key extracted from row")

        return current_time, key
    except (IndexError, AttributeError, TypeError) as e:
        msg = "Failed to extract time and key from row"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
