import math
from typing import Tuple

from torch import Tensor

from pipeline.const import HOURS_IN_DAY, SECONDS_IN_HOUR
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info
from utils.time.encoding_decoding.trig_decoder import (
    decode_time_trigonometrically,
)


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

        # Decode trigonometric time back to seconds
        current_time = decode_time_trigonometrically(sin_time, cos_time)

        # Extract the requested key
        key = y_key.item()

        debug(
            f"Current time: {current_time}, and key: {key},"
            f" extracted from row"
        )

        info("Time and key extracted from row")

        return current_time, key
    except (IndexError, AttributeError, TypeError) as e:
        msg = "Failed to extract time and key from row"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
