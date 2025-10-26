from torch.utils.data import DataLoader

from components.const import (
    TIME_SECONDS_IN_HOUR,
)
from components.dataset.rows.extractions.sliding_window_extractor import (
    extract_sliding_window_dataset_rows,
)
from components.dataset.rows.extractions.tuple_extractor import (
    extract_tuple_from_dataset_row,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.time.transforms.trig_decoder import (
    decode_time_trigonometrically,
)


def extract_last_rows_from_dataset(
    current_idx: int,
    seq_len: int,
    df: DataLoader,
    time_conversion_factor: float = TIME_SECONDS_IN_HOUR,
) -> list[tuple[float, int]] | None:
    """Extract the last rows from a dataset.

    This function extracts the last rows from a dataset, returning
    them as a list of tuples (hour, key).

    Args:
        current_idx (int): Current dataset index.
        seq_len (int): Sequence length to extract.
        df (DataLoader): Dataset to extract rows from.
        time_conversion_factor (float): Factor to convert extracted seconds to
                                        desired unit.

    Returns:
        list[tuple[float, int]] | None: Each tuple is (hour, key),
                                        None if no enough data is
                                        available.

    Raises:
        RuntimeError: If last rows extraction from dataset fails:
            * Invalid arguments or data types
              (AttributeError, TypeError, ValueError)
            * DataLoader does not contain attribute data (AttributeError)
            * Dataset rows cannot be iterated (TypeError)
            * Row tuples cannot be decoded from features (ValueError)
    """
    try:
        # Extract a sliding window from dataset
        # of sequence length size
        window_df = extract_sliding_window_dataset_rows(
            df.data,
            current_idx,
            seq_len,
        )

        # Check whether the extracted window is None
        if not window_df:
            debug(
                "Insufficient data to extract last rows",
                extra={
                    "current_idx": current_idx,
                    "seq_len": seq_len,
                    "available_rows": (
                        len(df.data) if hasattr(df, "data") else None
                    ),
                    "context": "Dataset last rows extraction",
                },
            )
            return None

        # Extract last rows (hour and key) from window
        last_rows = []
        for _, row in window_df.iterrows():
            # Extract tuple from current row
            x_features, y_key = extract_tuple_from_dataset_row(row)

            # Decode trigonometrically the time
            # extracted from features
            sin_time, cos_time = x_features
            current_time = decode_time_trigonometrically(sin_time, cos_time)

            # Store the current row as a couple (hour, key)
            last_rows.append(
                (current_time / time_conversion_factor, y_key.item()),
            )

        return last_rows
    except AttributeError as e:
        msg = "Dataset last rows extraction failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "current_idx": current_idx,
                "seq_len": seq_len,
                "available_rows": (
                    len(df.data) if hasattr(df, "data") else None
                ),
                "context": "Dataset last rows extraction",
            },
        )
        raise RuntimeError(msg) from e
