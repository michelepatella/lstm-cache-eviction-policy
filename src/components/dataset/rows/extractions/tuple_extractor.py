from typing import Tuple

import pandas as pd
import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import (
    COS_TIME_COLUMN_NAME,
    DATASET_REQUEST_COLUMN_NAME,
    SIN_TIME_COLUMN_NAME,
)


def extract_tuple_from_dataset_row(
    row: pd.Series,
    feature_columns: Tuple[str] = (COS_TIME_COLUMN_NAME, SIN_TIME_COLUMN_NAME),
    target_column: str = DATASET_REQUEST_COLUMN_NAME,
    feature_dtype: torch.dtype = torch.float,
    target_dtype: torch.dtype = torch.long,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a tuple from a dataset row.

    This function extracts specified feature(s) and target
    from a dataset row, returning the resulting tuple of
    tensors according to specified types.

    Args:
        row (pd.Series): A dataset row.
        feature_columns (Tuple[str]): Names of the feature column(s) to extract.
        target_column (str): Name of the target column to extract.
        feature_dtype (torch.dtype): Type for the feature tensor.
        target_dtype (torch.dtype): Type for the target tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - features: Tensor containing the extracted feature value(s) from
                        the dataset row.
            - target: Tensor containing the extracted target value from the
                      dataset row.

    Raises:
        RuntimeError: If tuple extraction from dataset row fails:
            * Feature columns missing (KeyError).
            * Target column missing (KeyError).
            * Invalid data types (TypeError).
            * Invalid values (ValueError).
    """
    try:
        # Extract feature tensor(s)
        features = torch.tensor(
            row[feature_columns].values,
            dtype=feature_dtype,
        )

        # Extract target tensor
        target = torch.tensor(
            int(row[target_column]),
            dtype=target_dtype,
        )

        debug(
            f"Feature(s): {features}, and target: {target} "
            f"extracted from dataset row"
        )

        return features, target
    except (KeyError, TypeError, ValueError) as e:
        msg = "Failed to extract tuple from dataset row"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
