import pandas as pd
import torch

from components.const import (
    DATASET_COLUMN_COS_TIME_NAME,
    DATASET_COLUMN_SIN_TIME_NAME,
)
from components.logs.levels.error_logger import error
from src.const import (
    DATASET_COLUMN_REQUEST_NAME,
)


def extract_tuple_from_dataset_row(
    row: pd.Series,
    feature_columns: list[str] = [
        DATASET_COLUMN_COS_TIME_NAME,
        DATASET_COLUMN_SIN_TIME_NAME,
    ],
    target_column: str = DATASET_COLUMN_REQUEST_NAME,
    feature_dtype: torch.dtype = torch.float,
    target_dtype: torch.dtype = torch.long,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a tuple from a dataset row.

    This function extracts specified feature(s) and target
    from a dataset row, returning the resulting tuple of
    tensors according to specified types.

    Args:
        row (pd.Series): A dataset row.
        feature_columns (tuple[str]): Names of the feature column(s)
                                      to extract.
        target_column (str): Name of the target column to extract.
        feature_dtype (torch.dtype): Type for the feature tensor.
        target_dtype (torch.dtype): Type for the target tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
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

        return features, target
    except (KeyError, TypeError, ValueError) as e:
        msg = "Dataset row tuple extraction failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "row_index": getattr(row, "name", None),
                "feature_columns_requested": feature_columns,
                "target_column_requested": target_column,
                "context": "Dataset row tuple extraction",
            },
        )
        raise RuntimeError(msg) from e
