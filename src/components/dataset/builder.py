from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def build_dataset(
    columns: Dict[str, Union[Sequence, np.ndarray]],
) -> pd.DataFrame:
    """
    Create dataset for the given columns.

    This function creates a Pandas dataframe
    from the given columns.

    Args:
        columns (Dict[str, Union[Sequence, np.ndarray]]): Columns to create
                                                          dataset for.

    Returns:
        pd.DataFrame: Pandas dataframe built from the given columns.

    Raises:
        RuntimeError: If dataset building fails:
            * Invalid column data type (TypeError).
            * Mismatched lengths of column values (ValueError).
    """
    try:
        debug(
            "Dataset building started",
            extra={
                "columns_provided": list(columns.keys()),
                "columns_num": len(columns),
                "context": "Dataset building",
            },
        )

        # Create dataframe for the
        # given columns
        df = pd.DataFrame(columns)

        debug(
            "Dataset building completed",
            extra={
                "rows_num": len(df),
                "columns_num": len(df.columns),
                "column_names": df.columns.tolist(),
                "context": "Dataset building",
            },
        )

        return df
    except (ValueError, TypeError) as e:
        msg = "Dataset building failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "columns_provided": (
                    list(columns.keys()) if isinstance(columns, dict) else None
                ),
                "columns_num": (
                    len(columns) if isinstance(columns, dict) else None
                ),
                "context": "Dataset building",
            },
        )
        raise RuntimeError(msg) from e
