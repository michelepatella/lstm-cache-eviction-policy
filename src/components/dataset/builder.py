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
        debug(f"Columns number of dataset to be built: {len(columns)}")
        debug(
            f"Amount of data for dataframe to be"
            f" built: {sum(len(v) for v in columns.values())}"
        )

        # Create dataframe for the
        # given columns
        df = pd.DataFrame(columns)

        debug(
            f"Dataset built with {len(df)} rows and {len(df.columns)} columns"
        )

        return df
    except (ValueError, TypeError) as e:
        msg = "Failed to create dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
