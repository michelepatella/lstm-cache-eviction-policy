from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def create_dataset(
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
        pd.DataFrame: Pandas dataframe created from the given columns.

    Raises:
        RuntimeError: If an error occurs while creating the dataset, e.g.:
            * The columns have not the same length.
            * The columns have a format that cannot be
              converted to create a Pandas DataFrame.
    """
    debug(f"Columns number of dataset to be created: {len(columns)}")
    debug(
        f"Amount of data for dataframe to be"
        f" created: {sum(len(v) for v in columns.values())}"
    )

    try:
        # Create dataframe for the
        # given columns
        df = pd.DataFrame(columns)
    except (ValueError, TypeError) as e:
        msg = "Failed to create dataset"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Dataset created with {len(df)} rows and {len(df.columns)} columns")

    return df
