from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def create_dataframe(
    columns: Dict[str, Union[Sequence, np.ndarray]],
) -> pd.DataFrame:
    """
    Create dataframe for the given columns.

    This function creates a Pandas dataframe
    from the given columns.

    Parameters:
        columns (Dict[str, Union[Sequence, np.ndarray]]): Columns to create
                                                          dataframe for.

    Returns:
        pd.DataFrame: Pandas dataframe created from the given columns.

    Raises:
        RuntimeError: If an error occurs while creating the DataFrame, e.g.:
            * The columns have not the same length.
            * The columns have a format that cannot be
              converted to create a Pandas DataFrame.
    """
    debug(f"Columns number of dataframe to be created: {len(columns)}")
    debug(
        f"Amount of data for dataframe to be created: {sum(len(v) for v in columns.values())}"
    )

    try:
        # Create dataframe for the
        # given columns
        df = pd.DataFrame(columns)
    except (ValueError, TypeError) as e:
        msg = "Failed to create dataframe"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(
        f"Dataframe created with {len(df)} rows and {len(df.columns)} columns"
    )

    return df
