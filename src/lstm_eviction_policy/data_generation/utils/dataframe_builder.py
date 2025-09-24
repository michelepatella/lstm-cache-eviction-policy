from typing import Sequence, Union

import numpy as np
import pandas as pd

from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    error,
    info,
)


def create_dataframe(
    columns: dict[str, Union[Sequence, np.ndarray]],
) -> pd.DataFrame:
    """
    Create dataframe for the given columns.

    This function creates a Pandas dataframe
    from the given columns.

    Parameters:
        columns (dict[str, Union[Sequence, np.ndarray]]): Columns to create
                                                          dataframe for.

    Returns:
        pd.DataFrame: Pandas dataframe created
                      from the given columns.

    Raises:
        ValueError: If columns have not the same length.
        TypeError: If columns have a format such that cannot be
                   converted for creating a Pandas dataframe.
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
