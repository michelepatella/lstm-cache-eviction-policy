"""builder.py

Module defining a Ray remote task for parallelized feature engineering
on dataset chunks.

This module provides the `build_features_task` function, which wraps the
feature construction logic to be executed via remote tasks via Ray. This
enables efficient, parallel processing of large datasets on smaller DataFrame
segments.

Functions:
    build_features_task(df_chunk: pd.DataFrame, seq_len: int) -> pd.DataFrame
        Remote task to build and return features for a chunk of the dataset.
"""

import pandas as pd
import ray

from components.dataset.features.builder import build_features


@ray.remote
def build_features_task(df_chunk: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """Remote task to build features for a single chunk of the dataset.

    This function is executed remotely via Ray. It calls the utility to
    apply feature engineering to the input DataFrame chunk.

    Args:
        df_chunk (pd.DataFrame): A segment (chunk) of the full dataset
                                 DataFrame to be processed.
        seq_len (int): The sequence length of the model required for deriving
                       sequence-dependent features.

    Returns:
        pd.DataFrame: The DataFrame chunk after feature engineering, with
                      new features added and the target column positioned last.
    """
    return build_features(df_chunk, seq_len)
