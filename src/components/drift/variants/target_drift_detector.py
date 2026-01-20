"""target_drift_detector.py

This module implements target drift detection using the Chi-Square test.

It monitors shifts in the distribution of the target variable (labels) between
historical reference data and new production data.

Functions:
    detect_target_drift(
        new_df: pd.DataFrame,
        hist_df: pd.DataFrame
    ) -> dict[str, Any]:
        Performs a Chi-Square test on the target column and returns drift results.
"""

from typing import Any

import pandas as pd
from alibi_detect.cd import ChiSquareDrift
from box import Box

from components.const import (
    DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME,
    DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME,
    DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME,
    DRIFT_DETECTION_TARGET_RESULT_PATH,
)
from components.json.io.saver import save_json
from const import DATASET_COLUMN_REQUEST_NAME


def detect_target_drift(
    new_df: pd.DataFrame,
    hist_df: pd.DataFrame,
) -> dict[str, Any]:
    """Detects drift in the target variable using the Chi-Square test.

    This method compares the categorical distribution of the target column
    in the new production data against the historical baseline. It calculates
    the p-value and distance to determine if a significant shift has occurred
    and saves the results to a JSON file.

    Args:
        new_df (pd.DataFrame): DataFrame containing the latest production data.
        hist_df (pd.DataFrame): DataFrame containing the historical reference data.

    Returns:
        dict[str, Any]: A dictionary containing the drift status, p-value,
                        and distance for the target variable.
    """
    # Extract target column both from historical and new datasets
    hist_target = hist_df[DATASET_COLUMN_REQUEST_NAME].to_numpy()
    new_target = new_df[DATASET_COLUMN_REQUEST_NAME].to_numpy()

    # Create target drift detector (Chi-Square)
    # on historical data
    target_detector = ChiSquareDrift(hist_target)

    # Use target detector on new data
    pred = Box(target_detector.predict(new_target))

    # Save target drift results
    target_drift_results = {
        DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: int(pred.data.is_drift),
        DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: float(pred.data.p_val),
        DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: float(pred.data.distance),
    }
    save_json(target_drift_results, DRIFT_DETECTION_TARGET_RESULT_PATH)

    return target_drift_results
